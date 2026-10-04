"""RQ1 robustness checks on cached outputs (no new model calls).

  (1) Distance-aware individual fidelity for ORDINAL questions: exact-match
      accuracy is harsh on long scales. We add MAE and Earth Mover's Distance
      (EMD, = |CDF| distance on an ordinal scale) between the model's argmax and
      the true human answer, and compare to the naive demographic baseline's
      argmax on the same rows. Does the "below baseline" gap survive a metric
      that gives partial credit for being close?

  (2) Learned baselines. We fit two supervised predictors on the SAME held-out
      'fit' fold used for the lookup baseline -- multinomial logistic regression
      and a random-forest classifier -- on the prompt demographics, and score
      their individual accuracy on the pilot rows.

  (3) Proper scoring for the distribution prompt (Style C). Argmax throws away
      the distribution. We score the probability the model assigns to the TRUE
      human answer via log-loss (NLL) and Brier score, for the model vs. the
      baseline's conditional distribution on the same rows. Lower is better.

Run from experiment/code:  LLM_FAULTS_DATASET=GSS python analyze_rq1_robustness.py
Writes analysis/<DATASET>/rq1_robustness.json
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C

MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6", "llama3.1-8b", "llama3.3-70b"]


def load_runs(model_name):
    path = C.BUILD_DIR / "llm_runs" / f"{model_name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ----------------------------------------------------------------- (1) distance
def emd_ordinal(p_code, true_code, codes):
    """EMD between two point masses on an ordinal scale = |rank(p) - rank(true)|
    in units of scale steps (codes are the ordinal points)."""
    idx = {c: i for i, c in enumerate(codes)}
    if p_code not in idx or true_code not in idx:
        return None
    return abs(idx[p_code] - idx[true_code])


def distance_metrics(model_name):
    """MAE and EMD (in scale steps) for ordinal questions, model argmax vs human,
    alongside the baseline argmax on the same rows."""
    runs = load_runs(model_name)
    bp = pd.read_parquet(C.BUILD_DIR / "baseline_preds.parquet").set_index(
        ["respondent_id", "question_id"])
    ord_qs = {q for q, m in C.QUESTIONS.items() if m["ordinal"]}
    out = {}
    for style in sorted({r["style"] for r in runs}):
        m_ae, m_emd, b_ae, b_emd, n = [], [], [], [], 0
        for r in runs:
            if r["style"] != style or r["question_id"] not in ord_qs:
                continue
            codes = list(C.ANSWER_CODES[r["question_id"]].keys())
            # model argmax
            if r["style"] == "A":
                pc = r["pred"] if r["valid"] else None
            else:
                pc = int(max(r["dist"], key=r["dist"].get)) if (r["valid"] and r["dist"]) else None
            if pc is None:
                continue
            true = r["human_answer"]
            try:
                base = int(bp.loc[(r["respondent_id"], r["question_id"]), "baseline_pred"])
            except KeyError:
                continue
            n += 1
            m_ae.append(abs(pc - true)); b_ae.append(abs(base - true))
            m_emd.append(emd_ordinal(pc, true, codes)); b_emd.append(emd_ordinal(base, true, codes))
        if n:
            out[style] = {
                "n": n,
                "model_mae": round(float(np.mean(m_ae)), 4),
                "baseline_mae": round(float(np.mean(b_ae)), 4),
                "model_emd_steps": round(float(np.mean([x for x in m_emd if x is not None])), 4),
                "baseline_emd_steps": round(float(np.mean([x for x in b_emd if x is not None])), 4),
            }
    return out


# ----------------------------------------------------------------- (2) learned baselines
def learned_baselines():
    """Fit logistic regression + random forest on the 'fit' fold demographics,
    score individual accuracy on the pilot rows (per question, then pooled)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    long_file = {"GSS": "gss_pilot_long.parquet", "WVS": "wvs_pilot_long.parquet"}[C.DATASET]
    long_path = C.BUILD_DIR / long_file
    if not long_path.exists():
        # WVS microdata is not redistributed (WVSA terms); see docs/RAW_DATA.md.
        raise SystemExit(
            f"\n{long_file} is not present.\n\n"
            f"  This script fits the learned baselines over the full human sample, so it\n"
            f"  needs the complete {C.DATASET} table. For WVS that table is NOT shipped in\n"
            f"  this repository: the World Values Survey requires each user to accept its\n"
            f"  usage agreement, and its terms restrict redistribution of the microdata.\n"
            f"  A schema-identical 1,985-row demonstration sample is provided instead as\n"
            f"  wvs_pilot_long_SAMPLE.parquet.\n\n"
            f"  To run this script, download WVS yourself and rebuild the table:\n"
            f"      see docs/RAW_DATA.md   (~10 min, free, requires the WVS form)\n\n"
            f"  Every other analysis script runs without it, and the published output of\n"
            f"  this one is included at experiment/analysis/{C.DATASET}/rq1_robustness.json\n")
    long = pd.read_parquet(long_path)
    from sample_and_baseline import _fold_of
    long["fold"] = long["respondent_id"].map(_fold_of)
    pilot = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")

    feats = list(C.DEMOGRAPHIC_VARS)
    # treat all demographics as categorical (country, polviews, etc.) -> one-hot
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), feats)])

    res = {"logistic": {}, "random_forest": {}}
    per_q_acc = {"logistic": [], "random_forest": [], "n": []}
    for qid in C.QUESTIONS:
        tr = long[(long.fold == "fit") & (long.question_id == qid)]
        te = pilot[pilot.question_id == qid]
        if len(tr) < 50 or len(te) == 0 or tr["answer_code"].nunique() < 2:
            continue
        Xtr, ytr = tr[feats], tr["answer_code"].astype(int)
        Xte, yte = te[feats], te["answer_code"].astype(int)
        for name, clf in [
            ("logistic", LogisticRegression(max_iter=2000)),
            ("random_forest", RandomForestClassifier(n_estimators=300, random_state=C.RANDOM_SEED, n_jobs=-1)),
        ]:
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(Xtr, ytr)
            acc = float((pipe.predict(Xte) == yte.values).mean())
            res[name][qid] = {"n": int(len(te)), "acc": round(acc, 4)}
        per_q_acc["logistic"].append((len(te), res["logistic"][qid]["acc"]))
        per_q_acc["random_forest"].append((len(te), res["random_forest"][qid]["acc"]))

    def pooled(pairs):
        num = sum(n * a for n, a in pairs); den = sum(n for n, _ in pairs)
        return round(num / den, 4) if den else None
    return {
        "pooled_accuracy": {
            "logistic": pooled(per_q_acc["logistic"]),
            "random_forest": pooled(per_q_acc["random_forest"]),
        },
        "by_question": res,
    }


# ----------------------------------------------------------------- (3) proper scoring (Style C)
def proper_scoring(model_name):
    """Log-loss (NLL) and Brier on the TRUE human answer, model Style-C dist vs.
    baseline conditional dist, on the same rows (Style C only)."""
    runs = load_runs(model_name)
    bp = pd.read_parquet(C.BUILD_DIR / "baseline_preds.parquet").set_index(
        ["respondent_id", "question_id"])
    EPS = 1e-6
    m_nll, m_brier, b_nll, b_brier, n = [], [], [], [], 0
    for r in runs:
        if r["style"] != "C" or not r["valid"] or not r["dist"]:
            continue
        codes = list(C.ANSWER_CODES[r["question_id"]].keys())
        true = r["human_answer"]
        if true not in codes:
            continue
        # model distribution over codes (normalise, clip)
        md = np.array([float(r["dist"].get(str(c), 0.0)) for c in codes])
        md = md / md.sum() if md.sum() > 0 else md
        if md.sum() <= 0:
            continue
        try:
            bd_raw = json.loads(bp.loc[(r["respondent_id"], r["question_id"]), "baseline_dist"])
        except KeyError:
            continue
        bd = np.array([float(bd_raw.get(str(c), 0.0)) for c in codes])
        bd = bd / bd.sum() if bd.sum() > 0 else bd
        ti = codes.index(true)
        onehot = np.zeros(len(codes)); onehot[ti] = 1.0
        n += 1
        m_nll.append(-np.log(max(md[ti], EPS))); b_nll.append(-np.log(max(bd[ti], EPS)))
        m_brier.append(float(((md - onehot) ** 2).sum())); b_brier.append(float(((bd - onehot) ** 2).sum()))
    if not n:
        return None
    return {
        "n": n,
        "model_nll": round(float(np.mean(m_nll)), 4),
        "baseline_nll": round(float(np.mean(b_nll)), 4),
        "model_brier": round(float(np.mean(m_brier)), 4),
        "baseline_brier": round(float(np.mean(b_brier)), 4),
    }


def main():
    result = {"dataset": C.DATASET, "distance": {}, "proper_scoring": {}}
    print(f"\n########## {C.DATASET} ##########")

    print("\n=== (2) learned baselines (individual accuracy, pooled over questions) ===")
    lb = learned_baselines()
    result["learned_baselines"] = lb
    print(f"  logistic regression : {lb['pooled_accuracy']['logistic']}")
    print(f"  random forest       : {lb['pooled_accuracy']['random_forest']}")

    for model in MODELS:
        try:
            d = distance_metrics(model)
        except FileNotFoundError:
            print(f"  (skip {model}: no runs)"); continue
        result["distance"][model] = d
        result["proper_scoring"][model] = proper_scoring(model)
        print(f"\n--- {model} ---")
        print("  (1) distance-aware (ordinal Qs), model vs baseline:")
        for s, m in d.items():
            print(f"    style {s}: MAE {m['model_mae']} vs {m['baseline_mae']} | "
                  f"EMD steps {m['model_emd_steps']} vs {m['baseline_emd_steps']} (n={m['n']})")
        p = result["proper_scoring"][model]
        if p:
            print(f"  (3) Style-C proper scoring: NLL {p['model_nll']} vs base {p['baseline_nll']} | "
                  f"Brier {p['model_brier']} vs base {p['baseline_brier']} (n={p['n']})")

    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / "rq1_robustness.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {C.ANALYSIS_DIR / 'rq1_robustness.json'}")


if __name__ == "__main__":
    main()

