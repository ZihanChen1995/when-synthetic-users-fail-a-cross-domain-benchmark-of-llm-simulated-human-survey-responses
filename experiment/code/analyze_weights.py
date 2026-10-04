"""Survey-weight robustness (cached outputs only; no new model calls).

The human ground truth is estimated unweighted, but GSS and WVS
ship official design/population weights (GSS `wtssps`, WVS `W_WEIGHT`). If the
human targets shift under weighting, every fidelity comparison shifts with them.

We recompute the two human-side quantities the paper's claims rest on, weighted
and unweighted, on the FULL in-scope table, and report how far they move:

  (A) Aggregate answer distribution per question (the RQ2 target the model's
      distribution is compared to). We report the mean Jensen-Shannon divergence
      between the unweighted and the weighted human distribution across questions:
      how much the population target itself moves when weights are applied.

  (B) Human eta^2 per (question, axis) -- the RQ3 human baseline the stereotyping
      index subtracts. We report the mean |eta2_weighted - eta2_unweighted|. The
      stereotyping index is eta2_model - eta2_human; if eta2_human barely moves,
      the index (and its sign) is unaffected by weighting.

If both movements are small relative to the effects the paper reports (JS gaps of
0.04-0.38; stereotyping indices of +0.05 to +0.7), unweighted analysis is
justified for these claims.

Run per dataset (LLM_FAULTS_DATASET). Writes analysis/<DS>/weight_robustness.json.
"""
import json
import numpy as np
import pandas as pd
import pyreadstat

import config as C

WEIGHT_EPS = 1e-9


def _js(p, q):
    from scipy.spatial.distance import jensenshannon
    p = np.asarray(p, float); q = np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p = p / p.sum(); q = q / q.sum()
    return float(jensenshannon(p, q, base=2) ** 2)


def load_weights():
    """respondent_id -> survey weight, matching the long-table id scheme."""
    if C.DATASET == "GSS":
        df, _ = pyreadstat.read_dta(str(C.GSS_DTA), usecols=["year", "id", "wtssps"],
                                    encoding=C.STATA_ENCODING)
        df = df[(df["year"] >= C.YEAR_MIN) & (df["year"] <= C.YEAR_MAX)].copy()
        df["rid"] = df["year"].astype(int).astype(str) + "_" + df["id"].astype(int).astype(str)
        w = pd.to_numeric(df["wtssps"], errors="coerce")
        return dict(zip(df["rid"], w))
    else:
        df = pd.read_csv(C.WVS_CSV, usecols=["B_COUNTRY", "D_INTERVIEW", "W_WEIGHT"])
        df["rid"] = df["B_COUNTRY"].astype(int).astype(str) + "_" + df["D_INTERVIEW"].astype(int).astype(str)
        w = pd.to_numeric(df["W_WEIGHT"], errors="coerce")
        return dict(zip(df["rid"], w))


def _eta2(df, qval, axis, w):
    """Weighted eta^2 of answer_code on the categorical axis. w=None -> unweighted."""
    g = df[[axis, qval]].copy()
    g["w"] = 1.0 if w is None else df["w"].values
    g = g[g["w"].notna() & (g["w"] > 0)]
    if g[axis].nunique() < 3 or len(g) < 30:
        return None
    wsum = g["w"].sum()
    grand = (g["w"] * g[qval]).sum() / wsum
    between = within = 0.0
    for _, sub in g.groupby(axis):
        sw = sub["w"].sum()
        if sw <= 0:
            continue
        m = (sub["w"] * sub[qval]).sum() / sw
        v = (sub["w"] * (sub[qval] - m) ** 2).sum() / sw
        between += sw * (m - grand) ** 2
        within += sw * v
    tot = between + within
    return float(between / tot) if tot > 1e-9 else None


def main():
    W = load_weights()
    long = pd.read_parquet(C.BUILD_DIR / (
        "gss_pilot_long.parquet" if C.DATASET == "GSS" else "wvs_pilot_long.parquet"))
    long["w"] = long["respondent_id"].map(W)
    cov = long["w"].notna().mean()

    # (A) aggregate distribution move per question
    js_moves = []
    for qid in C.QUESTIONS:
        sub = long[long["question_id"] == qid]
        codes = sorted(sub["answer_code"].unique())
        unw = np.array([(sub["answer_code"] == c).sum() for c in codes], float)
        ww = sub[sub["w"].notna()]
        wt = np.array([ww.loc[ww["answer_code"] == c, "w"].sum() for c in codes], float)
        j = _js(unw, wt)
        if j is not None:
            js_moves.append(j)

    # (B) human eta^2 move per (question, axis), ordinal questions only
    eta_moves = []
    for qid in C.QUESTIONS:
        if not C.QUESTIONS[qid].get("ordinal", False):
            continue
        sub = long[long["question_id"] == qid].copy()
        for axis in C.SUBGROUP_VARS:
            e_u = _eta2(sub, "answer_code", axis, w=None)
            e_w = _eta2(sub, "answer_code", axis, w=True)
            if e_u is not None and e_w is not None:
                eta_moves.append(abs(e_w - e_u))

    result = {
        "dataset": C.DATASET,
        "weight_var": "wtssps" if C.DATASET == "GSS" else "W_WEIGHT",
        "weight_coverage": round(float(cov), 4),
        "aggregate_distribution": {
            "n_questions": len(js_moves),
            "mean_js_unweighted_vs_weighted": round(float(np.mean(js_moves)), 4),
            "max_js_unweighted_vs_weighted": round(float(np.max(js_moves)), 4),
        },
        "human_eta2": {
            "n_pairs": len(eta_moves),
            "mean_abs_delta": round(float(np.mean(eta_moves)), 4),
            "max_abs_delta": round(float(np.max(eta_moves)), 4),
        },
    }
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / "weight_robustness.json").write_text(json.dumps(result, indent=2))
    print(f"\n########## WEIGHT ROBUSTNESS: {C.DATASET} ##########")
    print(f"  weight var {result['weight_var']}, coverage {cov:.1%}")
    print(f"  (A) human aggregate distribution move (JS unweighted vs weighted):")
    print(f"      mean {result['aggregate_distribution']['mean_js_unweighted_vs_weighted']}, "
          f"max {result['aggregate_distribution']['max_js_unweighted_vs_weighted']}  "
          f"(cf. model JS gaps 0.04-0.38)")
    print(f"  (B) human eta^2 move (|weighted-unweighted|):")
    print(f"      mean {result['human_eta2']['mean_abs_delta']}, "
          f"max {result['human_eta2']['max_abs_delta']}  "
          f"(cf. stereotyping indices +0.05 to +0.7)")
    print(f"  wrote {C.ANALYSIS_DIR / 'weight_robustness.json'}")


if __name__ == "__main__":
    main()
