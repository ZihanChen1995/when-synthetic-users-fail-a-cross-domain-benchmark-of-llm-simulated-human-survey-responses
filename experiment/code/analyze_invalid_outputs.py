"""RQ4 invalid-output bounds and Style-C distributional EMD (cached outputs only).

  - Invalid-as-failure sensitivity: individual accuracy with invalid outputs
    counted as wrong over ALL attempts, alongside the conditional-on-valid
    accuracy, per style.
  - Style-C distributional EMD: expected |scale distance| under the predicted
    distribution vs the true answer, for ordinal questions, model vs baseline.

Run per dataset (LLM_FAULTS_DATASET). Writes analysis/<DS>/invalid_output_bounds.json.
"""
import json

import numpy as np
import pandas as pd

import config as C

MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6", "llama3.1-8b", "llama3.3-70b"]


def load_runs(model_name):
    path = C.BUILD_DIR / "llm_runs" / f"{model_name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pred_code(rec):
    if rec["style"] == "A":
        return rec["pred"] if rec["valid"] else None
    if rec["valid"] and rec["dist"]:
        return int(max(rec["dist"], key=rec["dist"].get))
    return None


def distributional_emd_and_invalid(model_name):
    """Style-C distributional EMD (expected |scale-distance| under the predicted
    distribution vs the true answer) for ordinal Qs, model vs baseline; plus an
    invalid-as-failure sensitivity for individual accuracy (invalid counted wrong)
    alongside the conditional-on-valid accuracy, per style."""
    runs = load_runs(model_name)
    bp = pd.read_parquet(C.BUILD_DIR / "baseline_preds.parquet").set_index(
        ["respondent_id", "question_id"])
    ord_qs = {q for q, m in C.QUESTIONS.items() if m["ordinal"]}

    # ---- distributional EMD (Style C only) ----
    m_emd, b_emd, n_emd = [], [], 0
    for r in runs:
        if r["style"] != "C" or r["question_id"] not in ord_qs or not r["valid"] or not r["dist"]:
            continue
        codes = list(C.ANSWER_CODES[r["question_id"]].keys())
        true = r["human_answer"]
        if true not in codes:
            continue
        md = np.array([float(r["dist"].get(str(c), 0.0)) for c in codes])
        if md.sum() <= 0:
            continue
        md = md / md.sum()
        try:
            bd_raw = json.loads(bp.loc[(r["respondent_id"], r["question_id"]), "baseline_dist"])
        except KeyError:
            continue
        bd = np.array([float(bd_raw.get(str(c), 0.0)) for c in codes])
        bd = bd / bd.sum() if bd.sum() > 0 else bd
        cc = np.array(codes, float)
        m_emd.append(float((md * np.abs(cc - true)).sum()))
        b_emd.append(float((bd * np.abs(cc - true)).sum()))
        n_emd += 1

    # ---- invalid-as-failure sensitivity ----
    inval = {}
    for style in sorted({r["style"] for r in runs}):
        srecs = [r for r in runs if r["style"] == style]
        n_all = len(srecs)
        n_valid = sum(1 for r in srecs if r["valid"])
        # conditional-on-valid accuracy
        cov_correct = 0
        for r in srecs:
            pc = _pred_code(r)
            if pc is None:
                continue
            cov_correct += int(pc == r["human_answer"])
        cov = cov_correct / n_valid if n_valid else float("nan")
        # invalid-as-failure: invalid counts as wrong over ALL attempts
        iaf = cov_correct / n_all if n_all else float("nan")
        inval[style] = {
            "n_all": n_all, "n_valid": n_valid,
            "invalid_rate": round(1 - n_valid / n_all, 4) if n_all else None,
            "acc_conditional_on_valid": round(cov, 4),
            "acc_invalid_as_failure": round(iaf, 4),
        }
    return {
        "dist_emd_styleC": {
            "n": n_emd,
            "model": round(float(np.mean(m_emd)), 4) if m_emd else None,
            "baseline": round(float(np.mean(b_emd)), 4) if b_emd else None,
        },
        "invalid_sensitivity": inval,
    }


def main():
    result = {"dataset": C.DATASET}
    print(f"\n########## INVALID-OUTPUT BOUNDS: {C.DATASET} ##########")
    for model in MODELS:
        try:
            load_runs(model)
        except FileNotFoundError:
            continue
        r = distributional_emd_and_invalid(model)
        result[model] = r
        e = r["dist_emd_styleC"]
        print(f"\n--- {model} ---")
        print(f"  Style-C distributional EMD: model {e['model']} vs base {e['baseline']} (n={e['n']})")
        for s, v in r["invalid_sensitivity"].items():
            print(f"  style {s}: invalid {v['invalid_rate']} | acc cond-on-valid {v['acc_conditional_on_valid']} "
                  f"| acc invalid-as-failure {v['acc_invalid_as_failure']}")
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / "invalid_output_bounds.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {C.ANALYSIS_DIR / 'invalid_output_bounds.json'}")


if __name__ == "__main__":
    main()
