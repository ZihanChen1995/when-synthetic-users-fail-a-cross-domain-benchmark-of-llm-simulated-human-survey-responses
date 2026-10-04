"""Compute pilot metrics from cached LLM runs.

Reads datasets/<DS>/build/llm_runs/<model>.jsonl and produces, per (model, style):
  - individual accuracy  (style A) vs. the naive demographic baseline
  - aggregate fidelity    Jensen-Shannon divergence between the model's answer
                          distribution and the human distribution, per question
  - invalid/refusal rate

For style C (distributions) the individual "prediction" is the argmax of the
returned distribution; aggregate fidelity uses the mean predicted distribution.

Outputs analysis/<DS>/pilot_metrics_<model>.json and prints a summary table.
Subgroup structure is computed separately in analyze_subgroups.py.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

import config as C


def load_runs(model_name):
    path = C.BUILD_DIR / "llm_runs" / f"{model_name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def human_dist():
    return json.loads((C.BUILD_DIR / "human_dist.json").read_text())


def baseline_preds():
    return pd.read_parquet(C.BUILD_DIR / "baseline_preds.parquet")


def _pred_code(rec):
    """Single predicted code for a record (argmax for style C)."""
    if rec["style"] == "A":
        return rec["pred"]
    if rec["dist"]:
        return int(max(rec["dist"], key=rec["dist"].get))
    return None


def _model_dist_for_question(recs, codes):
    """Aggregate model answer distribution over records for one question."""
    counts = defaultdict(float)
    n = 0
    for r in recs:
        if r["style"] == "A":
            if r["valid"]:
                counts[r["pred"]] += 1; n += 1
        else:  # style C: average the returned distributions
            if r["valid"] and r["dist"]:
                for k, v in r["dist"].items():
                    counts[int(k)] += v
                n += 1
    if n == 0:
        return None
    return np.array([counts[c] / n for c in codes])


def js(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() else p; q = q / q.sum() if q.sum() else q
    return float(jensenshannon(p, q, base=2) ** 2)  # JS divergence (squared JS distance)


def compute(model_name):
    runs = load_runs(model_name)
    H = human_dist()
    bp = baseline_preds().set_index(["respondent_id", "question_id"])

    out = {}
    styles = sorted({r["style"] for r in runs})
    for style in styles:
        srecs = [r for r in runs if r["style"] == style]
        # ---- validity ----
        invalid_rate = 1 - np.mean([r["valid"] for r in srecs])

        # ---- individual accuracy (model vs baseline) on same rows ----
        correct, base_correct, n = 0, 0, 0
        for r in srecs:
            pc = _pred_code(r)
            if pc is None:
                continue
            n += 1
            correct += int(pc == r["human_answer"])
            try:
                base = bp.loc[(r["respondent_id"], r["question_id"]), "baseline_pred"]
                base_correct += int(int(base) == r["human_answer"])
            except KeyError:
                pass
        acc = correct / n if n else float("nan")
        base_acc = base_correct / n if n else float("nan")

        # ---- aggregate JS divergence per question ----
        js_by_q = {}
        for qid in C.QUESTIONS:
            codes = H[qid]["codes"]
            qrecs = [r for r in srecs if r["question_id"] == qid]
            mdist = _model_dist_for_question(qrecs, codes)
            if mdist is None:
                continue
            hdist = np.array([H[qid]["overall"][str(c)] for c in codes])
            js_by_q[qid] = js(mdist, hdist)

        out[style] = {
            "n_predictions": n,
            "invalid_rate": round(invalid_rate, 4),
            "individual_accuracy": round(acc, 4),
            "baseline_accuracy": round(base_acc, 4),
            "acc_minus_baseline": round(acc - base_acc, 4),
            "js_divergence_by_question": {k: round(v, 4) for k, v in js_by_q.items()},
            "mean_js_divergence": round(float(np.mean(list(js_by_q.values()))), 4) if js_by_q else None,
        }
    return out


def main():
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4.5"
    res = compute(model_name)
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / f"pilot_metrics_{model_name}.json").write_text(json.dumps(res, indent=2))
    print(f"\n===== {model_name} =====")
    for style, m in res.items():
        print(f"\n--- style {style} (n={m['n_predictions']}, invalid={m['invalid_rate']:.1%}) ---")
        print(f"  individual acc : {m['individual_accuracy']:.3f}   "
              f"baseline : {m['baseline_accuracy']:.3f}   "
              f"delta : {m['acc_minus_baseline']:+.3f}")
        print(f"  mean JS div    : {m['mean_js_divergence']}")
        print(f"  JS by question : {m['js_divergence_by_question']}")


if __name__ == "__main__":
    main()
