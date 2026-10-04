#!/usr/bin/env python3
"""Load the benchmark and reproduce the paper's headline finding in ~40 lines.

    python examples/load_benchmark.py            # GSS
    python examples/load_benchmark.py --domain WVS

Shows how to get at the three things you need: the evaluation sample (personas +
questions + human ground truth), the non-LLM baseline, and our four models'
cached outputs. See docs/DATA.md for the full schema.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6", "llama3.1-8b", "llama3.3-70b"]


def load_eval_sample(domain: str) -> pd.DataFrame:
    """The respondent x question rows the models were asked about.

    Includes the real respondent's demographics and their true answer
    (`answer_code`). This is what you prompt from if you bring your own model.
    """
    return pd.read_parquet(ROOT / f"experiment/datasets/{domain}/build/pilot_sample.parquet")


def load_baseline(domain: str) -> pd.DataFrame:
    """Non-LLM baseline predictions, fit on held-out humans (the `fit` fold)."""
    return pd.read_parquet(ROOT / f"experiment/datasets/{domain}/build/baseline_preds.parquet")


# WARNING: respondent_id must be read as a string. WVS ids look like "20_20070915",
# and Python's int parser treats "_" as a digit separator -- so a bare
# pd.read_json(...) silently turns that id into the integer 2020070915, which then
# fails to join against the parquet tables. Always pass dtype={"respondent_id": str}.
ID_DTYPE = {"respondent_id": str}


def load_model_runs(domain: str, model: str) -> pd.DataFrame:
    """Cached model outputs. One row per (respondent, question, style, seed) call."""
    return pd.read_json(
        ROOT / f"experiment/datasets/{domain}/build/llm_runs/{model}.jsonl",
        lines=True, dtype=ID_DTYPE)


def predicted_code(row) -> float:
    """The single predicted answer code for one record.

    IMPORTANT: Style A puts the answer in `pred`. Style C returns a probability
    distribution in `dist` and leaves `pred` null -- the individual prediction is
    the argmax of that distribution. This mirrors `metrics._pred_code`; scoring
    Style C on `pred` alone silently yields zero accuracy.
    """
    if pd.notna(row["pred"]):
        return row["pred"]
    if row["dist"]:
        return int(max(row["dist"], key=row["dist"].get))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="GSS", choices=["GSS", "WVS"])
    args = ap.parse_args()
    d = args.domain

    sample = load_eval_sample(d)
    baseline = load_baseline(d)

    print(f"\n{d}: {len(sample):,} evaluation rows "
          f"({sample.respondent_id.nunique():,} respondents x "
          f"{sample.question_id.nunique()} questions)\n")

    # --- the baseline everyone has to beat -------------------------------------
    merged = sample.merge(baseline, on=["respondent_id", "question_id"])
    base_acc = (merged.baseline_pred == merged.answer_code).mean()
    print(f"  non-LLM baseline accuracy: {base_acc:.4f}")
    print(f"  (backoff levels used: {merged.baseline_backoff.value_counts().to_dict()})\n")

    # --- how the four benchmarked models did -----------------------------------
    print(f"  {'model':<20} {'style':<6} {'valid':>7} {'acc':>8} {'vs baseline':>12}")
    print("  " + "-" * 58)
    for model in MODELS:
        runs = load_model_runs(d, model)
        for style in ["A", "C"]:
            cell = runs[(runs.style == style) & (runs.seed == 0)]
            ok = cell[cell.valid]
            if len(ok) == 0:
                continue
            acc = (ok.apply(predicted_code, axis=1) == ok.human_answer).mean()
            valid_rate = len(ok) / len(cell)
            flag = "  <-- unusable" if valid_rate < 0.6 else ""
            print(f"  {model:<20} {style:<6} {valid_rate:>6.1%} {acc:>8.4f} "
                  f"{acc - base_acc:>+12.4f}{flag}")

    print("\n  Style A = pick one option. Style C = give a probability distribution.")
    print("  Negative 'vs baseline' is the paper's headline: the LLM loses to a")
    print("  simple demographic lookup fit on held-out humans.")
    print("\n  Note: this uses seed 0 only, so numbers differ slightly from the")
    print("  paper's table, which averages seeds 0 and 1. Run `make verify` for")
    print("  the exact published values.\n")


if __name__ == "__main__":
    main()
