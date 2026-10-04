#!/usr/bin/env python3
"""Benchmark YOUR model against the same human ground truth used in the paper.

Two steps. Neither requires our model outputs or any API key.

  1. Export the exact prompts the paper used:

         python examples/evaluate_your_model.py export --domain GSS --out prompts.jsonl

     Each line has `prompt`, plus the `respondent_id` / `question_id` / `key_map`
     you need to map the answer back to a code. Run these through your model.

  2. Score whatever your model returned:

         python examples/evaluate_your_model.py score --domain GSS --predictions preds.jsonl

     `preds.jsonl` needs one object per line with `respondent_id`, `question_id`,
     and either `pred` (an integer answer code) or `raw` (the model's text, which
     we parse with the paper's own parser). Missing rows are counted as invalid.

You are scored on the same footing as the paper's four models: individual accuracy,
accuracy minus the non-LLM baseline (the headline metric), and the demographic
over-determination index.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment" / "code"))


def _setup(domain):
    import os
    os.environ["LLM_FAULTS_DATASET"] = domain
    for m in ("config", "prompts", "inference"):
        sys.modules.pop(m, None)
    import config, prompts, inference           # noqa: E402
    return config, prompts, inference


# --------------------------------------------------------------------- export
def cmd_export(args):
    C, P, _ = _setup(args.domain)
    sample = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")

    n = 0
    with open(args.out, "w") as f:
        for _, row in sample.iterrows():
            prompt, key_map = P.build_prompt(row, style=args.style)
            f.write(json.dumps({
                "respondent_id": row["respondent_id"],
                "question_id": row["question_id"],
                "style": args.style,
                "prompt": prompt,
                "key_map": key_map,          # {answer key -> answer code}
                "n_options": int(row["n_options"]),
            }) + "\n")
            n += 1

    print(f"wrote {n:,} prompts to {args.out}  (domain={args.domain}, style={args.style})")
    print("\nRun these through your model, then write one JSON object per line with")
    print("`respondent_id`, `question_id`, and `pred` (int code) or `raw` (text):")
    print(f"\n  python examples/evaluate_your_model.py score "
          f"--domain {args.domain} --predictions preds.jsonl")


# ---------------------------------------------------------------------- score
def cmd_score(args):
    C, _, INF = _setup(args.domain)
    sample = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")
    baseline = pd.read_parquet(C.BUILD_DIR / "baseline_preds.parquet")
    truth = sample.merge(baseline, on=["respondent_id", "question_id"])

    # respondent_id MUST be read as a string -- see the note in load_benchmark.py.
    preds = pd.read_json(args.predictions, lines=True, dtype={"respondent_id": str})
    preds["respondent_id"] = preds["respondent_id"].astype(str)
    required = {"respondent_id", "question_id"}
    if not required.issubset(preds.columns):
        sys.exit(f"ERROR: predictions file needs columns {required}")
    if "pred" not in preds.columns and "raw" not in preds.columns:
        sys.exit("ERROR: predictions file needs either a `pred` or a `raw` column")

    # parse `raw` with the paper's own parser when no `pred` is supplied
    if "pred" not in preds.columns:
        parser = INF.PARSERS[args.style]
        import prompts as P
        key_maps = {}
        for _, row in sample.iterrows():
            _, km = P.build_prompt(row, style=args.style)
            key_maps[(row["respondent_id"], row["question_id"])] = km
        parsed = []
        for _, r in preds.iterrows():
            km = key_maps.get((r["respondent_id"], r["question_id"]), {})
            try:
                out = parser(r["raw"], km)
                parsed.append(out[0] if isinstance(out, tuple) else out)
            except Exception:
                parsed.append(None)
        preds["pred"] = parsed

    merged = truth.merge(preds[["respondent_id", "question_id", "pred"]],
                         on=["respondent_id", "question_id"], how="left")

    n_total = len(merged)
    valid = merged[merged.pred.notna()]
    n_valid = len(valid)
    if n_valid == 0:
        sys.exit("ERROR: no predictions matched the evaluation sample. Check your ids.")

    acc = (valid.pred.astype(int) == valid.answer_code).mean()
    base_acc = (valid.baseline_pred == valid.answer_code).mean()

    print("=" * 62)
    print(f"Your model on {args.domain}  (style {args.style})")
    print("=" * 62)
    print(f"  rows scored          {n_valid:,} / {n_total:,}")
    print(f"  invalid / missing    {1 - n_valid / n_total:.1%}")
    print(f"  accuracy             {acc:.4f}")
    print(f"  non-LLM baseline     {base_acc:.4f}   (fit on held-out humans)")
    print(f"  delta-base           {acc - base_acc:+.4f}   <-- the headline metric")
    print()
    if acc > base_acc:
        print("  You BEAT the baseline. No model in the paper did, on either domain.")
        print("  Worth checking for leakage before believing it.")
    else:
        print("  Below the baseline -- the same failure the paper reports for all")
        print("  four models it tested.")

    # ------------------------------------------------------ context: our models
    print("\n  For comparison, the paper's models (seed 0, same rows):")
    runs_dir = C.BUILD_DIR / "llm_runs"
    for p in sorted(runs_dir.glob("*.jsonl")):
        r = pd.read_json(p, lines=True, dtype={"respondent_id": str})
        cell = r[(r.style == args.style) & (r.seed == 0) & r.valid]
        if len(cell) == 0:
            continue
        codes = cell.apply(
            lambda x: x["pred"] if pd.notna(x["pred"])
            else (int(max(x["dist"], key=x["dist"].get)) if x["dist"] else None), axis=1)
        ok = codes.notna()
        a = (codes[ok] == cell.human_answer[ok]).mean()
        print(f"    {p.stem:<20} acc={a:.4f}  delta-base={a - base_acc:+.4f}")

    print("\n  Note: seed 0 only. The paper's table averages seeds 0 and 1.")
    print("  For the over-determination (stereotyping) analysis, feed your")
    print("  predictions through experiment/code/analyze_subgroups.py.")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="export the paper's prompts")
    e.add_argument("--domain", default="GSS", choices=["GSS", "WVS"])
    e.add_argument("--style", default="A", choices=["A", "B", "C"])
    e.add_argument("--out", default="prompts.jsonl")
    e.set_defaults(func=cmd_export)

    s = sub.add_parser("score", help="score your model's predictions")
    s.add_argument("--domain", default="GSS", choices=["GSS", "WVS"])
    s.add_argument("--style", default="A", choices=["A", "B", "C"])
    s.add_argument("--predictions", required=True)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
