"""Analyze robustness runs (RQ4): prediction-flip rate under perturbations.

For each model, compares the recovered answer code under 'base' (style A, canonical
order) against 'order' (reversed options) and 'persona' (style B), on the same
(respondent, question, seed). Reports:
  - flip rate: fraction of comparable predictions whose code changes
  - seed flip rate: base seed 0 vs base seed 1 (decoding-noise floor for reference)
  - accuracy under each variant (to show whether the perturbation also moves fidelity)

The 'base' predictions come from the robustness cache if present; otherwise this
falls back to the main pilot cache (style A), so base calls are not repeated.

Usage: LLM_FAULTS_DATASET=GSS python analyze_robustness.py claude-haiku-4.5 [...]
"""
import json
import sys
from collections import defaultdict

import pandas as pd

import config as C

RB = C.BUILD_DIR / "robustness_runs"
MAIN = C.BUILD_DIR / "llm_runs"


def load_records(model):
    recs = []
    p = RB / f"{model}.jsonl"
    if p.exists():
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    have_base = any(r["variant"] == "base" for r in recs)
    if not have_base:
        # fall back to the main pilot's style-A records as 'base'
        mp = MAIN / f"{model}.jsonl"
        if mp.exists():
            for l in mp.read_text().splitlines():
                if not l.strip():
                    continue
                r = json.loads(l)
                if r["style"] == "A":
                    recs.append({"variant": "base", "seed": r["seed"],
                                 "respondent_id": r["respondent_id"],
                                 "question_id": r["question_id"],
                                 "human_answer": r["human_answer"],
                                 "valid": r["valid"], "pred": r["pred"]})
    return recs


def index(recs, variant):
    d = {}
    for r in recs:
        if r["variant"] == variant and r["valid"] and r["pred"] is not None:
            d[(r["seed"], r["respondent_id"], r["question_id"])] = r["pred"]
    return d


def flip_rate(a, b):
    keys = set(a) & set(b)
    if not keys:
        return None, 0
    flips = sum(1 for k in keys if a[k] != b[k])
    return flips / len(keys), len(keys)


def accuracy(recs, variant):
    n = c = 0
    for r in recs:
        if r["variant"] == variant and r["valid"] and r["pred"] is not None:
            n += 1
            c += int(r["pred"] == r["human_answer"])
    return (c / n if n else None), n


def main():
    models = sys.argv[1:] or ["claude-haiku-4.5", "claude-sonnet-4.6"]
    out = {}
    print(f"\n===== robustness ({C.DATASET}) =====")
    print("flip rate = fraction of predictions that change under the perturbation\n")
    for m in models:
        recs = load_records(m)
        base = index(recs, "base")
        res = {}
        for variant in ("order", "persona"):
            v = index(recs, variant)
            fr, n = flip_rate(base, v)
            acc, _ = accuracy(recs, variant)
            res[variant] = {"flip_rate": round(fr, 3) if fr is not None else None,
                            "n_compared": n,
                            "variant_accuracy": round(acc, 3) if acc is not None else None}
        # seed-noise floor: base seed0 vs seed1
        b0 = {(k[1], k[2]): p for k, p in base.items() if k[0] == 0}
        b1 = {(k[1], k[2]): p for k, p in base.items() if k[0] == 1}
        sf, sn = flip_rate(b0, b1)
        bacc, _ = accuracy(recs, "base")
        res["_base_accuracy"] = round(bacc, 3) if bacc is not None else None
        res["_seed_flip_rate"] = round(sf, 3) if sf is not None else None
        out[m] = res
        print(f"--- {m} (base acc={res['_base_accuracy']}, seed-noise flip={res['_seed_flip_rate']}) ---")
        for variant in ("order", "persona"):
            r = res[variant]
            print(f"   {variant:8s}: flip_rate={r['flip_rate']}  "
                  f"(n={r['n_compared']}, variant_acc={r['variant_accuracy']})")
        print()

    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / "robustness.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {C.ANALYSIS_DIR/'robustness.json'}")


if __name__ == "__main__":
    main()
