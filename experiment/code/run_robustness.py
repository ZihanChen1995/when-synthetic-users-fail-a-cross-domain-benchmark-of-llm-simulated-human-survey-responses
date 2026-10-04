"""Robustness runner (RQ4): option-order and prompt-style sensitivity.

For each model we produce single-answer (style-A-type) predictions under three
variants on the SAME pilot rows and seeds, so predictions are directly comparable:
  base     : style A, canonical option order   (the paper's main style-A run)
  order    : style A, REVERSED option order
  persona  : style B, rich natural-language persona, canonical order

We then measure, per model, the prediction-flip rate between base and each
variant (fraction of rows whose recovered answer code changes). A low flip rate =
robust; a high flip rate = the conclusion is an artifact of surface framing.

Cache-resumable via its own JSONL per model.
Usage:
  LLM_FAULTS_DATASET=GSS python run_robustness.py --backend api \
     --model <model-id> --model-name claude-haiku-4.5 \
     --variants order persona --seeds 0 1
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config as C
import prompts
import inference
from run_inference import make_client

CACHE_DIR = C.BUILD_DIR / "robustness_runs"
MAX_WORKERS = 9

# variant -> (style, reverse)
VARIANTS = {
    "base":    ("A", False),
    "order":   ("A", True),
    "persona": ("B", False),
}


def load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["variant"], r["seed"], r["respondent_id"], r["question_id"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="api", choices=["api", "vllm"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--variants", nargs="+", default=["order", "persona"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=100)
    args = ap.parse_args()

    pilot = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{args.model_name}.jsonl"
    done = load_done(cache_path)

    client = make_client(args.backend, args.model, args.temperature, args.max_tokens)

    jobs = []
    for variant in args.variants:
        style, reverse = VARIANTS[variant]
        for seed in args.seeds:
            for _, row in pilot.iterrows():
                k = (variant, seed, row["respondent_id"], row["question_id"])
                if k in done:
                    continue
                jobs.append((variant, style, reverse, seed, row))
    print(f"model={args.model_name} to_run={len(jobs)} workers={MAX_WORKERS}", flush=True)

    def do_job(job):
        variant, style, reverse, seed, row = job
        prompt, mapping = prompts.build_prompt(row, style, reverse=reverse)
        try:
            raw = client.generate(prompt, seed=seed)
        except Exception as e:  # noqa: BLE001
            raw = f"__ERROR__ {type(e).__name__}: {e}"
        parsed, valid = inference.parse_style_A(raw, mapping)
        return {
            "model_name": args.model_name, "variant": variant, "style": style,
            "reverse": reverse, "seed": seed,
            "respondent_id": row["respondent_id"], "question_id": row["question_id"],
            "human_answer": int(row["answer_code"]),
            "raw": raw, "valid": valid, "pred": parsed,
        }

    lock = threading.Lock()
    n, t0 = 0, time.time()
    with cache_path.open("a") as fh, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(do_job, j) for j in jobs]
        for f in as_completed(futs):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                n += 1
                if n % 200 == 0:
                    print(f"  {n}/{len(jobs)} {n/(time.time()-t0):.1f}/s", flush=True)
    print(f"done model={args.model_name} new={n}  cache={cache_path}", flush=True)


if __name__ == "__main__":
    main()
