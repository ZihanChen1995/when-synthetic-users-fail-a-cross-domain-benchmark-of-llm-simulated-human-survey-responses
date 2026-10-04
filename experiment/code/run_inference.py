"""Run LLM inference over the pilot sample and cache every call.

Design: each (model, style, seed, respondent_id, question_id) is one record.
Results append to a JSONL cache; reruns skip already-cached records, so the run
is resumable and never repeats a completed call.

Usage:
  python run_inference.py --backend api --model <model-id> \
      --model-name claude-haiku-4.5 --styles A C --seeds 0 1 [--limit N]
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import config as C
import prompts
import inference

CACHE_DIR = C.BUILD_DIR / "llm_runs"
MAX_WORKERS = 9      # concurrent requests


def load_cache(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["model_name"], r["style"], r["seed"], r["respondent_id"], r["question_id"]))
    return done


def make_client(backend, model, temperature, max_tokens):
    if backend == "api":
        return inference.APIClient(model, temperature=temperature, max_tokens=max_tokens)
    if backend == "vllm":
        return inference.VLLMClient(model, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["api", "vllm"])
    ap.add_argument("--model", required=True, help="model identifier passed to the client")
    ap.add_argument("--model-name", required=True, help="short label used in outputs")
    ap.add_argument("--styles", nargs="+", default=["A", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="cap rows/(style,seed) for smoke tests")
    args = ap.parse_args()

    pilot = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")
    if args.limit:
        pilot = pilot.groupby("question_id", group_keys=False).head(max(1, args.limit // pilot.question_id.nunique()))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{args.model_name}.jsonl"
    done = load_cache(cache_path)

    client = make_client(args.backend, args.model, args.temperature, args.max_tokens)

    # build the work list (skip cached), then fan out across workers
    jobs = []
    n_skip = 0
    for style in args.styles:
        for seed in args.seeds:
            for _, row in pilot.iterrows():
                key = (args.model_name, style, seed, row["respondent_id"], row["question_id"])
                if key in done:
                    n_skip += 1
                    continue
                jobs.append((style, seed, row))
    total = len(pilot) * len(args.styles) * len(args.seeds)
    print(f"total slots={total} cached={n_skip} to_run={len(jobs)} workers={MAX_WORKERS}", flush=True)

    def do_job(job):
        style, seed, row = job
        prompt, mapping = prompts.build_prompt(row, style)
        try:
            raw = client.generate(prompt, seed=seed)
        except Exception as e:  # noqa: BLE001
            raw = f"__ERROR__ {type(e).__name__}: {e}"
        parsed, valid = inference.PARSERS[style](raw, mapping)
        return {
            "model_name": args.model_name,
            "style": style, "seed": seed,
            "respondent_id": row["respondent_id"], "question_id": row["question_id"],
            "human_answer": int(row["answer_code"]),
            "raw": raw, "valid": valid,
            "pred": parsed if style == "A" else None,
            "dist": {str(k): v for k, v in parsed.items()} if (style == "C" and valid) else None,
        }

    lock = threading.Lock()
    n_call, n_invalid, t0 = 0, 0, time.time()
    with cache_path.open("a") as fh, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(do_job, j) for j in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_call += 1
                if not rec["valid"]:
                    n_invalid += 1
                if n_call % 100 == 0:
                    rate = n_call / (time.time() - t0)
                    print(f"  {n_call}/{len(jobs)} ({n_invalid} invalid) {rate:.1f}/s", flush=True)

    print(f"\ndone. new_calls={n_call} skipped={n_skip} invalid={n_invalid}")
    print(f"cache: {cache_path}")


if __name__ == "__main__":
    main()
