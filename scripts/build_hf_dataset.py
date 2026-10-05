"""Build the Hugging Face release of the benchmark from the files in this repository.

Writes a self-contained dataset folder (parquet tables + dataset card) that loads with
`datasets.load_dataset(<folder>, <config>)`. Nothing is edited by hand on the Hub: the
release is always regenerated from this script, and the card records the data commit.

    python scripts/build_hf_dataset.py --domain GSS --out hf_release/GSS

Tables (one `datasets` config each):
  benchmark            respondent x question rows with prompts built in   (split: test)
  questions            question text, topic, answer options               (split: train)
  model_outputs        cached calls, styles A/C, seeds 0/1                (split: train)
  robustness_outputs   option-order and persona perturbations             (split: train)
  human_responses      all human answers, GSS only                        (splits: fit, eval)
  human_distributions  observed answer distributions, overall + subgroup  (split: train)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_pipeline(domain):
    """Import the paper's config/prompts modules for `domain` (they read the env at import)."""
    os.environ["LLM_FAULTS_DATASET"] = domain
    sys.path.insert(0, str(ROOT / "experiment" / "code"))
    import config
    import prompts
    return config, prompts


def _read_jsonl(path):
    # json.loads line by line, as the pipeline does: keeps respondent_id a string
    with open(path) as f:
        return [json.loads(line) for line in f]


def _dist_list(d):
    """{"1": 0.7, "2": 0.3} -> [{"code": 1, "prob": 0.7}, ...] (Arrow-friendly, sorted)."""
    if d is None:
        return None
    if isinstance(d, str):
        d = json.loads(d)
    return [{"code": int(k), "prob": float(v)} for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))]


def _pred_final(rec):
    """Single predicted code, mirroring metrics._pred_code (argmax of dist for style C)."""
    if rec["style"] != "C":
        return rec["pred"]
    if rec.get("dist"):
        return int(max(rec["dist"], key=rec["dist"].get))
    return None


def _demographic_columns(df, C):
    """Replace coded demographics with labels; keep the codes as <var>_code."""
    out = df.copy()
    for v in C.DEMOGRAPHIC_VARS:
        if v == "age":                       # continuous; 89 means "89 or older"
            continue
        out[f"{v}_code"] = out[v].astype(int)
        out[v] = out[v].astype(int).map(lambda c, v=v: C.DEMOGRAPHIC_CODES[v].get(c, str(c)))
    return out


def _options(C, prompts, qid):
    _, mapping, _ = prompts._options_block(qid)
    labels = C.ANSWER_CODES[qid]
    return [{"key": k, "code": int(c), "label": str(labels[c])} for k, c in mapping.items()]


def build_benchmark(C, prompts, b):
    s = pd.read_parquet(b / "pilot_sample.parquet")
    bp = pd.read_parquet(b / "baseline_preds.parquet")
    df = s.merge(bp, on=["respondent_id", "question_id"], how="left", validate="1:1")
    assert df["baseline_pred"].notna().all(), "every benchmark row needs a baseline prediction"

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "prompt_a": prompts.build_prompt(r, "A")[0],
            "prompt_b": prompts.build_prompt(r, "B")[0],
            "prompt_c": prompts.build_prompt(r, "C")[0],
            "options": _options(C, prompts, r["question_id"]),
            "baseline_dist": _dist_list(r["baseline_dist"]),
        })
    extra = pd.DataFrame(rows, index=df.index)

    demo = _demographic_columns(df, C)
    lead = ["respondent_id"] + (["year"] if "year" in df else []) + ["question_id", "topic", "ordinal", "question_text"]
    demo_cols = [c for v in C.DEMOGRAPHIC_VARS for c in (v, f"{v}_code") if c in demo]
    out = pd.concat([demo[lead + demo_cols], extra[["options"]]], axis=1)
    out["n_options"] = df["n_options"].astype(int)
    out["human_answer"] = df["answer_code"].astype(int)
    out["human_answer_label"] = df["answer_text"].astype(str)
    out["baseline_pred"] = df["baseline_pred"].astype(int)
    out["baseline_backoff"] = df["baseline_backoff"]
    out["baseline_dist"] = extra["baseline_dist"]
    for k in ("prompt_a", "prompt_b", "prompt_c"):
        out[k] = extra[k]
    return out


def build_questions(C, prompts):
    return pd.DataFrame([{
        "question_id": qid,
        "topic": q["topic"],
        "ordinal": bool(q["ordinal"]),
        "question_text": q["text"],
        "options": _options(C, prompts, qid),
    } for qid, q in C.QUESTIONS.items()])


def build_model_outputs(b):
    rows = []
    for p in sorted((b / "llm_runs").glob("*.jsonl")):
        for r in _read_jsonl(p):
            pf = _pred_final(r)
            rows.append({
                "model": r["model_name"], "style": r["style"], "seed": int(r["seed"]),
                "respondent_id": r["respondent_id"], "question_id": r["question_id"],
                "raw": r["raw"], "valid": bool(r["valid"]),
                "pred": r["pred"], "dist": _dist_list(r.get("dist")),
                "pred_final": pf, "human_answer": int(r["human_answer"]),
                "correct": None if pf is None else bool(pf == r["human_answer"]),
            })
    df = pd.DataFrame(rows)
    df["pred"] = df["pred"].astype("Int64")
    df["pred_final"] = df["pred_final"].astype("Int64")
    df["correct"] = df["correct"].astype("boolean")
    return df


def build_robustness_outputs(b):
    rows = []
    for p in sorted((b / "robustness_runs").glob("*.jsonl")):
        for r in _read_jsonl(p):
            rows.append({
                "model": r["model_name"], "variant": r["variant"], "style": r["style"],
                "seed": int(r["seed"]), "reverse": bool(r["reverse"]),
                "respondent_id": r["respondent_id"], "question_id": r["question_id"],
                "raw": r["raw"], "valid": bool(r["valid"]), "pred": r["pred"],
                "human_answer": int(r["human_answer"]),
                "correct": None if r["pred"] is None else bool(r["pred"] == r["human_answer"]),
            })
    df = pd.DataFrame(rows)
    df["pred"] = df["pred"].astype("Int64")
    df["correct"] = df["correct"].astype("boolean")
    return df


def build_human_responses(C, b, domain):
    """Full long-format human table split by the paper's fit/eval respondent folds."""
    long_path = b / f"{domain.lower()}_pilot_long.parquet"
    if not long_path.exists():               # WVS microdata is not redistributed
        return None
    df = pd.read_parquet(long_path)
    folds = pd.read_parquet(b / "splits.parquet")
    df = df.merge(folds, on="respondent_id", how="left", validate="m:1")
    assert df["fold"].isin(["fit", "eval"]).all(), "every respondent must have a fold"
    df = _demographic_columns(df, C)
    lead = ["respondent_id"] + (["year"] if "year" in df else []) + ["question_id", "topic"]
    demo_cols = [c for v in C.DEMOGRAPHIC_VARS for c in (v, f"{v}_code") if c in df]
    out = df[lead + demo_cols].copy()
    out["human_answer"] = df["answer_code"].astype(int)
    out["human_answer_label"] = df["answer_text"].astype(str)
    out["fold"] = df["fold"]
    return out


def build_human_distributions(C, b):
    hd = json.load(open(b / "human_dist.json"))
    rows = []
    for qid, q in hd.items():
        rows.append({"question_id": qid, "group_var": "overall", "group_code": None,
                     "group_label": "overall", "n": int(q["n"]), "dist": _dist_list(q["overall"])})
        for key, cell in q["subgroups"].items():
            var, code = key.split("=")
            rows.append({"question_id": qid, "group_var": var, "group_code": int(code),
                         "group_label": C.DEMOGRAPHIC_CODES.get(var, {}).get(int(code), code),
                         "n": int(cell["n"]), "dist": _dist_list(cell["dist"])})
    df = pd.DataFrame(rows)
    df["group_code"] = df["group_code"].astype("Int64")
    return df


def _write(df, out, name, split):
    path = out / "data" / name / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="GSS", choices=["GSS", "WVS"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    C, prompts = _load_pipeline(args.domain)
    b = C.BUILD_DIR
    out = args.out
    if out.exists():
        shutil.rmtree(out)

    tables = {
        ("benchmark", "test"): build_benchmark(C, prompts, b),
        ("questions", "train"): build_questions(C, prompts),
        ("model_outputs", "train"): build_model_outputs(b),
        ("robustness_outputs", "train"): build_robustness_outputs(b),
        ("human_distributions", "train"): build_human_distributions(C, b),
    }
    hr = build_human_responses(C, b, args.domain)
    if hr is not None:
        for fold in ("fit", "eval"):
            tables[("human_responses", fold)] = hr[hr["fold"] == fold].drop(columns="fold")

    for (name, split), df in tables.items():
        _write(df, out, name, split)
        print(f"  {name:20s} {split:5s} {len(df):>7,} rows")

    card = (ROOT / "scripts" / "hf_cards" / f"{args.domain}.md").read_text()
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--", "experiment/datasets"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    (out / "README.md").write_text(card.replace("{{DATA_COMMIT}}", commit))
    print(f"wrote {out}  (data commit {commit})")


if __name__ == "__main__":
    main()
