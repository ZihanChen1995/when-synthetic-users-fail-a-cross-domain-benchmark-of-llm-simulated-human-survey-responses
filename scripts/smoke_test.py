#!/usr/bin/env python3
"""Fast integrity check (~30 s, no model calls, no heavy computation).

Run this FIRST, before committing to the 30-60 min full reproduction. It answers
one question: is this checkout complete and are the dependencies importable?

    python scripts/smoke_test.py

Exits 0 if everything checks out, 1 otherwise.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiment"

# expected record counts, from the paper's runs. respondents x questions x styles x seeds.
EXPECTED_RUNS = {
    ("GSS", "llm_runs"): 3972,
    ("WVS", "llm_runs"): 5832,
    ("GSS", "robustness_runs"): 3972,
    ("WVS", "robustness_runs"): 1458,
}
MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6", "llama3.1-8b", "llama3.3-70b"]
ROBUSTNESS_MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6"]
# schemas differ by run type: llm_runs carries `dist` (the Style-C distribution),
# robustness_runs drops it and adds the perturbation fields `variant` / `reverse`.
_BASE_FIELDS = {"model_name", "style", "seed", "respondent_id", "question_id",
                "human_answer", "raw", "valid", "pred"}
RUN_FIELDS = {
    "llm_runs": _BASE_FIELDS | {"dist"},
    "robustness_runs": _BASE_FIELDS | {"variant", "reverse"},
}

failures, checks = [], 0


def check(label, ok, detail=""):
    global checks
    checks += 1
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


print("=" * 70)
print("When Synthetic Users Fail -- smoke test")
print("=" * 70)

# ------------------------------------------------------------------ 0. interpreter
# The pinned requirements need Python >= 3.11 (pandas 3.x). macOS ships 3.9 as
# `python3`, which fails at pip-install time with a confusing resolver error --
# so check here and say so plainly.
print("\n0. Interpreter")
pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
check("Python >= 3.11", sys.version_info >= (3, 11), f"running {pyver}")
if sys.version_info < (3, 11):
    print(f"\n  This interpreter is {pyver}. The pinned dependencies require 3.11+.")
    print("  On macOS the system `python3` is often 3.9 -- use a newer one, e.g.:")
    print("      python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt")
    sys.exit(1)

# ------------------------------------------------------------------ 1. dependencies
print("\n1. Dependencies")
for mod in ["pandas", "numpy", "scipy", "sklearn", "matplotlib", "pyarrow", "tabulate"]:
    try:
        m = __import__(mod)
        check(mod, True, getattr(m, "__version__", ""))
    except ImportError as e:
        check(mod, False, str(e))

if failures:
    print("\nDependencies missing. Run:  pip install -r requirements.txt")
    sys.exit(1)

import pandas as pd  # noqa: E402

# ------------------------------------------------------------------ 2. config
print("\n2. Configuration (both domains)")
sys.path.insert(0, str(EXP / "code"))
import os  # noqa: E402
for ds, n_q, n_demo in [("GSS", 10, 7), ("WVS", 16, 5)]:
    os.environ["LLM_FAULTS_DATASET"] = ds
    for mod in [m for m in list(sys.modules) if m == "config"]:
        del sys.modules[mod]
    import config as c  # noqa: E402
    ok = (len(c.QUESTIONS) == n_q and len(c.DEMOGRAPHIC_VARS) == n_demo
          and c.BUILD_DIR.exists() and c.ANALYSIS_DIR.exists())
    check(f"{ds} config", ok, f"{len(c.QUESTIONS)} questions, {len(c.DEMOGRAPHIC_VARS)} demographics")

# ------------------------------------------------------------------ 3. cached model outputs
print("\n3. Cached model outputs")
for (ds, kind), expected_n in EXPECTED_RUNS.items():
    models = MODELS if kind == "llm_runs" else ROBUSTNESS_MODELS
    for model in models:
        p = EXP / "datasets" / ds / "build" / kind / f"{model}.jsonl"
        if not p.exists():
            check(f"{ds}/{kind}/{model}", False, "missing")
            continue
        with p.open() as f:
            lines = f.readlines()
        n = len(lines)
        try:
            rec = json.loads(lines[0])
            schema_ok = set(rec.keys()) == RUN_FIELDS[kind]
        except (json.JSONDecodeError, IndexError):
            schema_ok = False
        check(f"{ds}/{kind}/{model}", n == expected_n and schema_ok,
              f"{n:,} records" + ("" if schema_ok else ", BAD SCHEMA"))

# ------------------------------------------------------------------ 4. build tables
print("\n4. Derived build tables")
BUILD_TABLES = {
    ("GSS", "gss_pilot_long.parquet"): 85_898,
    ("GSS", "pilot_sample.parquet"): 993,
    ("GSS", "baseline_preds.parquet"): 993,
    ("GSS", "splits.parquet"): 14_704,
    ("WVS", "pilot_sample.parquet"): 1_458,
    ("WVS", "baseline_preds.parquet"): 1_458,
    ("WVS", "splits.parquet"): 91_760,
    ("WVS", "wvs_pilot_long_SAMPLE.parquet"): 1_985,
}
for (ds, name), expected_rows in BUILD_TABLES.items():
    p = EXP / "datasets" / ds / "build" / name
    if not p.exists():
        check(f"{ds}/{name}", False, "missing")
        continue
    df = pd.read_parquet(p)
    check(f"{ds}/{name}", len(df) == expected_rows,
          f"{len(df):,} rows x {len(df.columns)} cols")

# The full WVS microdata table is deliberately NOT redistributed (WVSA terms).
# Present = the user rebuilt it themselves, which unlocks analyze_rq1_robustness.py on WVS.
wvs_full = EXP / "datasets" / "WVS" / "build" / "wvs_pilot_long.parquet"
if wvs_full.exists():
    n = len(pd.read_parquet(wvs_full))
    check("WVS full microdata (user-rebuilt)", n == 1_426_473, f"{n:,} rows -- analyze_rq1_robustness.py will run")
else:
    check("WVS full microdata absent (expected)", True,
          "not redistributed; analyze_rq1_robustness.py is skipped on WVS")

for ds in ["GSS", "WVS"]:
    for name in ["human_dist.json", f"{ds.lower()}_pilot_meta.json"]:
        p = EXP / "datasets" / ds / "build" / name
        check(f"{ds}/{name}", p.exists(), "" if p.exists() else "missing")

# ------------------------------------------------------------------ 5. published analysis outputs
print("\n5. Published analysis outputs")
ANALYSIS_FILES = ["summary_table.csv", "summary_table.md", "rq1_robustness.json",
                  "supplementary_metrics.json", "invalid_output_bounds.json",
                  "robustness.json", "appendix_robustness.json", "decision_impact.json",
                  "weight_robustness.json"]
for ds in ["GSS", "WVS"]:
    missing = [f for f in ANALYSIS_FILES if not (EXP / "analysis" / ds / f).exists()]
    check(f"{ds} metrics ({len(ANALYSIS_FILES)} files)", not missing,
          "" if not missing else f"missing {missing}")
    figs = sorted((EXP / "analysis" / ds).glob("fig_*.png"))
    check(f"{ds} figures", len(figs) == 6, f"{len(figs)} figures")

# ------------------------------------------------------------------ 6. id round-trip
# Guards a real trap: WVS respondent ids like "20_20070915" are silently coerced to
# int by pandas (Python's int parser accepts "_" as a digit separator). If that ever
# regresses, joins between llm_runs and the parquet tables break.
print("\n6. Respondent-id join integrity")
for ds in ["GSS", "WVS"]:
    runs = pd.read_json(EXP / "datasets" / ds / "build" / "llm_runs" / "claude-haiku-4.5.jsonl",
                        lines=True, dtype={"respondent_id": str})
    sample = pd.read_parquet(EXP / "datasets" / ds / "build" / "pilot_sample.parquet")
    overlap = set(runs.respondent_id) & set(sample.respondent_id.astype(str))
    check(f"{ds} llm_runs joins to pilot_sample", len(overlap) == sample.respondent_id.nunique(),
          f"{len(overlap):,} of {sample.respondent_id.nunique():,} respondents match")

# ------------------------------------------------------------------ 7. reference outputs
print("\n7. Reference outputs (for scripts/verify.py)")
for ds in ["GSS", "WVS"]:
    p = ROOT / "expected" / ds / "summary_table.csv"
    if not p.exists():
        check(f"expected/{ds}/summary_table.csv", False, "missing")
        continue
    df = pd.read_csv(p)
    check(f"expected/{ds}/summary_table.csv", len(df) == 8, f"{len(df)} rows")

# ------------------------------------------------------------------ summary
print("\n" + "=" * 70)
if failures:
    print(f"FAILED -- {len(failures)} of {checks} checks did not pass:")
    for f in failures:
        print(f"    - {f}")
    print("\nSee docs/REPRODUCE.md for what each artifact should contain.")
    sys.exit(1)
print(f"OK -- all {checks} checks passed. This checkout is complete.")
print("\nNext:  make reproduce   (30-60 min, regenerates every number in the paper)")
print("       make verify      (<1 min, checks the main table against the paper)")
print("=" * 70)
