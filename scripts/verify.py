#!/usr/bin/env python3
"""Regenerate the paper's main results table and diff it against the published values.

This is the strongest single claim this repository can make: run it, and if it
prints OK, the 16 rows of the paper's main table have been re-derived on your
machine from the cached model outputs and match the published numbers exactly.

    python scripts/verify.py            # both domains
    python scripts/verify.py --domain GSS

Takes under a minute. Runs `summarize.py` for each domain, then compares the regenerated
`experiment/analysis/<DS>/summary_table.csv` against the frozen copy in
`expected/<DS>/summary_table.csv`.

Exits 0 on an exact match, 1 on any discrepancy.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiment" / "code"

# columns compared, and how they map onto the paper's main table
COLUMN_MAP = {
    "indiv_acc": "Acc",
    "baseline_acc": "Base",
    "acc_minus_baseline": "d_base",
    "mean_js_div": "JS",
    "median_stereotyping_index": "d_eta2",
    "n_pairs_ci_excl_0": "sig",
    "n_pairs_total": "n_pairs",
    "invalid_rate": "invalid",
    "n": "n",
}
KEYS = ["model", "style"]
TOLERANCE = 1e-6   # floats are written at 4 dp; anything above this is a real difference


def run_summarize(domain: str) -> int:
    env = dict(os.environ, LLM_FAULTS_DATASET=domain)
    print(f"  running summarize.py for {domain} ...", flush=True)
    proc = subprocess.run([sys.executable, "summarize.py"], cwd=CODE, env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ERROR: summarize.py failed for {domain}\n{proc.stderr[-2000:]}")
    return proc.returncode


def compare(domain: str) -> bool:
    got_path = ROOT / "experiment" / "analysis" / domain / "summary_table.csv"
    exp_path = ROOT / "expected" / domain / "summary_table.csv"
    got = pd.read_csv(got_path).sort_values(KEYS).reset_index(drop=True)
    exp = pd.read_csv(exp_path).sort_values(KEYS).reset_index(drop=True)

    if len(got) != len(exp):
        print(f"  FAIL {domain}: row count {len(got)} != expected {len(exp)}")
        return False
    if not (got[KEYS] == exp[KEYS]).all().all():
        print(f"  FAIL {domain}: model/style keys differ")
        return False

    bad = []
    for col in COLUMN_MAP:
        if col not in got.columns or col not in exp.columns:
            bad.append((col, "column missing"))
            continue
        diff = (got[col] - exp[col]).abs()
        off = diff > TOLERANCE
        if off.any():
            for i in got.index[off]:
                bad.append((f"{got.at[i,'model']}/{got.at[i,'style']}.{col}",
                            f"got {got.at[i,col]}, expected {exp.at[i,col]}"))

    if bad:
        print(f"  FAIL {domain}: {len(bad)} discrepancies")
        for name, detail in bad[:20]:
            print(f"      {name}: {detail}")
        if len(bad) > 20:
            print(f"      ... and {len(bad)-20} more")
        return False

    print(f"  OK   {domain}: all {len(got)} rows x {len(COLUMN_MAP)} columns match the paper")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=["GSS", "WVS"], action="append",
                    help="verify only this domain (repeatable). Default: both.")
    args = ap.parse_args()
    domains = args.domain or ["GSS", "WVS"]

    print("=" * 70)
    print("Verifying the paper's main results table against a fresh re-run")
    print("=" * 70)

    ok = True
    for d in domains:
        print(f"\n{d}:")
        if run_summarize(d) != 0:
            ok = False
            continue
        ok = compare(d) and ok

    print("\n" + "=" * 70)
    if ok:
        n = 8 * len(domains)
        print(f"VERIFIED -- {n} rows regenerated from cached model outputs and")
        print("matching the published table exactly.")
        print("=" * 70)
        return 0
    print("MISMATCH -- see the discrepancies above.")
    print("If you changed anything under experiment/, this is expected.")
    print("Otherwise please open an issue with this output attached.")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
