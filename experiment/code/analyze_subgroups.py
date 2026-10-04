"""Subgroup structure analysis — the headline metric.

We ask, per attitude question and per demographic axis: *how deterministic is the
demographic of the answer* — for real humans vs. for the model. The standard,
bounded effect-size for "fraction of answer variance explained by a categorical
factor" is eta-squared:

    eta2 = between_group_variance / (between_group_variance + within_group_variance)   in [0, 1]

Computed identically for three sources, so they are directly comparable:
  - human   : per-cell answer distribution from the FULL in-scope table (stable, large n)
  - model_A : per-cell empirical distribution of style-A argmax predictions
  - model_C : per-cell mean of style-C predicted distributions

Headline quantity:
    stereotyping_index = eta2_model - eta2_human
      > 0  model makes demographics MORE deterministic than reality (stereotyping /
           caricature — amplified between-group gaps and/or compressed within-group spread)
      < 0  model makes demographics LESS deterministic (flattening)

Properties:
  - bounded in [-1, 1]; no near-zero-denominator explosions
  - accounts for within-group spread, not just group means
  - identical per-cell-distribution construction removes the style-A argmax bias

We bootstrap over pilot respondents to put a 95% CI on eta2_model (hence on the
index), so we can say whether the effect is real at pilot scale. Ordinal questions
only (a mean answer must be meaningful).

Outputs analysis/<DS>/flattening_<model>.json and prints a summary.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C

MIN_CELL_MODEL = 10      # min model predictions in a cell to include it
MIN_AXIS_LEVELS = 3      # need >=3 subgroup levels to talk about "variance across groups"
HUMAN_TOTAL_VAR_EPS = 1e-4   # skip axes where humans show ~no variance at all (metric meaningless)
N_BOOT = 500
BOOT_SEED = C.RANDOM_SEED


def load_runs(model_name):
    path = C.BUILD_DIR / "llm_runs" / f"{model_name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pred_code(rec):
    if rec["style"] == "A":
        return rec["pred"] if rec["valid"] else None
    if rec["valid"] and rec["dist"]:
        return int(max(rec["dist"], key=rec["dist"].get))
    return None


def _dist_moments(dist_over_codes, codes):
    """(mean, within-variance) of an ordinal categorical distribution given as {code: prob}."""
    p = np.array([dist_over_codes.get(c, 0.0) for c in codes], float)
    s = p.sum()
    if s <= 0:
        return None, None
    p = p / s
    cc = np.array(codes, float)
    mean = float((p * cc).sum())
    var = float((p * (cc - mean) ** 2).sum())
    return mean, var


def eta2_from_cells(cells, codes):
    """cells: list of (weight, {code: prob}). Returns eta2 = between/(between+within)."""
    means, withins, weights = [], [], []
    for w, dist in cells:
        m, v = _dist_moments(dist, codes)
        if m is None:
            continue
        means.append(m); withins.append(v); weights.append(w)
    if len(means) < MIN_AXIS_LEVELS:
        return None, None, None
    w = np.array(weights, float); w = w / w.sum()
    means = np.array(means); withins = np.array(withins)
    grand = float((w * means).sum())
    between = float((w * (means - grand) ** 2).sum())
    within = float((w * withins).sum())
    total = between + within
    if total <= HUMAN_TOTAL_VAR_EPS:
        return None, between, within
    return between / total, between, within


def _human_cells(H, qid, var, codes):
    cells = []
    for cellkey, cell in H[qid]["subgroups"].items():
        if not cellkey.startswith(var + "="):
            continue
        dist = {int(c): cell["dist"][str(c)] for c in codes}
        cells.append((cell["n"], dist))
    return cells


def _model_cells(preds_by_level, codes):
    """preds_by_level: {level: [codes...]} -> list of (weight=n, empirical dist)."""
    cells = []
    for level, preds in preds_by_level.items():
        if len(preds) < MIN_CELL_MODEL:
            continue
        counts = defaultdict(float)
        for c in preds:
            counts[c] += 1
        n = len(preds)
        dist = {c: counts.get(c, 0.0) / n for c in codes}
        cells.append((n, dist))
    return cells


def analyze_style(srecs, H, demo, style):
    per_q = {}
    for qid, qmeta in C.QUESTIONS.items():
        if not qmeta["ordinal"]:
            continue
        codes = [int(c) for c in H[qid]["codes"]]
        axes = {}
        for var in C.SUBGROUP_VARS:
            eta_h, betw_h, with_h = eta2_from_cells(_human_cells(H, qid, var, codes), codes)
            if eta_h is None:
                continue

            # gather model predictions for this (question) keyed by subgroup level,
            # keeping respondent ids so we can bootstrap
            rows = []
            for r in srecs:
                if r["question_id"] != qid:
                    continue
                pc = _pred_code(r)
                if pc is None:
                    continue
                d = demo.get((r["respondent_id"], qid))
                if d is None:
                    continue
                rows.append((int(d[var]), pc))
            if not rows:
                continue

            by_level = defaultdict(list)
            for lvl, pc in rows:
                by_level[lvl].append(pc)
            eta_m, betw_m, with_m = eta2_from_cells(_model_cells(by_level, codes), codes)
            if eta_m is None:
                continue

            # bootstrap eta2_model over respondents
            rng = np.random.default_rng(BOOT_SEED)
            arr = np.array(rows)
            boot = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(arr), len(arr))
                samp = arr[idx]
                bl = defaultdict(list)
                for lvl, pc in samp:
                    bl[int(lvl)].append(int(pc))
                e, _, _ = eta2_from_cells(_model_cells(bl, codes), codes)
                if e is not None:
                    boot.append(e)
            if len(boot) < 50:
                ci = [None, None]
            else:
                ci = [round(float(np.percentile(boot, 2.5)), 3),
                      round(float(np.percentile(boot, 97.5)), 3)]

            index = eta_m - eta_h
            axes[var] = {
                "n_model_preds": len(rows),
                "eta2_human": round(eta_h, 3),
                "eta2_model": round(eta_m, 3),
                "stereotyping_index": round(index, 3),
                "eta2_model_ci95": ci,
                # index CI: shift the model-eta2 CI by the (fixed, large-n) human eta2
                "index_ci95": [round(ci[0] - eta_h, 3), round(ci[1] - eta_h, 3)]
                               if ci[0] is not None else [None, None],
            }
        if axes:
            per_q[qid] = axes
    return per_q


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4.5"
    runs = load_runs(model_name)
    H = json.loads((C.BUILD_DIR / "human_dist.json").read_text())
    pilot = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")
    demo = pilot.set_index(["respondent_id", "question_id"])[C.SUBGROUP_VARS].to_dict("index")

    result = {}
    for style in sorted({r["style"] for r in runs}):
        srecs = [r for r in runs if r["style"] == style]
        per_q = analyze_style(srecs, H, demo, style)
        idxs = [a["stereotyping_index"] for q in per_q.values() for a in q.values()]
        # count axes whose index CI excludes 0 (a "real" effect at pilot scale)
        sig = [a for q in per_q.values() for a in q.values()
               if a["index_ci95"][0] is not None and
               (a["index_ci95"][0] > 0 or a["index_ci95"][1] < 0)]
        result[style] = {
            "median_stereotyping_index": round(float(np.median(idxs)), 3) if idxs else None,
            "n_question_axis_pairs": len(idxs),
            "n_pairs_ci_excludes_0": len(sig),
            "n_pairs_stereotyping": sum(1 for x in idxs if x > 0),
            "n_pairs_flattening": sum(1 for x in idxs if x < 0),
            "by_question": per_q,
        }

    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / f"flattening_{model_name}.json").write_text(json.dumps(result, indent=2))

    print(f"\n===== subgroup determinism: {model_name} =====")
    print("stereotyping_index = eta2_model - eta2_human   (>0 model over-associates demo->answer)\n")
    for style, r in result.items():
        print(f"--- style {style}: median index = {r['median_stereotyping_index']}  "
              f"({r['n_pairs_stereotyping']} stereotyping / {r['n_pairs_flattening']} flattening "
              f"of {r['n_question_axis_pairs']} pairs; {r['n_pairs_ci_excludes_0']} with CI excluding 0) ---")
        for qid, axes in r["by_question"].items():
            for var, a in axes.items():
                star = "*" if (a["index_ci95"][0] is not None and
                               (a["index_ci95"][0] > 0 or a["index_ci95"][1] < 0)) else " "
                print(f"  {star} {qid:9s} x {var:9s}  "
                      f"eta2_h={a['eta2_human']:.3f}  eta2_m={a['eta2_model']:.3f}  "
                      f"index={a['stereotyping_index']:+.3f}  CI95={a['index_ci95']}")
        print()


if __name__ == "__main__":
    main()
