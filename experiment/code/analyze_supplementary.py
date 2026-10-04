"""Supplementary analyses (cached outputs only).

  (1) Coding-invariant stereotyping. The eta^2 stereotyping index assumes a
      numeric/ordinal coding, so we add a purely nominal association measure that
      needs NO numeric codes: Cramer's V on the (group x answer) contingency
      table, computed identically for humans and for the model. The
      coding-invariant stereotyping index is  Delta V = V_model - V_human. We run
      it on ALL questions (nominal + ordinal), so it also covers the nominal GSS
      items the eta^2 index skips. Bootstrap 95% CI over respondents on V_model.

  (2) Paired-bootstrap CI on the individual-accuracy margin over the baseline.
      We resample pilot rows (paired: model and baseline scored on the SAME rows)
      500 times and report the 95% CI of (model_acc - baseline_acc).

  (3) Population-reweighted aggregate JS. Because the pilot samples by cell, the
      raw pooled distribution reflects the sampling design, not the population.
      We recompute the model's aggregate answer distribution weighting each cell
      back to its human population share, and report JS vs the true (full-table)
      human distribution. If population-reweighted JS tracks the raw JS, the
      aggregate-fidelity claim is not an artifact of the cell sampling.

Run per dataset (LLM_FAULTS_DATASET). Writes analysis/<DS>/supplementary_metrics.json.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C

MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6", "llama3.1-8b", "llama3.3-70b"]
N_BOOT = 500
BOOT_SEED = C.RANDOM_SEED
MIN_CELL_MODEL = 10


def load_runs(model_name):
    path = C.BUILD_DIR / "llm_runs" / f"{model_name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pred_code(rec):
    if rec["style"] == "A":
        return rec["pred"] if rec["valid"] else None
    if rec["valid"] and rec["dist"]:
        return int(max(rec["dist"], key=rec["dist"].get))
    return None


# --------------------------------------------------- (1) Cramer's V (coding-invariant)
def cramers_v_from_table(table):
    """table: 2D np array of counts (rows=groups, cols=answers). Bias-corrected
    Cramer's V (Bergsma 2013). Returns None if degenerate."""
    table = np.asarray(table, float)
    n = table.sum()
    if n <= 0:
        return None
    row = table.sum(1, keepdims=True); col = table.sum(0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0))
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(phi2corr / denom))


def _human_table(H, qid, var, codes):
    """Human counts table (groups x answers) from full-table per-cell dists * n."""
    rows = []
    for cellkey, cell in H[qid]["subgroups"].items():
        if not cellkey.startswith(var + "="):
            continue
        n = cell["n"]
        rows.append([cell["dist"].get(str(c), 0.0) * n for c in codes])
    return np.array(rows) if len(rows) >= 3 else None


def coding_invariant(model_name):
    runs = load_runs(model_name)
    H = json.loads((C.BUILD_DIR / "human_dist.json").read_text())
    pilot = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")
    demo = pilot.set_index(["respondent_id", "question_id"])[C.SUBGROUP_VARS].to_dict("index")

    out = {}
    for style in sorted({r["style"] for r in runs}):
        srecs = [r for r in runs if r["style"] == style]
        pairs, sig = [], 0
        details = {}
        for qid in C.QUESTIONS:
            codes = [int(c) for c in H[qid]["codes"]]
            for var in C.SUBGROUP_VARS:
                htab = _human_table(H, qid, var, codes)
                if htab is None:
                    continue
                v_h = cramers_v_from_table(htab)
                if v_h is None:
                    continue
                # model rows: (group level, predicted code) for this question
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
                if len(rows) < 30:
                    continue
                levels = sorted({l for l, _ in rows})
                cidx = {c: i for i, c in enumerate(codes)}
                lidx = {l: i for i, l in enumerate(levels)}
                if len(levels) < 3:
                    continue
                tab = np.zeros((len(levels), len(codes)))
                for l, pc in rows:
                    if pc in cidx:
                        tab[lidx[l], cidx[pc]] += 1
                # drop cells (rows) with too few model preds
                keep = tab.sum(1) >= MIN_CELL_MODEL
                if keep.sum() < 3:
                    continue
                tab = tab[keep]
                v_m = cramers_v_from_table(tab)
                if v_m is None:
                    continue
                # bootstrap V_model over respondents
                arr = np.array(rows)
                rng = np.random.default_rng(BOOT_SEED)
                boot = []
                for _ in range(N_BOOT):
                    samp = arr[rng.integers(0, len(arr), len(arr))]
                    t = np.zeros((len(levels), len(codes)))
                    for l, pc in samp:
                        if int(pc) in cidx and int(l) in lidx:
                            t[lidx[int(l)], cidx[int(pc)]] += 1
                    t = t[t.sum(1) >= MIN_CELL_MODEL]
                    if t.shape[0] >= 3:
                        vv = cramers_v_from_table(t)
                        if vv is not None:
                            boot.append(vv)
                if len(boot) >= 50:
                    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
                    idx_lo, idx_hi = lo - v_h, hi - v_h
                    if idx_lo > 0 or idx_hi < 0:
                        sig += 1
                else:
                    idx_lo = idx_hi = None
                idx = v_m - v_h
                pairs.append(idx)
                details[f"{qid}:{var}"] = {
                    "V_human": round(v_h, 3), "V_model": round(v_m, 3),
                    "deltaV": round(idx, 3),
                    "deltaV_ci95": [round(idx_lo, 3), round(idx_hi, 3)] if idx_lo is not None else [None, None],
                }
        out[style] = {
            "median_deltaV": round(float(np.median(pairs)), 3) if pairs else None,
            "n_pairs": len(pairs),
            "n_pairs_positive": sum(1 for x in pairs if x > 0),
            "n_pairs_ci_excludes_0": sig,
            "by_pair": details,
        }
    return out


# --------------------------------------------------- (2) paired-bootstrap CI on Delta_base
def delta_base_ci(model_name):
    runs = load_runs(model_name)
    bp = pd.read_parquet(C.BUILD_DIR / "baseline_preds.parquet").set_index(
        ["respondent_id", "question_id"])
    out = {}
    for style in sorted({r["style"] for r in runs}):
        rows = []  # (model_correct, base_correct)
        for r in runs:
            if r["style"] != style:
                continue
            pc = _pred_code(r)
            if pc is None:
                continue
            try:
                base = int(bp.loc[(r["respondent_id"], r["question_id"]), "baseline_pred"])
            except KeyError:
                continue
            rows.append((int(pc == r["human_answer"]), int(base == r["human_answer"])))
        if not rows:
            continue
        arr = np.array(rows)
        diff = arr[:, 0] - arr[:, 1]
        point = float(diff.mean())
        rng = np.random.default_rng(BOOT_SEED)
        boot = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(N_BOOT)]
        out[style] = {
            "n": len(rows),
            "delta_base": round(point, 4),
            "delta_base_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                 round(float(np.percentile(boot, 97.5)), 4)],
        }
    return out


# --------------------------------------------------- (3) population-reweighted JS
def _js(p, q):
    from scipy.spatial.distance import jensenshannon
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() else p; q = q / q.sum() if q.sum() else q
    return float(jensenshannon(p, q, base=2) ** 2)


def reweighted_js(model_name):
    """Reweight each stratum's model distribution by its human population share
    (using the primary subgroup axis), then JS vs the full-table human dist."""
    runs = load_runs(model_name)
    H = json.loads((C.BUILD_DIR / "human_dist.json").read_text())
    pilot = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")
    demo = pilot.set_index(["respondent_id", "question_id"]).to_dict("index")
    axis = C.SUBGROUP_VARS[0]  # primary subgroup axis (GSS: sex, WVS: country)
    out = {}
    for style in sorted({r["style"] for r in runs}):
        raw_js, rew_js = [], []
        for qid in C.QUESTIONS:
            codes = [int(c) for c in H[qid]["codes"]]
            hdist = np.array([H[qid]["overall"].get(str(c), 0.0) for c in codes])
            # gather model preds/dists per axis level
            by_level_counts = defaultdict(lambda: np.zeros(len(codes)))
            raw_counts = np.zeros(len(codes))
            cidx = {c: i for i, c in enumerate(codes)}
            for r in runs:
                if r["style"] != style or r["question_id"] != qid or not r["valid"]:
                    continue
                d = demo.get((r["respondent_id"], qid))
                if d is None:
                    continue
                lvl = int(d[axis])
                if r["style"] == "A":
                    if r["pred"] in cidx:
                        by_level_counts[lvl][cidx[r["pred"]]] += 1
                        raw_counts[cidx[r["pred"]]] += 1
                else:
                    if r["dist"]:
                        for k, v in r["dist"].items():
                            if int(k) in cidx:
                                by_level_counts[lvl][cidx[int(k)]] += v
                                raw_counts[cidx[int(k)]] += v
            if raw_counts.sum() <= 0:
                continue
            # population weights from human full-table cell n
            hw = {}
            for cellkey, cell in H[qid]["subgroups"].items():
                if cellkey.startswith(axis + "="):
                    hw[int(cellkey.split("=")[1])] = cell["n"]
            rew = np.zeros(len(codes)); wsum = 0.0
            for lvl, cnt in by_level_counts.items():
                if cnt.sum() <= 0 or lvl not in hw:
                    continue
                w = hw[lvl]
                rew += w * (cnt / cnt.sum()); wsum += w
            raw_js.append(_js(raw_counts, hdist))
            if wsum > 0:
                rew_js.append(_js(rew, hdist))
        out[style] = {
            "raw_js": round(float(np.mean(raw_js)), 4) if raw_js else None,
            "pop_reweighted_js": round(float(np.mean(rew_js)), 4) if rew_js else None,
        }
    return out


def main():
    result = {"dataset": C.DATASET, "coding_invariant": {}, "delta_base_ci": {}, "reweighted_js": {}}
    print(f"\n########## {C.DATASET} ##########")
    for model in MODELS:
        try:
            load_runs(model)
        except FileNotFoundError:
            continue
        ci = coding_invariant(model)
        db = delta_base_ci(model)
        rw = reweighted_js(model)
        result["coding_invariant"][model] = ci
        result["delta_base_ci"][model] = db
        result["reweighted_js"][model] = rw
        print(f"\n--- {model} ---")
        for s in ci:
            c = ci[s]
            print(f"  [coding-invariant Cramer's V] style {s}: median deltaV={c['median_deltaV']} "
                  f"({c['n_pairs_positive']}/{c['n_pairs']} positive, {c['n_pairs_ci_excludes_0']} CI-excl-0)")
        for s in db:
            print(f"  [Delta_base CI] style {s}: {db[s]['delta_base']} "
                  f"CI95 {db[s]['delta_base_ci95']} (n={db[s]['n']})")
        for s in rw:
            print(f"  [JS reweight] style {s}: raw {rw[s]['raw_js']} -> pop-reweighted {rw[s]['pop_reweighted_js']}")

    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / "supplementary_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {C.ANALYSIS_DIR / 'supplementary_metrics.json'}")


if __name__ == "__main__":
    main()
