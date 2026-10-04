"""Appendix robustness checks (cached outputs only; no new model calls).

  (1) Multiple comparisons for the stereotyping tests. The stereotyping index is
      tested over many (question, axis) pairs; a per-pair 90% CI does not control
      the family-wise / false-discovery error. We recompute, for each pair, a
      one-sided bootstrap p-value for H0: index <= 0 (no over-determination), then
      apply Benjamini-Hochberg FDR at q = 0.05 across all pairs for a model/style,
      and report how many pairs remain significant after correction. If most
      survive, "significant for the large majority of pairs" holds under FDR.

  (2) Respondent-clustered bootstrap on the RQ1 margin. Some respondents answer
      several questions, so rows are not independent. We recompute the 95% CI on
      (model_acc - baseline_acc) with a CLUSTER bootstrap: resample respondents
      (not rows) with replacement, take all their rows. If the clustered CI still
      excludes zero, the individual-level deficit is not an artifact of
      within-respondent correlation.

Run per dataset (LLM_FAULTS_DATASET). Writes analysis/<DS>/appendix_robustness.json.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C

MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6", "llama3.1-8b", "llama3.3-70b"]
MIN_CELL_MODEL = 10
MIN_AXIS_LEVELS = 3
N_BOOT = 500
BOOT_SEED = C.RANDOM_SEED
FDR_Q = 0.05


def load_runs(model_name):
    path = C.BUILD_DIR / "llm_runs" / f"{model_name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pred_code(rec):
    if rec["style"] == "A":
        return rec["pred"] if rec["valid"] else None
    if rec["valid"] and rec["dist"]:
        return int(max(rec["dist"], key=rec["dist"].get))
    return None


def _dist_moments(p, codes):
    p = np.asarray(p, float); s = p.sum()
    if s <= 0:
        return None, None
    p = p / s; cc = np.array(codes, float)
    m = float((p * cc).sum())
    return m, float((p * (cc - m) ** 2).sum())


def _eta2_cells(cells, codes):
    means, withins, weights = [], [], []
    for w, counts in cells:
        p = np.array([counts.get(c, 0.0) for c in codes], float)
        m, v = _dist_moments(p, codes)
        if m is None:
            continue
        means.append(m); withins.append(v); weights.append(w)
    if len(means) < MIN_AXIS_LEVELS:
        return None
    w = np.array(weights, float); w = w / w.sum()
    means = np.array(means); withins = np.array(withins)
    grand = float((w * means).sum())
    between = float((w * (means - grand) ** 2).sum())
    within = float((w * withins).sum())
    tot = between + within
    return float(between / tot) if tot > 1e-4 else None


# ----------------------------------------------------- (1) FDR on stereotyping
def bh_fdr(pvals, q):
    """Benjamini-Hochberg: return boolean mask of rejections at level q."""
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(n, bool)
    kmax = np.max(np.where(passed)[0])
    cut = p[order][kmax]
    return p <= cut


def stereo_fdr(model_name):
    runs = load_runs(model_name)
    H = json.loads((C.BUILD_DIR / "human_dist.json").read_text())
    pilot = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")
    demo = pilot.set_index(["respondent_id", "question_id"])[C.SUBGROUP_VARS].to_dict("index")

    out = {}
    for style in sorted({r["style"] for r in runs}):
        pair_ids, pvals, indices = [], [], []
        for qid in C.QUESTIONS:
            if not C.QUESTIONS[qid].get("ordinal", False):
                continue
            codes = [int(c) for c in H[qid]["codes"]]
            for axis in C.SUBGROUP_VARS:
                # human eta^2 from full-table cells
                hcells = []
                for cellkey, cell in H[qid]["subgroups"].items():
                    if cellkey.startswith(axis + "="):
                        counts = {int(c): cell["dist"].get(str(c), 0.0) for c in codes}
                        hcells.append((cell["n"], counts))
                eta_h = _eta2_cells(hcells, codes)
                if eta_h is None:
                    continue
                rows = []
                for r in runs:
                    if r["style"] != style or r["question_id"] != qid:
                        continue
                    pc = _pred_code(r)
                    if pc is None:
                        continue
                    d = demo.get((r["respondent_id"], qid))
                    if d is None:
                        continue
                    rows.append((int(d[axis]), pc))
                if len(rows) < 30:
                    continue
                arr = np.array(rows)

                def eta_of(sample):
                    by = defaultdict(lambda: defaultdict(int))
                    for l, pc in sample:
                        by[int(l)][int(pc)] += 1
                    cells = [(sum(c.values()), c) for c in by.values() if sum(c.values()) >= MIN_CELL_MODEL]
                    return _eta2_cells(cells, codes)

                eta_m = eta_of(arr)
                if eta_m is None:
                    continue
                rng = np.random.default_rng(BOOT_SEED)
                boot = []
                for _ in range(N_BOOT):
                    s = arr[rng.integers(0, len(arr), len(arr))]
                    e = eta_of(s)
                    if e is not None:
                        boot.append(e - eta_h)
                if len(boot) < 50:
                    continue
                boot = np.array(boot)
                # one-sided p for H0: index <= 0  ->  fraction of boot draws <= 0
                p = float((boot <= 0).mean())
                pair_ids.append(f"{qid}:{axis}")
                pvals.append(p)
                indices.append(eta_m - eta_h)
        if not pvals:
            continue
        rej = bh_fdr(pvals, FDR_Q)
        # count only positive-index rejections (over-determination direction)
        pos_sig = sum(1 for i, r in enumerate(rej) if r and indices[i] > 0)
        out[style] = {
            "n_pairs": len(pvals),
            "n_sig_uncorrected_p05": int(sum(1 for i, p in enumerate(pvals) if p < 0.05 and indices[i] > 0)),
            "n_sig_bh_fdr_q05": int(pos_sig),
            "median_index": round(float(np.median(indices)), 3),
        }
    return out


# ----------------------------------------------------- (2) clustered bootstrap on Delta_base
def clustered_delta_base(model_name):
    runs = load_runs(model_name)
    bp = pd.read_parquet(C.BUILD_DIR / "baseline_preds.parquet").set_index(
        ["respondent_id", "question_id"])
    out = {}
    for style in sorted({r["style"] for r in runs}):
        by_resp = defaultdict(list)  # respondent -> list of (m_correct, b_correct)
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
            by_resp[r["respondent_id"]].append(
                (int(pc == r["human_answer"]), int(base == r["human_answer"])))
        if not by_resp:
            continue
        resp_ids = list(by_resp.keys())
        # point estimate over all rows
        all_rows = np.array([x for v in by_resp.values() for x in v])
        point = float((all_rows[:, 0] - all_rows[:, 1]).mean())
        # cluster bootstrap: resample respondents
        rng = np.random.default_rng(BOOT_SEED)
        boot = []
        for _ in range(N_BOOT):
            pick = [resp_ids[i] for i in rng.integers(0, len(resp_ids), len(resp_ids))]
            rows = np.array([x for rid in pick for x in by_resp[rid]])
            boot.append(float((rows[:, 0] - rows[:, 1]).mean()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[style] = {
            "n_respondents": len(resp_ids),
            "n_rows": len(all_rows),
            "delta_base": round(point, 4),
            "clustered_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(hi < 0 or lo > 0),
        }
    return out


def main():
    result = {"dataset": C.DATASET, "fdr_q": FDR_Q, "stereo_fdr": {}, "clustered_delta_base": {}}
    print(f"\n########## APPENDIX ROBUSTNESS: {C.DATASET} ##########")
    for model in MODELS:
        try:
            load_runs(model)
        except FileNotFoundError:
            continue
        sf = stereo_fdr(model)
        cb = clustered_delta_base(model)
        result["stereo_fdr"][model] = sf
        result["clustered_delta_base"][model] = cb
        print(f"\n--- {model} ---")
        for s, v in sf.items():
            print(f"  [FDR stereo] style {s}: {v['n_sig_bh_fdr_q05']}/{v['n_pairs']} sig after BH-FDR q=.05 "
                  f"(uncorrected {v['n_sig_uncorrected_p05']}); median index {v['median_index']:+.3f}")
        for s, v in cb.items():
            print(f"  [clustered Delta_base] style {s}: {v['delta_base']:+.4f} "
                  f"CI95 {v['clustered_ci95']} excl0={v['excludes_zero']} "
                  f"(n_resp={v['n_respondents']}, n_rows={v['n_rows']})")

    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / "appendix_robustness.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {C.ANALYSIS_DIR / 'appendix_robustness.json'}")


if __name__ == "__main__":
    main()
