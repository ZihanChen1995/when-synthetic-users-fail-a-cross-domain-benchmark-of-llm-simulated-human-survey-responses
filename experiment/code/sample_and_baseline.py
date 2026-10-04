"""Stratified pilot sampler + naive demographic baseline.

Two distinct populations, deliberately kept separate:

  1. FULL in-scope table (all ~86k examples) -> used to estimate
     (a) human ground-truth answer distributions per question and per subgroup cell,
     (b) the naive demographic baseline (conditional distribution predictor).

  2. PILOT sample (small, stratified) -> the rows we actually send to the LLM.
     Every LLM number is scored against the human ground truth and compared to the
     baseline evaluated on the SAME pilot rows.

Held-out baseline: the baseline is FIT on a held-out 50% of the full table and
EVALUATED on the pilot rows (which are drawn from the other 50%), so it is never
fit and scored on the same respondents. Back-off: full cell -> coarser cell ->
question marginal, so it always yields a prediction.

Outputs (datasets/<DS>/build/):
  splits.parquet          respondent_id -> fold ('fit' | 'eval')
  pilot_sample.parquet    the rows to send to the LLM (subset of 'eval' fold)
  human_dist.json         per-question and per-cell human answer distributions (full table)
  baseline_preds.parquet  baseline prediction (argmax) + full distribution per pilot row
"""
import json
import hashlib
import numpy as np
import pandas as pd

import config as C

# per-question pilot budget (rows sent to the LLM). Sums to ~1000.
PILOT_PER_QUESTION = 100


def _fold_of(respondent_id: str) -> str:
    """Deterministic 50/50 split by hashing the respondent id (stable, no RNG state)."""
    h = int(hashlib.md5(respondent_id.encode()).hexdigest(), 16)
    return "fit" if (h % 2 == 0) else "eval"


def _dist(counts: dict, codes) -> dict:
    total = sum(counts.get(c, 0) for c in codes)
    if total == 0:
        return {str(c): 0.0 for c in codes}
    return {str(c): counts.get(c, 0) / total for c in codes}


# dataset-specific baseline cell hierarchy (finest -> coarse) and pilot stratum axis.
# baseline = demographic-conditional distribution predictor; the "full" cell is the
# finest demographic conditioning we trust, backing off to "coarse" then marginal.
if C.DATASET == "GSS":
    LONG_FILE = "gss_pilot_long.parquet"
    FULL_CELL = ["degree", "race", "sex", "polviews"]
    COARSE_CELL = ["degree", "race"]
    STRATUM = ["degree", "race"]
elif C.DATASET == "WVS":
    LONG_FILE = "wvs_pilot_long.parquet"
    # country is the dominant driver of values -> condition on it first
    FULL_CELL = ["country", "education", "agecat"]
    COARSE_CELL = ["country"]
    STRATUM = ["country"]
else:
    raise ValueError(C.DATASET)


def _cell_key(row, level):
    if level == "full":
        return tuple(row[c] for c in FULL_CELL)
    if level == "coarse":
        return tuple(row[c] for c in COARSE_CELL)
    return None


def main():
    long = pd.read_parquet(C.BUILD_DIR / LONG_FILE)
    long["fold"] = long["respondent_id"].map(_fold_of)

    # ---- human ground-truth distributions (from FULL table) ----
    human = {}
    for qid, g in long.groupby("question_id"):
        codes = list(C.ANSWER_CODES[qid].keys())
        overall = _dist(g["answer_code"].value_counts().to_dict(), codes)
        cells = {}
        for var in C.SUBGROUP_VARS:
            for level, gl in g.groupby(var):
                if len(gl) < C.MIN_CELL_HUMANS:
                    continue
                cells[f"{var}={int(level)}"] = {
                    "n": int(len(gl)),
                    "dist": _dist(gl["answer_code"].value_counts().to_dict(), codes),
                }
        human[qid] = {"codes": codes, "overall": overall,
                      "n": int(len(g)), "subgroups": cells}
    (C.BUILD_DIR / "human_dist.json").write_text(json.dumps(human, indent=2))

    # ---- naive demographic baseline: fit conditional distributions on the 'fit' fold ----
    fit = long[long.fold == "fit"]
    cell_key = _cell_key

    baseline_tables = {}
    for qid, g in fit.groupby("question_id"):
        codes = list(C.ANSWER_CODES[qid].keys())
        tbl = {"full": {}, "coarse": {}, "marginal": _dist(g["answer_code"].value_counts().to_dict(), codes)}
        for lvl in ("full", "coarse"):
            keys = g.apply(lambda r: cell_key(r, lvl), axis=1)
            for k, idx in g.groupby(keys).groups.items():
                sub = g.loc[idx]
                if len(sub) >= 20:   # need a few humans to trust a conditional cell
                    tbl[lvl][str(k)] = _dist(sub["answer_code"].value_counts().to_dict(), codes)
        baseline_tables[qid] = tbl

    # ---- pilot sample: stratified over subgroup cells, drawn from 'eval' fold ----
    rng = np.random.default_rng(C.RANDOM_SEED)
    pilot_parts = []
    ev = long[long.fold == "eval"]
    for qid, g in ev.groupby("question_id"):
        # stratify by the dataset's primary subgroup axis so cells are well covered
        g = g.copy()
        g["_stratum"] = list(zip(*[g[c] for c in STRATUM])) if len(STRATUM) > 1 else list(g[STRATUM[0]])
        take = min(PILOT_PER_QUESTION, len(g))
        # proportional allocation across strata, at least 1 where possible
        frac = take / len(g)
        picks = []
        for _, gs in g.groupby("_stratum"):
            k = max(1, int(round(len(gs) * frac)))
            k = min(k, len(gs))
            picks.append(gs.sample(n=k, random_state=int(rng.integers(1e9))))
        samp = pd.concat(picks)
        if len(samp) > take:
            samp = samp.sample(n=take, random_state=C.RANDOM_SEED)
        pilot_parts.append(samp.drop(columns="_stratum"))
    pilot = pd.concat(pilot_parts).reset_index(drop=True)
    pilot.to_parquet(C.BUILD_DIR / "pilot_sample.parquet", index=False)

    # ---- baseline predictions on the pilot rows (fit on 'fit' fold, applied to 'eval' rows) ----
    def predict(row):
        tbl = baseline_tables[row["question_id"]]
        for lvl in ("full", "coarse"):
            k = str(cell_key(row, lvl))
            if k in tbl[lvl]:
                return tbl[lvl][k], lvl
        return tbl["marginal"], "marginal"

    bpreds = []
    for _, r in pilot.iterrows():
        dist, lvl = predict(r)
        argmax = max(dist, key=dist.get)
        bpreds.append({"respondent_id": r["respondent_id"], "question_id": r["question_id"],
                       "baseline_backoff": lvl, "baseline_pred": int(argmax),
                       "baseline_dist": json.dumps(dist)})
    pd.DataFrame(bpreds).to_parquet(C.BUILD_DIR / "baseline_preds.parquet", index=False)

    long[["respondent_id", "fold"]].drop_duplicates().to_parquet(C.BUILD_DIR / "splits.parquet", index=False)

    # ---- report ----
    print(f"pilot rows total : {len(pilot)}")
    print(pilot.groupby("question_id").size().to_string())
    bp = pd.DataFrame(bpreds)
    print("\nbaseline back-off usage:")
    print(bp["baseline_backoff"].value_counts().to_string())
    # baseline individual accuracy on pilot (sanity)
    merged = pilot.merge(bp, on=["respondent_id", "question_id"])
    acc = (merged["answer_code"] == merged["baseline_pred"]).mean()
    print(f"\nnaive baseline individual accuracy on pilot: {acc:.3f}")


if __name__ == "__main__":
    main()
