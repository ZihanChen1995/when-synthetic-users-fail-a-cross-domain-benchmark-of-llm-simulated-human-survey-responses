"""Decision-impact analysis (cached outputs only; no new model calls).

Question: when a product / marketing / policy team reads
*segment-level* signal off synthetic users and acts on it, how wrong is the
resulting decision relative to acting on the real human population?

We operationalise a single, common decision-support task that synthetic users
are explicitly sold for: **segment targeting / prioritisation.** For a given
attitude question and a demographic axis (e.g. GSS political views, WVS country),
a team wants to know *which segment is most extreme on the attitude* and *how
large the between-segment gap is*, so it can target the top segment, size a niche,
or claim "this attitude splits along <axis>."

For each (question, axis) we compute, identically for humans and for the model:

  share(segment)  = P(answer in the "high"/target end of the scale | segment)
      - ordinal questions: P(answer >= midpoint of the scale)
      - binary/nominal   : P(answer == the modal "high" code); we use code order

  gap    = max_segment share - min_segment share        (the between-segment spread
                                                          a targeting decision keys on)
  argmax = the segment a team would target (highest share)

Three decision-impact quantities, all from REAL cached predictions vs REAL humans:

  (1) Gap inflation:   gap_model / gap_human, and gap_model - gap_human.
       How much larger the model makes the between-segment difference look. A team
       sizing a segment gap off synthetic users over-estimates it by this factor.

  (2) Target agreement: does the model's top segment match the human top segment?
       A mis-match means the team targets the WRONG segment. We report the rate of
       mis-match across (question, axis) pairs, and the human-side cost of the
       error: how much attitude the team leaves on the table by targeting the
       model's pick instead of the true human pick
       (= share_human[human_argmax] - share_human[model_argmax]).

  (3) Spurious-split rate: pairs where humans show a negligible gap
       (<= NEG_GAP) but the model shows a large one (>= BIG_GAP). These are the
       cases where a team would "discover" a segment split that does not exist in
       real people -- the most damaging error for policy/market structure reads.

We restrict to axes with >= MIN_AXIS_LEVELS segments and cells with enough model
predictions / human n, exactly as the eta^2 analysis does, so the decision numbers
rest on the same well-powered cells. Bootstrap 95% CIs over pilot respondents on
the model-side gap. Human shares use the FULL in-scope table (large, stable n).

Run per dataset (LLM_FAULTS_DATASET). Writes analysis/<DS>/decision_impact.json.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C

MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6", "llama3.1-8b", "llama3.3-70b"]
MIN_AXIS_LEVELS = 3
MIN_CELL_MODEL = 10
MIN_CELL_HUMAN = 20
N_BOOT = 500
BOOT_SEED = C.RANDOM_SEED
NEG_GAP = 0.10     # humans: "no meaningful segment split" if max-min share <= 10 points
BIG_GAP = 0.25     # model: "large segment split" if max-min share >= 25 points


def load_runs(model_name):
    path = C.BUILD_DIR / "llm_runs" / f"{model_name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pred_code(rec):
    if rec["style"] == "A":
        return rec["pred"] if rec["valid"] else None
    if rec["valid"] and rec["dist"]:
        return int(max(rec["dist"], key=rec["dist"].get))
    return None


def _high_codes(codes):
    """The 'high/target' end of the scale: codes >= midpoint. For a 2-option
    question this is the larger code; for ordinal scales the top half. This defines
    a single scalar 'share' per segment that a targeting decision reads."""
    codes = sorted(int(c) for c in codes)
    mid = (codes[0] + codes[-1]) / 2.0
    return set(c for c in codes if c >= mid)


def _share(counts_by_code, high):
    tot = sum(counts_by_code.values())
    if tot <= 0:
        return None
    return sum(v for c, v in counts_by_code.items() if c in high) / tot


def human_shares(H, qid, axis, high):
    """{segment_level: share_high} from the full-table per-cell human distributions."""
    out = {}
    for cellkey, cell in H[qid]["subgroups"].items():
        if not cellkey.startswith(axis + "="):
            continue
        if cell["n"] < MIN_CELL_HUMAN:
            continue
        lvl = int(cellkey.split("=")[1])
        counts = {int(c): cell["dist"].get(str(c), 0.0) for c in H[qid]["codes"]}
        s = _share(counts, high)
        if s is not None:
            out[lvl] = s
    return out


def model_rows(runs, qid, axis, demo, style):
    """list of (segment_level, pred_code) for one question/style."""
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
    return rows


def model_shares(rows, high):
    by = defaultdict(lambda: defaultdict(int))
    for lvl, pc in rows:
        by[lvl][pc] += 1
    out = {}
    for lvl, counts in by.items():
        if sum(counts.values()) < MIN_CELL_MODEL:
            continue
        out[lvl] = _share(counts, high)
    return out


def _gap_and_argmax(shares, restrict=None):
    """gap = max-min share; argmax = segment with the max share. restrict = set of
    segments to consider (so model and human are compared on the SAME segments)."""
    items = [(l, s) for l, s in shares.items() if s is not None and (restrict is None or l in restrict)]
    if len(items) < MIN_AXIS_LEVELS:
        return None, None, None
    levels = [l for l, _ in items]; vals = [s for _, s in items]
    gap = max(vals) - min(vals)
    argmax = levels[int(np.argmax(vals))]
    argmin = levels[int(np.argmin(vals))]
    return gap, argmax, argmin


def analyze_model(model_name):
    runs = load_runs(model_name)
    H = json.loads((C.BUILD_DIR / "human_dist.json").read_text())
    pilot = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")
    demo = pilot.set_index(["respondent_id", "question_id"])[C.SUBGROUP_VARS].to_dict("index")

    out = {}
    for style in sorted({r["style"] for r in runs}):
        pairs = []
        for qid in C.QUESTIONS:
            codes = [int(c) for c in H[qid]["codes"]]
            high = _high_codes(codes)
            for axis in C.SUBGROUP_VARS:
                hsh = human_shares(H, qid, axis, high)
                if len(hsh) < MIN_AXIS_LEVELS:
                    continue
                rows = model_rows(runs, qid, axis, demo, style)
                msh = model_shares(rows, high)
                # compare on the shared set of segments
                shared = set(hsh) & set(msh)
                if len(shared) < MIN_AXIS_LEVELS:
                    continue
                hgap, hargmax, _ = _gap_and_argmax(hsh, restrict=shared)
                mgap, margmax, _ = _gap_and_argmax(msh, restrict=shared)
                if hgap is None or mgap is None:
                    continue
                # bootstrap model gap over respondents
                arr = np.array(rows)
                rng = np.random.default_rng(BOOT_SEED)
                bgaps = []
                for _ in range(N_BOOT):
                    samp = arr[rng.integers(0, len(arr), len(arr))]
                    bsh = model_shares([(int(l), int(p)) for l, p in samp], high)
                    g, _, _ = _gap_and_argmax(bsh, restrict=shared)
                    if g is not None:
                        bgaps.append(g)
                gap_ci = [round(float(np.percentile(bgaps, 2.5)), 3),
                          round(float(np.percentile(bgaps, 97.5)), 3)] if len(bgaps) >= 50 else [None, None]
                target_mismatch = int(margmax != hargmax)
                # human-side attitude cost of targeting the model's pick vs the true pick
                cost = round(float(hsh[hargmax] - hsh.get(margmax, hsh[hargmax])), 3)
                pairs.append({
                    "question": qid, "axis": axis,
                    "n_segments": len(shared),
                    "gap_human": round(hgap, 3), "gap_model": round(mgap, 3),
                    "gap_model_ci95": gap_ci,
                    "gap_inflation": round(mgap / hgap, 2) if hgap > 1e-6 else None,
                    "gap_excess": round(mgap - hgap, 3),
                    "human_argmax": hargmax, "model_argmax": margmax,
                    "target_mismatch": target_mismatch,
                    "targeting_cost_human_share": cost,
                    "spurious_split": int(hgap <= NEG_GAP and mgap >= BIG_GAP),
                })
        if not pairs:
            continue
        infl = [p["gap_inflation"] for p in pairs if p["gap_inflation"] is not None]
        excess = [p["gap_excess"] for p in pairs]
        mism = [p["target_mismatch"] for p in pairs]
        costs = [p["targeting_cost_human_share"] for p in pairs if p["target_mismatch"] == 1]
        spur = [p["spurious_split"] for p in pairs]
        out[style] = {
            "n_pairs": len(pairs),
            "median_gap_inflation": round(float(np.median(infl)), 2) if infl else None,
            "median_gap_excess": round(float(np.median(excess)), 3),
            "target_mismatch_rate": round(float(np.mean(mism)), 3),
            "n_target_mismatch": int(sum(mism)),
            "median_targeting_cost_when_wrong": round(float(np.median(costs)), 3) if costs else None,
            "max_targeting_cost": round(float(np.max([p["targeting_cost_human_share"] for p in pairs])), 3),
            "spurious_split_rate": round(float(np.mean(spur)), 3),
            "n_spurious_split": int(sum(spur)),
            "by_pair": pairs,
        }
    return out


def main():
    result = {"dataset": C.DATASET, "params": {
        "NEG_GAP": NEG_GAP, "BIG_GAP": BIG_GAP, "MIN_CELL_MODEL": MIN_CELL_MODEL,
        "MIN_CELL_HUMAN": MIN_CELL_HUMAN, "n_boot": N_BOOT, "ci": "95%"}, "models": {}}
    print(f"\n########## DECISION IMPACT: {C.DATASET} ##########")
    for model in MODELS:
        try:
            load_runs(model)
        except FileNotFoundError:
            continue
        r = analyze_model(model)
        result["models"][model] = r
        for style, s in r.items():
            print(f"\n--- {model} style {style} (n_pairs={s['n_pairs']}) ---")
            print(f"  median gap inflation (model/human)   : {s['median_gap_inflation']}x")
            print(f"  median gap excess (model-human)      : {s['median_gap_excess']:+.3f}")
            print(f"  wrong-target rate                    : {s['target_mismatch_rate']:.0%} ({s['n_target_mismatch']}/{s['n_pairs']})")
            print(f"  median cost when wrong (human share) : {s['median_targeting_cost_when_wrong']}")
            print(f"  spurious-split rate                  : {s['spurious_split_rate']:.0%} ({s['n_spurious_split']}/{s['n_pairs']})")

    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / "decision_impact.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {C.ANALYSIS_DIR / 'decision_impact.json'}")


if __name__ == "__main__":
    main()
