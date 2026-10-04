"""Decision-impact figure: model vs. human between-segment gap per (question, axis).

For the frontier model (Sonnet 4.6, single-answer prompt), scatter the human
between-segment gap (x) against the model's gap (y) for every (question, axis)
pair, with the model gap's 95% CI as a vertical bar. Points on the 45-degree line
would mean the model reads segment structure faithfully; points far above it are
gaps the model manufactures. The shaded region (human gap <= 0.10, model gap >=
0.25) is the "spurious split" zone -- segment differences a decision team would
act on that do not exist in real people.

Reads analysis/<DS>/decision_impact.json. Writes analysis/<DS>/fig_decision_gap.png.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 11,
})

import config as C

FRONTIER = "claude-sonnet-4.6"
NEG_GAP, BIG_GAP = 0.10, 0.25


def main():
    d = json.loads((C.ANALYSIS_DIR / "decision_impact.json").read_text())
    pairs = d["models"][FRONTIER]["A"]["by_pair"]
    hg = np.array([p["gap_human"] for p in pairs])
    mg = np.array([p["gap_model"] for p in pairs])
    ci = np.array([p["gap_model_ci95"] if p["gap_model_ci95"][0] is not None
                   else [p["gap_model"], p["gap_model"]] for p in pairs], float)
    spur = np.array([bool(p["spurious_split"]) for p in pairs])

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    # spurious-split zone: human gap <= NEG_GAP but model gap >= BIG_GAP
    ax.add_patch(plt.Rectangle((0, BIG_GAP), NEG_GAP, 1 - BIG_GAP,
                               facecolor="#f2c94c", alpha=0.22, edgecolor="none", zorder=0))
    # gap is a max-min statistic, so the point can sit just outside its bootstrap
    # CI; clamp the whisker lengths to non-negative for plotting.
    lo = np.clip(mg - ci[:, 0], 0, None); hi = np.clip(ci[:, 1] - mg, 0, None)
    ax.errorbar(hg[~spur], mg[~spur], yerr=[lo[~spur], hi[~spur]], fmt="o",
                ms=6, color="#2f6fb0", ecolor="#9db8d2", capsize=2,
                label="segment gap (per question x axis)", zorder=3)
    ax.errorbar(hg[spur], mg[spur], yerr=[lo[spur], hi[spur]], fmt="D",
                ms=7, color="#c0392b", ecolor="#e0a99f", capsize=2,
                label="spurious split (no real gap)", zorder=4)
    lim = max(1.0, mg.max() + 0.05)
    ax.plot([0, lim], [0, lim], "--", color="0.4", lw=1.2, label="faithful (model = human)")
    ax.set_xlim(0, min(0.7, hg.max() + 0.05)); ax.set_ylim(0, lim)
    ax.set_xlabel("Human between-segment gap")
    ax.set_ylabel("Model between-segment gap")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = C.ANALYSIS_DIR / "fig_decision_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  ({spur.sum()} spurious-split pairs of {len(pairs)})")


if __name__ == "__main__":
    main()
