"""Cross-model pilot summary + figures.

Consumes, for every model with a completed run:
  analysis/pilot_metrics_<model>.json   (metrics.py)
  analysis/flattening_<model>.json       (analyze_subgroups.py)

Both are recomputed from the cached model outputs on every run.
Produces:
  analysis/summary_table.csv / .md   — one row per (model, style)
  analysis/fig_individual_vs_baseline.png
  analysis/fig_aggregate_js.png
  analysis/fig_stereotyping_index.png

All numbers come straight from the analysis artifacts — nothing hand-entered.
"""
import json
import sys

import numpy as np
import pandas as pd

import config as C
import metrics as M
import analyze_subgroups as S

# models to include if their run cache exists
CANDIDATE_MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6", "llama3.1-8b", "llama3.3-70b"]


def available_models():
    d = C.BUILD_DIR / "llm_runs"
    out = []
    for m in CANDIDATE_MODELS:
        p = d / f"{m}.jsonl"
        if p.exists() and len(p.read_text().splitlines()) > 0:
            out.append(m)
    return out


_computed = {}


def ensure_analyses(model):
    """Recompute pilot_metrics_<model>.json and flattening_<model>.json from the
    cached model outputs (once per run) and return both."""
    if model in _computed:
        return _computed[model]
    mfile = C.ANALYSIS_DIR / f"pilot_metrics_{model}.json"
    ffile = C.ANALYSIS_DIR / f"flattening_{model}.json"
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    mfile.write_text(json.dumps(M.compute(model), indent=2))
    runs = S.load_runs(model)
    H = json.loads((C.BUILD_DIR / "human_dist.json").read_text())
    pilot = pd.read_parquet(C.BUILD_DIR / "pilot_sample.parquet")
    demo = pilot.set_index(["respondent_id", "question_id"])[C.SUBGROUP_VARS].to_dict("index")
    result = {}
    for style in sorted({r["style"] for r in runs}):
        srecs = [r for r in runs if r["style"] == style]
        per_q = S.analyze_style(srecs, H, demo, style)
        idxs = [a["stereotyping_index"] for q in per_q.values() for a in q.values()]
        sig = [a for q in per_q.values() for a in q.values()
               if a["index_ci95"][0] is not None and (a["index_ci95"][0] > 0 or a["index_ci95"][1] < 0)]
        result[style] = {
            "median_stereotyping_index": round(float(np.median(idxs)), 3) if idxs else None,
            "n_question_axis_pairs": len(idxs),
            "n_pairs_ci_excludes_0": len(sig),
            "n_pairs_stereotyping": sum(1 for x in idxs if x > 0),
            "n_pairs_flattening": sum(1 for x in idxs if x < 0),
            "by_question": per_q,
        }
    ffile.write_text(json.dumps(result, indent=2))
    _computed[model] = (json.loads(mfile.read_text()), json.loads(ffile.read_text()))
    return _computed[model]


def build_table(models):
    rows = []
    for m in models:
        met, flat = ensure_analyses(m)
        for style in ("A", "C"):
            if style not in met:
                continue
            mm = met[style]
            ff = flat.get(style, {})
            rows.append({
                "model": m, "style": style,
                "n": mm["n_predictions"],
                "invalid_rate": mm["invalid_rate"],
                "indiv_acc": mm["individual_accuracy"],
                "baseline_acc": mm["baseline_accuracy"],
                "acc_minus_baseline": mm["acc_minus_baseline"],
                "mean_js_div": mm["mean_js_divergence"],
                "median_stereotyping_index": ff.get("median_stereotyping_index"),
                "n_pairs_stereotyping": ff.get("n_pairs_stereotyping"),
                "n_pairs_total": ff.get("n_question_axis_pairs"),
                "n_pairs_ci_excl_0": ff.get("n_pairs_ci_excludes_0"),
            })
    return pd.DataFrame(rows)


# display labels for the models (short, human-readable), in a fixed order
MODEL_LABELS = {
    "claude-haiku-4.5": "Claude\nHaiku 4.5",
    "claude-sonnet-4.6": "Claude\nSonnet 4.6",
    "llama3.1-8b": "Llama-3.1\n8B",
    "llama3.3-70b": "Llama-3.3\n70B",
}
# Style-C cells with high invalid rates are unreliable; do not plot them as if solid.
INVALID_FLAG = 0.20   # if a cell's invalid_rate exceeds this, mark/suppress it


def _labels(models):
    return [MODEL_LABELS.get(m, m) for m in models]


def _bar_value_labels(ax, bars, fmt="{:.3f}", skip=None):
    """Annotate each bar with its value above the bar. skip: set of indices to skip."""
    for i, b in enumerate(bars):
        if skip and i in skip:
            continue
        h = b.get_height()
        if h is None or np.isnan(h):
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9,
                    xytext=(0, 1.5), textcoords="offset points")


def make_figures(models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # slightly larger fonts for readability in the paper (no titles: captions carry them)
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "legend.fontsize": 11,
    })

    data = {m: ensure_analyses(m) for m in models}
    ds = C.DATASET
    x = np.arange(len(models))
    w = 0.38
    labels = _labels(models)

    # ---- Fig 1: individual accuracy vs baseline ----
    accA = [data[m][0]["A"]["individual_accuracy"] for m in models]
    accC = [data[m][0]["C"]["individual_accuracy"] for m in models]
    base = float(np.mean([data[m][0]["A"]["baseline_accuracy"] for m in models]))
    # data-driven y-limits so bars are always visible
    vals = accA + accC + [base]
    lo = max(0.0, min(vals) - 0.08)
    hi = min(1.0, max(vals) + 0.08)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bA = ax.bar(x - w/2, accA, w, label="single-answer prompt", color="#4C72B0")
    bC = ax.bar(x + w/2, accC, w, label="distribution prompt", color="#DD8452")
    ax.axhline(base, ls="--", color="k", lw=1.4,
               label=f"demographic-baseline accuracy ({base:.3f})")
    _bar_value_labels(ax, bA); _bar_value_labels(ax, bC)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("individual-level accuracy"); ax.set_ylim(lo, hi)
    ax.legend(loc="best")
    ax.margins(x=0.02)
    fig.tight_layout(); fig.savefig(C.ANALYSIS_DIR / "fig_individual_vs_baseline.png", dpi=150)
    plt.close(fig)

    # ---- Fig 2: aggregate JS divergence ----
    jsA = [data[m][0]["A"]["mean_js_divergence"] for m in models]
    jsC = [data[m][0]["C"]["mean_js_divergence"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bA = ax.bar(x - w/2, jsA, w, label="single-answer prompt", color="#4C72B0")
    bC = ax.bar(x + w/2, jsC, w, label="distribution prompt", color="#DD8452")
    _bar_value_labels(ax, bA); _bar_value_labels(ax, bC)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean Jensen-Shannon divergence (lower = closer to humans)")
    ax.set_ylim(0, max(jsA + jsC) * 1.18)
    ax.legend(loc="best")
    ax.margins(x=0.02)
    fig.tight_layout(); fig.savefig(C.ANALYSIS_DIR / "fig_aggregate_js.png", dpi=150)
    plt.close(fig)

    # ---- Fig 3: median stereotyping index ----
    # suppress cells whose style-C invalid rate is high (unreliable), and note it.
    siA = [data[m][1].get("A", {}).get("median_stereotyping_index") for m in models]
    siC = []
    supp = []
    for i, m in enumerate(models):
        inv = data[m][0].get("C", {}).get("invalid_rate", 0.0)
        val = data[m][1].get("C", {}).get("median_stereotyping_index")
        if inv is not None and inv > INVALID_FLAG:
            siC.append(np.nan); supp.append(i)
        else:
            siC.append(val)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bA = ax.bar(x - w/2, siA, w, label="single-answer prompt", color="#4C72B0")
    bC = ax.bar(x + w/2, siC, w, label="distribution prompt", color="#DD8452")
    ax.axhline(0, color="k", lw=1.0)
    _bar_value_labels(ax, bA); _bar_value_labels(ax, bC, skip=set(supp))
    # mark suppressed (unreliable) style-C cells
    ymax = np.nanmax(siA + [v for v in siC if v is not None and not np.isnan(v)] + [0.01])
    for i in supp:
        ax.annotate("output\nunreliable", (x[i] + w/2, 0.0), ha="center", va="bottom",
                    fontsize=9, color="gray", xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"stereotyping index ($\eta^2_{model}-\eta^2_{human}$)")
    ax.set_ylim(min(0, np.nanmin(siA + siC)) - 0.01, ymax * 1.20)
    ax.legend(loc="best")
    ax.margins(x=0.02)
    fig.tight_layout(); fig.savefig(C.ANALYSIS_DIR / "fig_stereotyping_index.png", dpi=150)
    plt.close(fig)


def main():
    models = available_models()
    print("models with runs:", models)
    df = build_table(models)
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.ANALYSIS_DIR / "summary_table.csv", index=False)
    (C.ANALYSIS_DIR / "summary_table.md").write_text(df.to_markdown(index=False))
    make_figures(models)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote summary_table.{{csv,md}} and 3 figures to {C.ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
