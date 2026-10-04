"""Question-level variation and confidence-interval figures.

Fig A: per-question stereotyping index (Delta eta^2) for the frontier model
       (Sonnet 4.6), with bootstrap 95% CI error bars, so the reader sees that
       over-determination is not a model-average artifact but holds question by
       question. One panel per domain.

Fig B: per-model individual-accuracy margin over the demographic baseline
       (Delta_base) with paired-bootstrap 95% CI error bars, both domains.

Reads analysis/<DS>/flattening_<model>.json and supplementary_metrics.json.
Writes analysis/<DS>/fig_question_stereo.png and (shared) fig_delta_base_ci.png.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# slightly larger fonts for readability; titles are omitted (paper captions carry them)
plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 11,
})

import config as C

FRONTIER = "claude-sonnet-4.6"
MODELS = ["claude-haiku-4.5", "claude-sonnet-4.6", "llama3.1-8b", "llama3.3-70b"]
MLAB = {"claude-haiku-4.5": "Haiku 4.5", "claude-sonnet-4.6": "Sonnet 4.6",
        "llama3.1-8b": "Llama 8B", "llama3.3-70b": "Llama 70B"}


def question_stereo_fig():
    """Per-question stereotyping index for the frontier model (Style A) on the
    primary axis (GSS: polviews, WVS: country), with 95% CI error bars."""
    flat = json.loads((C.ANALYSIS_DIR / f"flattening_{FRONTIER}.json").read_text())
    style = "A"
    per_q = flat[style]["by_question"]
    # choose the primary axis per dataset
    axis = "polviews" if C.DATASET == "GSS" else "country"
    labels, vals, los, his = [], [], [], []
    for qid, axes in per_q.items():
        a = axes.get(axis)
        if a is None:
            # fall back to first available axis for that question
            if not axes:
                continue
            a = list(axes.values())[0]
        ci = a["index_ci95"]
        labels.append(qid)
        vals.append(a["stereotyping_index"])
        if ci[0] is not None:
            los.append(a["stereotyping_index"] - ci[0]); his.append(ci[1] - a["stereotyping_index"])
        else:
            los.append(0); his.append(0)
    order = np.argsort(vals)[::-1]
    labels = [labels[i] for i in order]; vals = [vals[i] for i in order]
    los = [los[i] for i in order]; his = [his[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    xpos = np.arange(len(labels))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in vals]
    ax.bar(xpos, vals, yerr=[los, his], capsize=3, color=colors, edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(r"stereotyping index $\Delta\eta^2$ (95% CI)")
    fig.tight_layout()
    out = C.ANALYSIS_DIR / "fig_question_stereo.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def delta_base_fig():
    rm2 = json.loads((C.ANALYSIS_DIR / "supplementary_metrics.json").read_text())
    db = rm2["delta_base_ci"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xpos = np.arange(len(MODELS)); w = 0.38
    for j, style in enumerate(["A", "C"]):
        vals, los, his = [], [], []
        for m in MODELS:
            d = db.get(m, {}).get(style)
            if d is None:
                vals.append(np.nan); los.append(0); his.append(0); continue
            v = d["delta_base"]; ci = d["delta_base_ci95"]
            vals.append(v); los.append(v - ci[0]); his.append(ci[1] - v)
        off = (-w/2 if style == "A" else w/2)
        color = "#4C72B0" if style == "A" else "#DD8452"
        lab = "single-answer prompt" if style == "A" else "distribution prompt"
        ax.bar(xpos + off, vals, w, yerr=[los, his], capsize=3, color=color, label=lab,
               edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(xpos); ax.set_xticklabels([MLAB[m] for m in MODELS], fontsize=12)
    ax.set_ylabel(r"accuracy margin over demographic baseline $\Delta_{\mathrm{base}}$ (95% CI)")
    ax.legend()
    fig.tight_layout()
    out = C.ANALYSIS_DIR / "fig_delta_base_ci.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    question_stereo_fig()
    delta_base_fig()
