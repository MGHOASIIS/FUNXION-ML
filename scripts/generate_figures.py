"""
generate_figures.py
-------------------
Generates 6 publication-quality PNG figures from results_summary.csv.

Figures produced:
  fig1_main_results.png        — BA grouped bar chart: tasks × paradigms × models
  fig2_auc_ci.png              — AUC dot-plot with 95% CI whiskers
  fig3_ba_heatmap.png          — BA heatmap (tasks × paradigms, faceted by model)
  fig4_overfitting_matrix.png  — Overfitting risk matrix (colour-coded)
  fig5_feature_importance.png  — Top feature importance bar chart per model
  fig6_model_comparison.png    — Radar + summary bar comparison

Usage:
    python generate_figures.py [--csv results_summary.csv] [--out figures/]
    python generate_figures.py --demo    # uses built-in synthetic data
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────

FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update({
    "font.family":        FONT_FAMILY,
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.35,
    "grid.linewidth":     0.6,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

# Palette — accessible, print-friendly
MODEL_COLORS  = {"CNN": "#2166AC", "RNN": "#1A9641", "TRANSFORMER": "#D73027"}
MODEL_DISPLAY = {"CNN": "CNN",     "RNN": "RNN",      "TRANSFORMER": "Transformer"}
MODEL_MARKERS = {"CNN": "o",       "RNN": "s",         "TRANSFORMER": "D"}

TASK_LABELS = {
    1: "Jar Opening", 2: "Key Turning", 3: "Cleaning",
    4: "Back Washing", 5: "Cutting",    6: "Hammering"
}
PARADIGM_LABELS = {1: "P1\nPat vs Ctrl", 2: "P2\nRCT vs Ctrl",
                   3: "P3\nOther vs Ctrl", 4: "P4\nRCT vs Other"}
PARADIGM_SHORT  = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}

RISK_PALETTE = {"HIGH": "#D73027", "MEDIUM": "#FC8D59",
                "MODERATE": "#FC8D59", "LOW": "#1A9641", None: "#CCCCCC"}
RISK_ORDER   = ["LOW", "MEDIUM", "HIGH"]

# ── Synthetic demo data ───────────────────────────────────────────────────────

def make_demo_data():
    rng = np.random.default_rng(42)
    rows = []
    feat_pool = [
        "head_pos_x","head_pos_y","head_pos_z",
        "head_rot_x","head_rot_y","head_rot_z",
        "right_hand_pos_x","right_hand_pos_y","right_hand_pos_z",
        "right_hand_rot_x","right_hand_rot_y","right_hand_rot_z",
        "left_hand_pos_x","left_hand_pos_y","left_hand_pos_z",
        "left_hand_rot_x","left_hand_rot_y","left_hand_rot_z",
    ]
    model_bias = {"CNN": 0.02, "RNN": 0.04, "TRANSFORMER": 0.00}
    task_bias  = {1:.06, 2:.04, 3:.00, 4:.02, 5:-.02, 6:-.06}
    par_bias   = {1:.02, 2:.04, 3:-.02, 4:-.04}

    for task in range(1, 7):
        for par in range(1, 5):
            for model in ["CNN", "RNN", "TRANSFORMER"]:
                ba = np.clip(
                    0.70 + model_bias[model] + task_bias[task] + par_bias[par]
                    + rng.normal(0, 0.04), 0.50, 0.98)
                auc = np.clip(ba + rng.normal(0.02, 0.03), 0.50, 0.99)
                ci_w = rng.uniform(0.10, 0.20)
                f1   = np.clip(ba + rng.normal(0.00, 0.03), 0.40, 0.99)
                rec  = np.clip(ba + rng.normal(0.01, 0.04), 0.40, 0.99)
                prec = np.clip(ba + rng.normal(-0.01, 0.04), 0.40, 0.99)
                gap  = rng.uniform(0.05, 0.80)
                risk = "HIGH" if gap > 0.50 else ("MEDIUM" if gap > 0.25 else "LOW")
                feats_sorted = rng.choice(feat_pool, size=3, replace=False)
                scores = sorted(rng.dirichlet(np.ones(3)) * 0.18 + 0.04, reverse=True)
                rows.append({
                    "task": task, "task_name": TASK_LABELS[task],
                    "paradigm": par, "paradigm_name": PARADIGM_LABELS[par],
                    "model": model,
                    "ba": round(ba, 3), "auc": round(auc, 3),
                    "auc_ci_low":  round(auc - ci_w/2, 3),
                    "auc_ci_high": round(auc + ci_w/2, 3),
                    "f1": round(f1, 3), "recall": round(rec, 3),
                    "precision": round(prec, 3),
                    "overfitting_risk": risk,
                    "generalization_gap": round(gap, 3),
                    "top_feature_1_name": feats_sorted[0], "top_feature_1_score": scores[0],
                    "top_feature_2_name": feats_sorted[1], "top_feature_2_score": scores[1],
                    "top_feature_3_name": feats_sorted[2], "top_feature_3_score": scores[2],
                })
    return pd.DataFrame(rows)

# ── Figure 1: BA Grouped Bar Chart ────────────────────────────────────────────

def fig_main_results(df, out_dir):
    models  = ["CNN", "RNN", "TRANSFORMER"]
    tasks   = sorted(df["task"].unique())
    pars    = sorted(df["paradigm"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.suptitle("Balanced Accuracy Across Tasks and Paradigms", fontsize=14, fontweight="bold", y=1.01)

    x      = np.arange(len(tasks))
    width  = 0.22
    offset = [-width, 0, width]

    for ax, par in zip(axes.flat, pars):
        for i, model in enumerate(models):
            vals = []
            for task in tasks:
                sub = df[(df["task"] == task) & (df["paradigm"] == par) & (df["model"] == model)]
                vals.append(sub["ba"].values[0] if len(sub) else np.nan)
            bars = ax.bar(x + offset[i], vals, width, label=MODEL_DISPLAY[model],
                          color=MODEL_COLORS[model], alpha=0.88, zorder=3,
                          edgecolor="white", linewidth=0.5)
            # Value labels on top
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="#333333")

        ax.set_title(PARADIGM_LABELS[par].replace("\n", " — "), fontsize=10, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t].replace(" ", "\n") for t in tasks], fontsize=8)
        ax.set_ylim(0.40, 1.02)
        ax.set_ylabel("Balanced Accuracy" if par in [1, 3] else "")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Chance")
        ax.axhline(0.8, color="#AAAAAA", linestyle=":", linewidth=0.6, alpha=0.5)

    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_DISPLAY[m]) for m in models]
    handles.append(plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03),
               frameon=True, edgecolor="#CCCCCC")

    plt.tight_layout()
    path = out_dir / "fig1_main_results.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path

# ── Figure 2: AUC with 95% CI ─────────────────────────────────────────────────

def fig_auc_ci(df, out_dir):
    models = ["CNN", "RNN", "TRANSFORMER"]
    tasks  = sorted(df["task"].unique())
    pars   = sorted(df["paradigm"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(16, 7), sharey=True)
    fig.suptitle("AUC-ROC with 95% Bootstrap Confidence Intervals", fontsize=14, fontweight="bold")

    y_labels = [TASK_LABELS[t] for t in tasks]
    y_pos    = np.arange(len(tasks))
    offset   = [-0.22, 0, 0.22]

    for ax, par in zip(axes, pars):
        for i, model in enumerate(models):
            aucs, lo, hi = [], [], []
            for task in tasks:
                sub = df[(df["task"] == task) & (df["paradigm"] == par) & (df["model"] == model)]
                if len(sub):
                    aucs.append(sub["auc"].values[0])
                    lo.append(sub["auc_ci_low"].values[0])
                    hi.append(sub["auc_ci_high"].values[0])
                else:
                    aucs.append(np.nan); lo.append(np.nan); hi.append(np.nan)

            aucs, lo, hi = np.array(aucs), np.array(lo), np.array(hi)
            yp = y_pos + offset[i]
            ax.errorbar(aucs, yp,
                        xerr=[aucs - lo, hi - aucs],
                        fmt=MODEL_MARKERS[model], color=MODEL_COLORS[model],
                        capsize=3.5, capthick=1.2, elinewidth=1.2,
                        markersize=6, label=MODEL_DISPLAY[model],
                        zorder=4, alpha=0.9)

        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.axvline(0.8, color="#AAAAAA", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_title(PARADIGM_SHORT[par], fontsize=11, pad=8)
        ax.set_xlim(0.30, 1.05)
        ax.set_xlabel("AUC-ROC")
        ax.set_yticks(y_pos)
        if par == 1:
            ax.set_yticklabels(y_labels, fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        # Alternating row shading
        for j in range(len(tasks)):
            if j % 2 == 0:
                ax.axhspan(j - 0.45, j + 0.45, color="#F5F5F5", zorder=0)

    handles = [plt.Line2D([0], [0], marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                           linestyle="None", markersize=7, label=MODEL_DISPLAY[m])
               for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04),
               frameon=True, edgecolor="#CCCCCC")
    plt.tight_layout()

    path = out_dir / "fig2_auc_ci.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path

# ── Figure 3: BA Heatmap ──────────────────────────────────────────────────────

def fig_ba_heatmap(df, out_dir):
    models = ["CNN", "RNN", "TRANSFORMER"]
    tasks  = sorted(df["task"].unique())
    pars   = sorted(df["paradigm"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Balanced Accuracy Heatmap  (Tasks × Paradigms)", fontsize=14, fontweight="bold")

    cmap = LinearSegmentedColormap.from_list(
        "ba_cmap", ["#CB181D", "#FC8D59", "#FEE08B", "#91CF60", "#1A9850"], N=256)

    for ax, model in zip(axes, models):
        mat = np.full((len(tasks), len(pars)), np.nan)
        for ti, task in enumerate(tasks):
            for pi, par in enumerate(pars):
                sub = df[(df["task"] == task) & (df["paradigm"] == par) & (df["model"] == model)]
                if len(sub):
                    mat[ti, pi] = sub["ba"].values[0]

        im = ax.imshow(mat, cmap=cmap, vmin=0.50, vmax=0.95, aspect="auto")

        # Annotate cells
        for ti in range(len(tasks)):
            for pi in range(len(pars)):
                v = mat[ti, pi]
                if not np.isnan(v):
                    # Star the best cell
                    is_best = (v == np.nanmax(mat))
                    txt = f"{v:.2f}" + (" ★" if is_best else "")
                    col = "white" if v > 0.78 or v < 0.57 else "#222222"
                    ax.text(pi, ti, txt, ha="center", va="center",
                            fontsize=8.5, color=col, fontweight="bold" if is_best else "normal")

        ax.set_title(MODEL_DISPLAY[model], fontsize=11, pad=10)
        ax.set_xticks(range(len(pars)))
        ax.set_xticklabels([PARADIGM_SHORT[p] for p in pars], fontsize=9)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([TASK_LABELS[t] for t in tasks] if model == "CNN" else [], fontsize=9)
        ax.set_xlabel("Paradigm")

    # Shared colourbar
    cb = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.04)
    cb.set_label("Balanced Accuracy", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = out_dir / "fig3_ba_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path

# ── Figure 4: Overfitting Risk Matrix ─────────────────────────────────────────

def fig_overfitting(df, out_dir):
    models = ["CNN", "RNN", "TRANSFORMER"]
    tasks  = sorted(df["task"].unique())
    pars   = sorted(df["paradigm"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Overfitting Risk Matrix  (Generalisation Gap)", fontsize=14, fontweight="bold")

    risk_num = {"LOW": 0, "MEDIUM": 1, "MODERATE": 1, "HIGH": 2}
    cmap_risk = matplotlib.colors.ListedColormap(["#1A9641", "#FC8D59", "#D73027"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap_risk.N)

    for ax, model in zip(axes, models):
        mat_num = np.full((len(tasks), len(pars)), -1.0)
        mat_lbl = [["" for _ in pars] for _ in tasks]

        for ti, task in enumerate(tasks):
            for pi, par in enumerate(pars):
                sub = df[(df["task"] == task) & (df["paradigm"] == par) & (df["model"] == model)]
                if len(sub):
                    risk = str(sub["overfitting_risk"].values[0]).upper()
                    gap  = sub["generalization_gap"].values[0] if "generalization_gap" in sub.columns else np.nan
                    mat_num[ti, pi] = risk_num.get(risk, -1)
                    gap_str = f"\n({gap:.2f})" if not np.isnan(gap) else ""
                    mat_lbl[ti][pi] = risk.capitalize() + gap_str

        im = ax.imshow(mat_num, cmap=cmap_risk, norm=norm, aspect="auto")

        for ti in range(len(tasks)):
            for pi in range(len(pars)):
                lbl = mat_lbl[ti][pi]
                if lbl:
                    v = mat_num[ti, pi]
                    col = "white" if v == 2 else "#111111"
                    ax.text(pi, ti, lbl, ha="center", va="center",
                            fontsize=7.5, color=col, linespacing=1.4)

        ax.set_title(MODEL_DISPLAY[model], fontsize=11, pad=10)
        ax.set_xticks(range(len(pars)))
        ax.set_xticklabels([PARADIGM_SHORT[p] for p in pars], fontsize=9)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([TASK_LABELS[t] for t in tasks] if model == "CNN" else [], fontsize=9)

    legend_patches = [
        mpatches.Patch(color="#1A9641", label="Low risk"),
        mpatches.Patch(color="#FC8D59", label="Medium risk"),
        mpatches.Patch(color="#D73027", label="High risk"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), frameon=True, edgecolor="#CCCCCC", fontsize=9)
    plt.tight_layout()

    path = out_dir / "fig4_overfitting_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path

# ── Figure 5: Feature Importance ─────────────────────────────────────────────

def fig_feature_importance(df, out_dir):
    models = ["CNN", "RNN", "TRANSFORMER"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    fig.suptitle("Top Feature Importances by Model\n(Mean Permutation Importance Across All Tasks & Paradigms)",
                 fontsize=13, fontweight="bold")

    # Category colour per sensor group
    def sensor_color(name):
        n = str(name).lower()
        if n.startswith("head"):     return "#2166AC"
        if n.startswith("right"):    return "#1A9641"
        if n.startswith("left"):     return "#D73027"
        return "#999999"

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        feat_agg = {}
        for _, row in sub.iterrows():
            for rank in range(1, 4):
                fn = row.get(f"top_feature_{rank}_name")
                fs = row.get(f"top_feature_{rank}_score")
                if fn and str(fn) != "nan":
                    feat_agg.setdefault(fn, []).append(float(fs))

        feat_mean = {f: np.mean(v) for f, v in feat_agg.items()}
        feat_count = {f: len(v) for f, v in feat_agg.items()}

        # Sort by mean importance, top 10
        top = sorted(feat_mean, key=feat_mean.get, reverse=True)[:10]
        vals   = [feat_mean[f] * 100 for f in top]
        counts = [feat_count[f] for f in top]
        colors = [sensor_color(f) for f in top]

        # Pretty labels
        labels = [f.replace("_", " ").title() for f in top]

        y = np.arange(len(top))
        bars = ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v, cnt in zip(bars, vals, counts):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f"{v:.1f}%  (n={cnt})", va="center", fontsize=7.5, color="#333333")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Mean Importance (%)")
        ax.set_title(MODEL_DISPLAY[model], fontsize=11, pad=8)
        ax.set_xlim(0, max(vals) * 1.45 if vals else 10)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", alpha=0)

    legend_patches = [
        mpatches.Patch(color="#2166AC", label="Head"),
        mpatches.Patch(color="#1A9641", label="Right Hand"),
        mpatches.Patch(color="#D73027", label="Left Hand"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor="#CCCCCC", fontsize=9)
    plt.tight_layout()

    path = out_dir / "fig5_feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path

# ── Figure 6: Model Comparison (Radar + Bar) ─────────────────────────────────

def fig_model_comparison(df, out_dir):
    models  = ["CNN", "RNN", "TRANSFORMER"]
    metrics = ["ba", "auc", "f1", "recall", "precision"]
    metric_labels = ["BA", "AUC", "F1", "Recall", "Precision"]

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle("Overall Model Performance Comparison", fontsize=14, fontweight="bold")

    gs = GridSpec(1, 3, figure=fig, wspace=0.38)
    ax_radar = fig.add_subplot(gs[0], polar=True)
    ax_bar   = fig.add_subplot(gs[1])
    ax_over  = fig.add_subplot(gs[2])

    # ── Radar ─────────────────────────────────────
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_rlim(0.45, 0.95)
    ax_radar.set_rticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax_radar.set_yticklabels(["0.5","0.6","0.7","0.8","0.9"], fontsize=7)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_labels, fontsize=9)
    ax_radar.set_title("Mean Metrics\n(All Tasks & Paradigms)", fontsize=10, pad=18)

    for model in models:
        sub = df[df["model"] == model]
        vals = [pd.to_numeric(sub[m], errors="coerce").mean() for m in metrics]
        vals += vals[:1]
        ax_radar.plot(angles, vals, color=MODEL_COLORS[model], linewidth=2,
                      label=MODEL_DISPLAY[model])
        ax_radar.fill(angles, vals, color=MODEL_COLORS[model], alpha=0.12)

    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8.5)

    # ── Grouped bar: mean BA per task ─────────────
    tasks = sorted(df["task"].unique())
    x = np.arange(len(tasks))
    w = 0.25
    offsets = [-w, 0, w]

    for i, model in enumerate(models):
        vals = []
        for task in tasks:
            sub = df[(df["task"] == task) & (df["model"] == model)]
            vals.append(pd.to_numeric(sub["ba"], errors="coerce").mean())
        ax_bar.bar(x + offsets[i], vals, w, label=MODEL_DISPLAY[model],
                   color=MODEL_COLORS[model], alpha=0.85, edgecolor="white")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([TASK_LABELS[t].replace(" ", "\n") for t in tasks], fontsize=8)
    ax_bar.set_ylabel("Mean BA (across paradigms)")
    ax_bar.set_ylim(0.45, 0.95)
    ax_bar.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_bar.set_title("Mean BA per Task", fontsize=10, pad=8)
    ax_bar.legend(fontsize=8.5)

    # ── Overfitting stacked bar ───────────────────
    risk_cats  = ["LOW", "MEDIUM", "HIGH"]
    risk_cols  = ["#1A9641", "#FC8D59", "#D73027"]
    risk_display = ["Low", "Medium", "High"]

    x_m = np.arange(len(models))
    bottoms = np.zeros(len(models))
    for rcat, rcol, rdis in zip(risk_cats, risk_cols, risk_display):
        counts = []
        for model in models:
            sub = df[df["model"] == model]
            col = sub["overfitting_risk"].str.upper() if "overfitting_risk" in sub.columns else pd.Series(dtype=str)
            counts.append((col == rcat).sum())
        ax_over.bar(x_m, counts, bottom=bottoms, color=rcol, alpha=0.85,
                    label=rdis, edgecolor="white")
        for xi, (cnt, bot) in enumerate(zip(counts, bottoms)):
            if cnt > 0:
                ax_over.text(xi, bot + cnt / 2, str(int(cnt)),
                             ha="center", va="center", fontsize=9,
                             color="white" if rcat == "HIGH" else "#111111", fontweight="bold")
        bottoms += np.array(counts, dtype=float)

    ax_over.set_xticks(x_m)
    ax_over.set_xticklabels([MODEL_DISPLAY[m] for m in models], fontsize=9)
    ax_over.set_ylabel("Number of experiments")
    ax_over.set_title("Overfitting Risk Distribution", fontsize=10, pad=8)
    ax_over.legend(fontsize=8.5, title="Risk", title_fontsize=8)
    ax_over.set_ylim(0, len(df) // len(models) + 2)

    path = out_dir / "fig6_model_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  default="nn-models-results/results_summary.csv")
    parser.add_argument("--out",  default="nn-models-results/figures")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data instead of CSV")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("Using synthetic demo data …")
        df = make_demo_data()
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[ERROR] {csv_path} not found — run with --demo to test")
            return
        df = pd.read_csv(csv_path)
        df["model"] = df["model"].str.upper()
        print(f"Loaded {len(df)} rows from {csv_path}")

    print(f"\nGenerating figures → {out_dir.resolve()}\n")
    paths = []
    paths.append(fig_main_results(df, out_dir))
    paths.append(fig_auc_ci(df, out_dir))
    paths.append(fig_ba_heatmap(df, out_dir))
    paths.append(fig_overfitting(df, out_dir))
    paths.append(fig_feature_importance(df, out_dir))
    paths.append(fig_model_comparison(df, out_dir))

    print(f"\nDone — {len(paths)} figures saved to {out_dir.resolve()}")
    return paths

if __name__ == "__main__":
    main()