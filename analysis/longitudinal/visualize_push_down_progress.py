"""
visualize_push_down_progress.py
================================
Longitudinal progress visualization for "Push Down Exercise" across sessions 1-4.

Reads push_down_longitudinal.csv and produces a multi-page PDF + individual PNGs.

Figures
-------
  1. Overview heatmap   — top 40 monotonic features, % change colour-coded
  2. Bilateral coordination — knee/hand/hip left-right correlation trends
  3. Range of motion        — hands reaching lower + knee/head ROM spread
  4. Posture                — back & head rotation baselines
  5. Movement complexity    — spectral entropy (hands + hips)
  6. Signal shape           — kurtosis normalisation (back + knee)
  7. Radar / spider chart   — 6 summary dimensions per session

Usage
-----
    python analysis/longitudinal/visualize_push_down_progress.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

HERE    = Path(__file__).resolve().parent
IN_CSV  = HERE / "push_down_longitudinal.csv"
OUT_DIR = HERE / "progress_plots"
OUT_PDF = HERE / "push_down_progress.pdf"
OUT_DIR.mkdir(exist_ok=True)

SESSIONS   = [1, 2, 3, 4]
SCOLS      = [f"session_{s}" for s in SESSIONS]
PALETTE    = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]   # one colour per session
SESSION_LABELS = [f"Session {s}" for s in SESSIONS]

# ── helpers ──────────────────────────────────────────────────────────────────

def load(feat_names):
    """Pull rows for given feature names from the longitudinal CSV."""
    df = pd.read_csv(IN_CSV)
    sub = df[df["feature"].isin(feat_names)].set_index("feature")
    return sub[SCOLS]


def trend_ax(ax, data_rows, labels, title, ylabel="", legend=True):
    """
    Line plot: x = sessions, one line per feature.
    data_rows : DataFrame with rows = features, cols = session_1..4
    labels    : display name per feature (same order as index)
    """
    x = np.arange(1, 5)
    for i, (feat, row) in enumerate(data_rows.iterrows()):
        vals = row[SCOLS].values.astype(float)
        color = plt.cm.tab10(i / max(len(data_rows) - 1, 1))
        ax.plot(x, vals, "o-", color=color, linewidth=2, markersize=7,
                label=labels[i] if labels else feat)
    ax.set_xticks(x)
    ax.set_xticklabels(SESSION_LABELS, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(fontsize=8, framealpha=0.7)


def bar_change_ax(ax, features, pct_changes, colors, title):
    """Horizontal bar chart of % change."""
    y = np.arange(len(features))
    bars = ax.barh(y, pct_changes, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% change  (session 1 → 4)", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, pct_changes):
        ax.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
                fontsize=7.5)


# ── figure 1 : overview heatmap ───────────────────────────────────────────────

def fig_heatmap():
    df = pd.read_csv(IN_CSV)
    mono = df[
        (df["monotonic"] == True) &
        (df["session_1"].abs() > 0.01) &
        (df["pct_change_1_to_4"].abs() < 500) &
        df["pct_change_1_to_4"].notna()
    ].copy()
    mono["abs_pct"] = mono["pct_change_1_to_4"].abs()
    top = mono.nlargest(40, "abs_pct").set_index("feature")

    # normalise each row to [0,1] for colour
    vals = top[SCOLS].values.astype(float)
    row_min = vals.min(axis=1, keepdims=True)
    row_max = vals.max(axis=1, keepdims=True)
    norm = (vals - row_min) / (row_max - row_min + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 13))
    im = ax.imshow(norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SESSION_LABELS, fontsize=10)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=7.5)
    ax.set_title("Top 40 monotonic features — row-normalised progress\n"
                 "(green = higher value, red = lower)", fontsize=12, fontweight="bold")

    # annotate pct change on right
    for i, (feat, row) in enumerate(top.iterrows()):
        pct = row["pct_change_1_to_4"]
        ax.text(4.15, i, f"{pct:+.0f}%", va="center", fontsize=6.5,
                color="#2c7bb6" if pct > 0 else "#d7191c")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.12, label="row-normalised value")
    fig.tight_layout()
    return fig


# ── figure 2 : bilateral coordination ────────────────────────────────────────

def fig_coordination():
    feats = {
        "Knee RotX corr":   "knee_lr_correlation_axis3",
        "Hand PosY corr":   "hand_lr_correlation_axis1",
        "Hand PosX corr":   "hand_lr_correlation_axis0",
        "Hip PosY corr":    "hip_lr_correlation_axis1",
        "Knee PosY corr":   "knee_lr_correlation_axis1",
    }
    available = {k: v for k, v in feats.items()
                 if v in pd.read_csv(IN_CSV)["feature"].values}

    data = load(list(available.values()))
    labels = list(available.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    trend_ax(axes[0], data, labels,
             title="Bilateral Coordination — L/R Correlation",
             ylabel="Pearson correlation")

    # bar chart of % change
    df_all = pd.read_csv(IN_CSV).set_index("feature")
    pcts   = [float(df_all.loc[v, "pct_change_1_to_4"]) for v in available.values()]
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pcts]
    bar_change_ax(axes[1], list(available.keys()), pcts, colors,
                  "% change session 1 → 4")

    fig.suptitle("Bilateral Coordination", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── figure 3 : range of motion ────────────────────────────────────────────────

def fig_rom():
    feats = {
        "LeftHand PosY min":    "LeftHand_PosY_min",
        "RightHand PosY min":   "RightHand_PosY_min",
        "LeftHand PosY IQR":    "LeftHand_PosY_iqr",
        "LeftKnee RotX IQR":    "LeftKnee_RotX_iqr",
        "Head PosY range":      "Head_PosY_range",
        "LeftKnee RotX std":    "LeftKnee_RotX_std",
    }
    avail = {k: v for k, v in feats.items()
             if v in pd.read_csv(IN_CSV)["feature"].values}

    data   = load(list(avail.values()))
    labels = list(avail.keys())
    df_all = pd.read_csv(IN_CSV).set_index("feature")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    trend_ax(axes[0], data, labels,
             title="Range of Motion trends",
             ylabel="value (m or °)")

    pcts   = [float(df_all.loc[v, "pct_change_1_to_4"]) for v in avail.values()]
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pcts]
    bar_change_ax(axes[1], list(avail.keys()), pcts, colors,
                  "% change session 1 → 4")

    fig.suptitle("Range of Motion", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── figure 4 : posture ────────────────────────────────────────────────────────

def fig_posture():
    feats = {
        "Head RotY min":          "Head_RotY_min",
        "Head RotY mean":         "Head_RotY_mean",
        "Back RotX mean":         "Back_RotX_mean",
        "Back RotZ pct25":        "Back_RotZ_percentile_25",
        "Head PosZ energy":       "Head_PosZ_energy",
        "Head PosZ sma":          "Head_PosZ_sma",
    }
    avail = {k: v for k, v in feats.items()
             if v in pd.read_csv(IN_CSV)["feature"].values}

    data   = load(list(avail.values()))
    labels = list(avail.keys())
    df_all = pd.read_csv(IN_CSV).set_index("feature")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    trend_ax(axes[0], data, labels,
             title="Posture & Back Rotation trends",
             ylabel="value (° or energy units)")

    pcts   = [float(df_all.loc[v, "pct_change_1_to_4"]) for v in avail.values()]
    colors = ["#2ecc71" if p < 0 else "#e74c3c" for p in pcts]   # decreasing = good here
    bar_change_ax(axes[1], list(avail.keys()), pcts, colors,
                  "% change session 1 → 4")

    fig.suptitle("Posture", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── figure 5 : movement complexity ───────────────────────────────────────────

def fig_complexity():
    feats = {
        "RightHand spectral entropy":  "RightHand_entropy_spectral_entropy",
        "LeftHand spectral entropy":   "LeftHand_entropy_spectral_entropy",
        "LeftHip perm entropy":        "LeftHip_entropy_perm_entropy",
        "RightHand freq spectral kurt":"RightHand_freq_spectral_kurtosis",
        "RightHand freq skewness":     "RightHand_freq_spectral_skewness",
        "LeftHand freq skewness":      "LeftHand_freq_spectral_skewness",
    }
    avail = {k: v for k, v in feats.items()
             if v in pd.read_csv(IN_CSV)["feature"].values}

    data   = load(list(avail.values()))
    labels = list(avail.keys())
    df_all = pd.read_csv(IN_CSV).set_index("feature")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    trend_ax(axes[0], data, labels,
             title="Movement Complexity / Entropy trends",
             ylabel="entropy / kurtosis")

    pcts   = [float(df_all.loc[v, "pct_change_1_to_4"]) for v in avail.values()]
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pcts]
    bar_change_ax(axes[1], list(avail.keys()), pcts, colors,
                  "% change session 1 → 4")

    fig.suptitle("Movement Complexity & Entropy", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── figure 6 : signal shape normalisation ────────────────────────────────────

def fig_signal_shape():
    feats = {
        "Back RotZ kurtosis":      "Back_RotZ_kurtosis",
        "Back RotY skewness":      "Back_RotY_skewness",
        "LeftHand PosY kurtosis":  "LeftHand_PosY_kurtosis",
        "RightKnee PosZ kurtosis": "RightKnee_PosZ_kurtosis",
        "Back RotZ skewness":      "Back_RotZ_skewness",
    }
    avail = {k: v for k, v in feats.items()
             if v in pd.read_csv(IN_CSV)["feature"].values}

    data   = load(list(avail.values()))
    labels = list(avail.keys())
    df_all = pd.read_csv(IN_CSV).set_index("feature")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    trend_ax(axes[0], data, labels,
             title="Signal Shape — kurtosis & skewness normalisation",
             ylabel="kurtosis / skewness")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    pcts   = [float(df_all.loc[v, "pct_change_1_to_4"]) for v in avail.values()]
    colors = ["#2ecc71" if abs(p) > 50 else "#3498db" for p in pcts]
    bar_change_ax(axes[1], list(avail.keys()), pcts, colors,
                  "% change session 1 → 4")

    fig.suptitle("Signal Shape Normalisation", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── figure 7 : radar chart ────────────────────────────────────────────────────

def fig_radar():
    """
    6 summary dimensions, each scored 0-100 (session 1 baseline = 0,
    best observed = 100).
    """
    dimensions = {
        "Bilateral\nCoordination": "knee_lr_correlation_axis3",
        "Hand ROM\n(reach lower)": "LeftHand_PosY_iqr",
        "Knee ROM\nspread":        "LeftKnee_RotX_iqr",
        "Movement\nComplexity":    "RightHand_entropy_spectral_entropy",
        "Posture\nStability":      "Head_PosZ_energy",    # lower = better → invert
        "Back\nRotation":          "Back_RotX_mean",      # lower = better → invert
    }
    INVERT = {"Head_PosZ_energy", "Back_RotX_mean"}   # decreasing = progress

    df = pd.read_csv(IN_CSV).set_index("feature")
    dim_names = list(dimensions.keys())
    feat_keys = list(dimensions.values())

    raw = np.array([
        [float(df.loc[fk, sc]) for sc in SCOLS]
        for fk in feat_keys
    ])   # shape (n_dims, n_sessions)

    # normalise to 0-100 across sessions
    scores = np.zeros_like(raw)
    for i, fk in enumerate(feat_keys):
        lo, hi = raw[i].min(), raw[i].max()
        if hi == lo:
            scores[i] = 50
        else:
            norm = (raw[i] - lo) / (hi - lo) * 100
            scores[i] = (100 - norm) if fk in INVERT else norm

    N   = len(dim_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_names, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="grey")
    ax.yaxis.set_tick_params(labelsize=7)

    for s_idx, (label, color) in enumerate(zip(SESSION_LABELS, PALETTE)):
        vals  = scores[:, s_idx].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title("Progress radar — Push Down Exercise\n(100 = best observed across sessions)",
                 fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


# ── figure 8 : small multiples — top 12 monotonic features ───────────────────

def fig_small_multiples():
    df_all = pd.read_csv(IN_CSV)
    mono   = df_all[
        (df_all["monotonic"] == True) &
        (df_all["session_1"].abs() > 0.01) &
        (df_all["pct_change_1_to_4"].abs() < 500) &
        df_all["pct_change_1_to_4"].notna()
    ].copy()
    mono["abs_pct"] = mono["pct_change_1_to_4"].abs()
    top12 = mono.nlargest(12, "abs_pct")

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    x = np.arange(1, 5)

    for i, (_, row) in enumerate(top12.iterrows()):
        ax  = axes[i]
        vals = row[SCOLS].values.astype(float)
        pct  = row["pct_change_1_to_4"]
        color = "#2ecc71" if pct > 0 else "#e74c3c"

        ax.plot(x, vals, "o-", color=color, linewidth=2.5, markersize=8)
        ax.fill_between(x, vals, alpha=0.12, color=color)

        # shade the trend arrow
        ax.annotate("", xy=(4, vals[-1]), xytext=(1, vals[0]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

        ax.set_xticks(x)
        ax.set_xticklabels(["S1","S2","S3","S4"], fontsize=8)
        ax.set_title(row["feature"], fontsize=7.5, fontweight="bold", pad=4)
        ax.text(0.97, 0.05, f"{pct:+.1f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                color=color, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Top 12 monotonic features — Push Down Exercise",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    figures = [
        ("01_heatmap",        fig_heatmap),
        ("02_coordination",   fig_coordination),
        ("03_rom",            fig_rom),
        ("04_posture",        fig_posture),
        ("05_complexity",     fig_complexity),
        ("06_signal_shape",   fig_signal_shape),
        ("07_radar",          fig_radar),
        ("08_small_multiples",fig_small_multiples),
    ]

    with PdfPages(OUT_PDF) as pdf:
        for name, fn in figures:
            print(f"  Generating {name} ...", end=" ", flush=True)
            fig = fn()
            # save PNG
            fig.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
            # add to PDF
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("done")

    print(f"\nPDF  -> {OUT_PDF}")
    print(f"PNGs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
