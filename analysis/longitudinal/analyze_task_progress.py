"""
analyze_task_progress.py
========================
Generic longitudinal analysis + visualization for any task in nm_features.csv.
Runs the same pipeline as the Push Down script for any task name.

For each task produces
    outputs/<task_slug>/
        longitudinal.csv          — one row per feature, values + delta/pct/direction
        plots/01_heatmap.png
        plots/02_coordination.png
        plots/03_rom.png
        plots/04_posture.png
        plots/05_complexity.png
        plots/06_signal_shape.png
        plots/07_radar.png
        plots/08_small_multiples.png
        progress.pdf              — all figures in one file

Usage
-----
    # specific tasks
    python analyze_task_progress.py "Sit-to-Stand Exercise" "Floor Pick-Up Exercise" "Twisting Task"

    # all exercise tasks (skips calibration/setup tasks)
    python analyze_task_progress.py --all-exercises
"""

import sys
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── paths ─────────────────────────────────────────────────────────────────────

HERE    = Path(__file__).resolve().parent
IN_CSV  = HERE / "nm_features.csv"
OUT_DIR = HERE / "outputs"

SESSIONS       = [1, 2, 3, 4]
SCOLS          = [f"session_{s}" for s in SESSIONS]
SESSION_LABELS = [f"Session {s}" for s in SESSIONS]
PALETTE        = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

META_COLS = [
    "session_id", "task_name",
    "task_start_s", "task_end_s", "task_duration_s",
    "dominant_side", "injured_side",
    "eye_height", "torso_width", "torso_height",
    "leg_height", "arm_length", "shoulder_height",
    "n_reps", "n_pickups", "n_stands",
    "furthest_rotation_deg", "piston_distance_m", "pain_score",
    "n_frames",
]

EXERCISE_TASKS = [
    "Push Down Exercise",
    "Sit-to-Stand Exercise",
    "Floor Pick-Up Exercise",
    "Twisting Task",
    "Right Lateral Reach 1 Exercise",
    "Left Lateral Reach 1 Exercise",
    "Right-Hand Head Reach 1 Exercise",
    "Left-Hand Head Reach 1 Exercise",
    "Right-Hand Back Reach 1 Exercise",
    "Left-Hand Back Reach 1 Exercise",
]


def slug(task_name):
    return re.sub(r"[^a-z0-9]+", "_", task_name.lower()).strip("_")


# ── comparison CSV ─────────────────────────────────────────────────────────────

def direction_label(values):
    v = [x for x in values if pd.notna(x)]
    if len(v) < 2:
        return "insufficient_data"
    d = np.diff(v)
    if all(x > 0 for x in d):  return "increasing"
    if all(x < 0 for x in d):  return "decreasing"
    if all(x == 0 for x in d): return "stable"
    return "mixed"


def is_monotonic(values):
    v = [x for x in values if pd.notna(x)]
    if len(v) < 2: return False
    d = np.diff(v)
    return bool(all(x >= 0 for x in d) or all(x <= 0 for x in d))


def build_longitudinal(task_name):
    df = pd.read_csv(IN_CSV)
    rows = df[df["task_name"] == task_name].sort_values("session_id").reset_index(drop=True)
    if len(rows) == 0:
        raise ValueError(f"Task not found: {task_name!r}")

    found = rows["session_id"].tolist()
    feat_cols = [c for c in rows.columns if c not in META_COLS]
    feat_data = rows[feat_cols].select_dtypes(include=np.number)
    feat_data = feat_data.loc[:, feat_data.notna().any()]

    records = []
    for feat in feat_data.columns:
        vals = feat_data[feat].tolist()
        session_vals = {f"session_{sid}": v for sid, v in zip(found, vals)}
        v1, v4 = vals[0], vals[-1]
        if pd.notna(v1) and pd.notna(v4):
            delta = v4 - v1
            pct   = (delta / abs(v1) * 100) if v1 != 0 else np.nan
        else:
            delta = pct = np.nan
        records.append({
            "feature":           feat,
            **session_vals,
            "delta_1_to_4":      delta,
            "pct_change_1_to_4": pct,
            "direction":         direction_label(vals),
            "monotonic":         is_monotonic(vals),
        })

    result = (pd.DataFrame(records)
              .sort_values("pct_change_1_to_4",
                           key=lambda s: s.abs(), ascending=False,
                           na_position="last")
              .reset_index(drop=True))
    return result, rows


def top_monotonic(df_long, n=40, pct_cap=500):
    mono = df_long[
        (df_long["monotonic"] == True) &
        (df_long["session_1"].abs() > 0.01) &
        (df_long["pct_change_1_to_4"].abs() < pct_cap) &
        df_long["pct_change_1_to_4"].notna()
    ].copy()
    mono["abs_pct"] = mono["pct_change_1_to_4"].abs()
    return mono.nlargest(n, "abs_pct")


# ── shared plot helpers ───────────────────────────────────────────────────────

def trend_ax(ax, data_rows, labels, title, ylabel=""):
    x = np.arange(1, 5)
    cmap = plt.cm.tab10
    for i, (feat, row) in enumerate(data_rows.iterrows()):
        vals  = row[SCOLS].values.astype(float)
        color = cmap(i / max(len(data_rows) - 1, 1))
        ax.plot(x, vals, "o-", color=color, linewidth=2, markersize=7,
                label=labels[i] if labels else feat)
    ax.set_xticks(x)
    ax.set_xticklabels(SESSION_LABELS, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, framealpha=0.7)


def bar_ax(ax, feat_labels, pcts, colors, title):
    y = np.arange(len(feat_labels))
    bars = ax.barh(y, pcts, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% change  (session 1 → 4)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, pcts):
        if pd.isna(val): continue
        ax.text(val + (1 if val >= 0 else -1),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center",
                ha="left" if val >= 0 else "right", fontsize=7)


def panel(df_long, feat_dict, title, invert_good=None):
    """Two-panel figure: trend lines + % change bars."""
    invert_good = invert_good or set()
    avail = {k: v for k, v in feat_dict.items()
             if v in df_long["feature"].values}
    if not avail:
        return None
    data   = df_long[df_long["feature"].isin(avail.values())].set_index("feature")
    labels = list(avail.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    trend_ax(axes[0], data.loc[list(avail.values())], labels, title)
    pcts   = [float(data.loc[v, "pct_change_1_to_4"]) if v in data.index else np.nan
              for v in avail.values()]
    colors = []
    for v, p in zip(avail.values(), pcts):
        if pd.isna(p):
            colors.append("#aaaaaa")
        elif v in invert_good:
            colors.append("#2ecc71" if p < 0 else "#e74c3c")
        else:
            colors.append("#2ecc71" if p > 0 else "#e74c3c")
    bar_ax(axes[1], labels, pcts, colors, "% change  S1 → S4")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── per-figure generators ─────────────────────────────────────────────────────

def fig_heatmap(df_long, task_name):
    top = top_monotonic(df_long, n=40).set_index("feature")
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
    ax.set_title(f"Top 40 monotonic features — {task_name}\n"
                 "(green = higher value, red = lower)", fontsize=12, fontweight="bold")
    for i, (feat, row) in enumerate(top.iterrows()):
        pct = row["pct_change_1_to_4"]
        ax.text(4.15, i, f"{pct:+.0f}%", va="center", fontsize=6.5,
                color="#2c7bb6" if pct > 0 else "#d7191c")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.12, label="row-normalised value")
    fig.tight_layout()
    return fig


def fig_small_multiples(df_long, task_name):
    top12 = top_monotonic(df_long, n=12)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    x = np.arange(1, 5)
    for i, (_, row) in enumerate(top12.iterrows()):
        ax   = axes[i]
        vals = row[SCOLS].values.astype(float)
        pct  = row["pct_change_1_to_4"]
        col  = "#2ecc71" if pct > 0 else "#e74c3c"
        ax.plot(x, vals, "o-", color=col, linewidth=2.5, markersize=8)
        ax.fill_between(x, vals, alpha=0.12, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(["S1","S2","S3","S4"], fontsize=8)
        ax.set_title(row["feature"], fontsize=7.5, fontweight="bold", pad=4)
        ax.text(0.97, 0.05, f"{pct:+.1f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=col, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Top 12 monotonic features — {task_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_radar(df_long, task_name):
    # 6 generic dimensions from available monotonic features
    candidates = {
        "Bilateral\nCoordination": ("knee_lr_correlation_axis3",
                                    "hand_lr_correlation_axis1",
                                    "hip_lr_correlation_axis1"),
        "Vertical\nROM":           ("LeftHand_PosY_iqr",
                                    "RightHand_PosY_iqr",
                                    "Head_PosY_range"),
        "Knee ROM\nspread":        ("LeftKnee_RotX_iqr",
                                    "RightKnee_RotX_iqr",
                                    "LeftKnee_RotX_std"),
        "Movement\nComplexity":    ("RightHand_entropy_spectral_entropy",
                                    "LeftHand_entropy_spectral_entropy",
                                    "LeftHip_entropy_perm_entropy"),
        "Posture\nStability":      ("Head_PosZ_energy",      # lower = better
                                    "Head_PosZ_sma"),
        "Smoothness":              ("Head_vel_sparc",         # less negative = better
                                    "RightHand_vel_sparc",
                                    "LeftHand_vel_sparc"),
    }
    INVERT = {"Posture\nStability"}  # decreasing = progress

    df_idx = df_long.set_index("feature")
    dim_scores = {}
    for dim, feat_options in candidates.items():
        for feat in feat_options:
            if feat in df_idx.index:
                vals = df_idx.loc[feat, SCOLS].values.astype(float)
                lo, hi = vals.min(), vals.max()
                if hi == lo:
                    norm = np.full(4, 50.0)
                else:
                    norm = (vals - lo) / (hi - lo) * 100
                    if dim in INVERT:
                        norm = 100 - norm
                dim_scores[dim] = norm
                break

    if len(dim_scores) < 3:
        return None

    dim_names = list(dim_scores.keys())
    N      = len(dim_names)
    scores = np.array([dim_scores[d] for d in dim_names])   # (N, 4)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_names, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25","50","75","100"], fontsize=7, color="grey")

    for s_idx, (label, color) in enumerate(zip(SESSION_LABELS, PALETTE)):
        vals  = scores[:, s_idx].tolist() + [scores[:, s_idx][0]]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title(f"Progress radar — {task_name}\n"
                 "(100 = best observed across sessions)",
                 fontsize=11, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# ── task-specific feature panels ─────────────────────────────────────────────

def panels_for_task(df_long, task_name):
    """
    Returns list of (fig_name, fig) tuples for task-specific panels.
    Uses the same 6 thematic groups as the Push Down script, adapted
    to each task's most relevant features.
    """
    figs = []

    # 1. Bilateral coordination
    f = panel(df_long, {
        "Knee RotX corr":   "knee_lr_correlation_axis3",
        "Hand PosY corr":   "hand_lr_correlation_axis1",
        "Hand PosX corr":   "hand_lr_correlation_axis0",
        "Hip PosY corr":    "hip_lr_correlation_axis1",
        "Knee PosY corr":   "knee_lr_correlation_axis1",
    }, f"Bilateral Coordination — {task_name}")
    if f: figs.append(("02_coordination", f))

    # 2. Range of motion / vertical reach
    f = panel(df_long, {
        "LeftHand PosY min":  "LeftHand_PosY_min",
        "RightHand PosY min": "RightHand_PosY_min",
        "LeftHand PosY IQR":  "LeftHand_PosY_iqr",
        "LeftKnee RotX IQR":  "LeftKnee_RotX_iqr",
        "Head PosY range":    "Head_PosY_range",
        "Head PosY IQR":      "Head_PosY_iqr",
    }, f"Range of Motion — {task_name}")
    if f: figs.append(("03_rom", f))

    # 3. Posture
    f = panel(df_long, {
        "Head RotY min":    "Head_RotY_min",
        "Head RotY mean":   "Head_RotY_mean",
        "Back RotX mean":   "Back_RotX_mean",
        "Back RotZ pct25":  "Back_RotZ_percentile_25",
        "Head PosZ energy": "Head_PosZ_energy",
        "Head PosZ sma":    "Head_PosZ_sma",
    }, f"Posture — {task_name}",
       invert_good={"Head_PosZ_energy", "Head_PosZ_sma",
                    "Back_RotX_mean", "Back_RotZ_percentile_25"})
    if f: figs.append(("04_posture", f))

    # 4. Movement complexity / entropy
    f = panel(df_long, {
        "RightHand spectral ent":  "RightHand_entropy_spectral_entropy",
        "LeftHand spectral ent":   "LeftHand_entropy_spectral_entropy",
        "LeftHip perm entropy":    "LeftHip_entropy_perm_entropy",
        "RightHip perm entropy":   "RightHip_entropy_perm_entropy",
        "RightHand freq kurt":     "RightHand_freq_spectral_kurtosis",
        "LeftHand freq skew":      "LeftHand_freq_spectral_skewness",
    }, f"Movement Complexity — {task_name}",
       invert_good={"RightHand_freq_spectral_kurtosis",
                    "LeftHand_freq_spectral_skewness"})
    if f: figs.append(("05_complexity", f))

    # 5. Smoothness / velocity
    f = panel(df_long, {
        "RightHip jerk RMS":   "RightHip_vel_jerk_rms",
        "LeftHip jerk RMS":    "LeftHip_vel_jerk_rms",
        "RightHand jerk RMS":  "RightHand_vel_jerk_rms",
        "LeftHand jerk RMS":   "LeftHand_vel_jerk_rms",
        "Head SPARC":          "Head_vel_sparc",
        "RightHand SPARC":     "RightHand_vel_sparc",
    }, f"Smoothness / Velocity — {task_name}",
       invert_good={"RightHip_vel_jerk_rms", "LeftHip_vel_jerk_rms",
                    "RightHand_vel_jerk_rms", "LeftHand_vel_jerk_rms"})
    if f: figs.append(("06_smoothness", f))

    # 6. Signal shape normalisation
    f = panel(df_long, {
        "Back RotZ kurtosis":     "Back_RotZ_kurtosis",
        "Back RotY skewness":     "Back_RotY_skewness",
        "LeftHand PosY kurtosis": "LeftHand_PosY_kurtosis",
        "RightKnee PosZ kurt":    "RightKnee_PosZ_kurtosis",
        "Head PosY kurtosis":     "Head_PosY_kurtosis",
    }, f"Signal Shape — {task_name}")
    if f: figs.append(("07_signal_shape", f))

    return figs


# ── task-level summary print ──────────────────────────────────────────────────

def print_summary(df_long, task_name, rows_meta):
    mono = df_long[
        (df_long["monotonic"] == True) &
        (df_long["session_1"].abs() > 0.01) &
        (df_long["pct_change_1_to_4"].abs() < 500) &
        df_long["pct_change_1_to_4"].notna()
    ]

    print(f"\n{'='*70}")
    print(f"  {task_name}")
    print(f"{'='*70}")
    print(f"  Sessions: {rows_meta['session_id'].tolist()}")
    print(f"  Durations (s): {[round(x,1) for x in rows_meta['task_duration_s'].tolist()]}")
    n_reps = rows_meta.get("n_reps", pd.Series([None]*4))
    if n_reps.notna().any():
        print(f"  Reps: {n_reps.tolist()}")
    print(f"  Monotonic features: {len(mono)}")
    print(f"  Direction breakdown:\n{mono['direction'].value_counts().to_string()}\n")

    inc = mono[mono["direction"]=="increasing"].nlargest(5,"pct_change_1_to_4")
    dec = mono[mono["direction"]=="decreasing"].nsmallest(5,"pct_change_1_to_4")

    print("  Top 5 increasing:")
    for _, r in inc.iterrows():
        print(f"    {r['feature']:<55} {r['pct_change_1_to_4']:>+8.1f}%")
    print("  Top 5 decreasing:")
    for _, r in dec.iterrows():
        print(f"    {r['feature']:<55} {r['pct_change_1_to_4']:>+8.1f}%")


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_task(task_name):
    s = slug(task_name)
    task_dir  = OUT_DIR / s
    plot_dir  = task_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- build longitudinal comparison ---
    df_long, rows_meta = build_longitudinal(task_name)
    csv_path = task_dir / "longitudinal.csv"
    df_long.to_csv(csv_path, index=False)

    print_summary(df_long, task_name, rows_meta)

    # --- generate figures ---
    all_figs = [
        ("01_heatmap",       lambda: fig_heatmap(df_long, task_name)),
        *panels_for_task(df_long, task_name),
        ("08_radar",         lambda: fig_radar(df_long, task_name)),
        ("09_small_multiples", lambda: fig_small_multiples(df_long, task_name)),
    ]

    pdf_path = task_dir / "progress.pdf"
    with PdfPages(pdf_path) as pdf:
        for name, fn in all_figs:
            print(f"  {name} ...", end=" ", flush=True)
            try:
                fig = fn() if callable(fn) else fn
                if fig is None:
                    print("skipped (no data)")
                    continue
                fig.savefig(plot_dir / f"{name}.png", dpi=150, bbox_inches="tight")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print("done")
            except Exception as e:
                print(f"error: {e}")

    print(f"\n  -> {csv_path}")
    print(f"  -> {pdf_path}")
    print(f"  -> {plot_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*",
                        help="Task names to analyse (quote multi-word names)")
    parser.add_argument("--all-exercises", action="store_true",
                        help="Run all standard exercise tasks")
    args = parser.parse_args()

    if args.all_exercises:
        tasks = EXERCISE_TASKS
    elif args.tasks:
        tasks = args.tasks
    else:
        # default: the three requested tasks
        tasks = [
            "Sit-to-Stand Exercise",
            "Floor Pick-Up Exercise",
            "Twisting Task",
        ]

    # validate against CSV
    available = pd.read_csv(IN_CSV)["task_name"].unique().tolist()
    for t in tasks:
        if t not in available:
            print(f"[Warning] '{t}' not found in nm_features.csv — skipping.")

    tasks = [t for t in tasks if t in available]
    if not tasks:
        print("No valid tasks to process.")
        sys.exit(1)

    print(f"Processing {len(tasks)} task(s): {tasks}\n")
    for t in tasks:
        run_task(t)

    print("\nAll done.")


if __name__ == "__main__":
    main()
