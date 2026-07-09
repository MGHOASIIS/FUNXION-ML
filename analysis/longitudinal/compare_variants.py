"""
compare_variants.py
===================
Compare task variants (1-2-3-4) across sessions for head reach and back reach.

For each hand × reach-type group (e.g. Right-Hand Head Reach 1-4), produces:
  outputs/comparisons/<group_slug>/
      variant_comparison.csv   — features as rows, columns: v1_s1 v1_s2 … v4_s4
      variant_overview.pdf     — 4-page PDF:
          Fig 1: Heatmap — top features across all variants × sessions
          Fig 2: Line grid — key features, one subplot per feature,
                 4 lines (one per variant), x-axis = sessions
          Fig 3: Difficulty progression — does difficulty 1→4 show increasing demand?
          Fig 4: Per-session radar — how each variant compares within a session

Groups processed by default:
  Right-Hand Head Reach 1-4
  Left-Hand Head Reach  1-4
  Right-Hand Back Reach 1-4
  Left-Hand Back Reach  1-4

Usage
-----
    python analysis/longitudinal/compare_variants.py
    python analysis/longitudinal/compare_variants.py --groups "Right-Hand Head Reach"
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

HERE    = Path(__file__).resolve().parent
IN_CSV  = HERE / "nm_features.csv"
OUT_DIR = HERE / "outputs" / "comparisons"

SESSIONS  = [1, 2, 3, 4]
SCOLS     = [f"session_{s}" for s in SESSIONS]
SESSION_LABELS = [f"S{s}" for s in SESSIONS]
VARIANTS  = [1, 2, 3, 4]

VARIANT_PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
SESSION_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

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

GROUPS = {
    "Right-Hand Head Reach": [f"Right-Hand Head Reach {v} Exercise" for v in VARIANTS],
    "Left-Hand Head Reach":  [f"Left-Hand Head Reach {v} Exercise"  for v in VARIANTS],
    "Right-Hand Back Reach": [f"Right-Hand Back Reach {v} Exercise" for v in VARIANTS],
    "Left-Hand Back Reach":  [f"Left-Hand Back Reach {v} Exercise"  for v in VARIANTS],
}

# Key features to highlight in the line-grid figure
KEY_FEATURES = [
    "RightHand_PosY_iqr",
    "LeftHand_PosY_iqr",
    "RightHand_PosY_min",
    "LeftHand_PosY_min",
    "hand_lr_correlation_axis1",
    "RightHand_entropy_spectral_entropy",
    "LeftHand_entropy_spectral_entropy",
    "Head_PosY_range",
    "Back_RotX_mean",
    "RightHand_vel_jerk_rms",
    "LeftHand_vel_jerk_rms",
    "RightHand_RotY_rom",
    "LeftHand_RotY_rom",
    "Head_RotY_mean",
    "Back_RotZ_percentile_25",
    "RightHip_PosY_iqr",
]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def load_group(task_names):
    """
    Load nm_features.csv rows for all task_names.
    Returns dict: task_name -> DataFrame (4 sessions, feat cols only)
    """
    df   = pd.read_csv(IN_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    result = {}
    for t in task_names:
        rows = df[df["task_name"] == t].sort_values("session_id").reset_index(drop=True)
        if len(rows) == 0:
            continue
        # keep only numeric feature cols
        fdata = rows[feat_cols].select_dtypes(include=np.number)
        fdata.index = rows["session_id"].values
        result[t] = fdata
    return result


def build_wide_table(group_data, task_names):
    """
    Build wide DataFrame: rows = features,
    cols = v{variant}_s{session} for all variant × session combos.
    """
    # collect all features present in at least one variant
    all_feats = set()
    for t in task_names:
        if t in group_data:
            all_feats.update(group_data[t].columns)
    all_feats = sorted(all_feats)

    records = {}
    for feat in all_feats:
        row = {}
        for vi, t in enumerate(task_names, start=1):
            if t not in group_data:
                for s in SESSIONS:
                    row[f"v{vi}_s{s}"] = np.nan
                continue
            fdata = group_data[t]
            for s in SESSIONS:
                row[f"v{vi}_s{s}"] = float(fdata.loc[s, feat]) if s in fdata.index and feat in fdata.columns else np.nan
        records[feat] = row

    return pd.DataFrame(records).T   # features as rows


# ── figures ──────────────────────────────────────────────────────────────────

def fig_heatmap(wide, group_name, task_names):
    """
    Heatmap: rows = top 30 features (most variance across all v×s),
    cols = v1_s1 … v4_s4, colour = row-normalised value.
    """
    cols = [f"v{v}_s{s}" for v in VARIANTS for s in SESSIONS]
    cols = [c for c in cols if c in wide.columns]

    # rank features by variance across all cells
    sub = wide[cols].astype(float)
    sub = sub.loc[sub.notna().all(axis=1)]
    top = sub.loc[sub.std(axis=1).nlargest(30).index]

    vals = top.values
    row_min = vals.min(axis=1, keepdims=True)
    row_max = vals.max(axis=1, keepdims=True)
    norm = (vals - row_min) / (row_max - row_min + 1e-12)

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    xlabels = [f"V{v}\nS{s}" for v in VARIANTS for s in SESSIONS]
    ax.set_xticklabels(xlabels, fontsize=8)

    # vertical dividers between variants
    for v in range(1, len(VARIANTS)):
        ax.axvline(v * len(SESSIONS) - 0.5, color="white", linewidth=2)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=7.5)
    ax.set_title(f"{group_name} — top 30 features across all variants × sessions\n"
                 "(green = higher, red = lower; each row normalised independently)",
                 fontsize=11, fontweight="bold")

    # variant labels at top
    for vi, vname in enumerate(task_names, start=1):
        ax.text((vi - 1) * len(SESSIONS) + (len(SESSIONS) - 1) / 2,
                -1.5, f"Variant {vi}", ha="center", fontsize=8,
                fontweight="bold", color=VARIANT_PALETTE[vi - 1])

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    fig.tight_layout()
    return fig


def fig_line_grid(group_data, task_names, group_name):
    """
    Grid of subplots — one per key feature.
    Each subplot: x = sessions 1-4, one line per variant.
    """
    avail = [f for f in KEY_FEATURES
             if any(f in group_data[t].columns for t in task_names if t in group_data)]
    if not avail:
        return None

    ncols = 4
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = axes.flatten()

    x = np.arange(1, 5)
    for i, feat in enumerate(avail):
        ax = axes[i]
        for vi, t in enumerate(task_names, start=1):
            if t not in group_data or feat not in group_data[t].columns:
                continue
            vals = [float(group_data[t].loc[s, feat])
                    if s in group_data[t].index else np.nan for s in SESSIONS]
            ax.plot(x, vals, "o-", color=VARIANT_PALETTE[vi - 1],
                    linewidth=1.8, markersize=5, label=f"V{vi}")

        ax.set_title(feat, fontsize=7.5, fontweight="bold", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(SESSION_LABELS, fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(fontsize=7, framealpha=0.7)

    # hide unused axes
    for j in range(len(avail), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{group_name} — key features across variants (V1-V4) and sessions",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_difficulty_progression(group_data, task_names, group_name):
    """
    For each session: bar chart of feature values across variants 1-4.
    Shows whether V1→V4 systematically increases/decreases (difficulty gradient).
    """
    # pick features with biggest spread across variants (within session 1)
    ref_task = task_names[0]
    if ref_task not in group_data:
        return None

    all_feats = set()
    for t in task_names:
        if t in group_data:
            all_feats.update(group_data[t].columns)
    all_feats = list(all_feats)

    # compute variance across variants for session 1
    var_scores = {}
    for feat in all_feats:
        vals = []
        for t in task_names:
            if t in group_data and feat in group_data[t].columns and 1 in group_data[t].index:
                v = float(group_data[t].loc[1, feat])
                if pd.notna(v):
                    vals.append(v)
        if len(vals) == len(task_names):
            var_scores[feat] = np.var(vals)

    top_feats = sorted(var_scores, key=var_scores.get, reverse=True)[:8]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    x = np.arange(1, 5)
    width = 0.18

    for fi, feat in enumerate(top_feats):
        ax = axes[fi]
        for si, (s, scolor) in enumerate(zip(SESSIONS, SESSION_PALETTE)):
            vals = []
            for t in task_names:
                if t in group_data and feat in group_data[t].columns and s in group_data[t].index:
                    vals.append(float(group_data[t].loc[s, feat]))
                else:
                    vals.append(np.nan)
            offsets = (np.arange(len(SESSIONS)) - 1.5) * width
            ax.bar(x + offsets[si], vals, width=width,
                   color=scolor, alpha=0.8, label=f"S{s}")

        ax.set_xticks(x)
        ax.set_xticklabels([f"V{v}" for v in VARIANTS], fontsize=8)
        ax.set_title(feat, fontsize=7.5, fontweight="bold", pad=3)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if fi == 0:
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"{group_name} — difficulty gradient (V1→V4) per session",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_per_session_radar(group_data, task_names, group_name):
    """
    4 radar charts (one per session) — each shows variant 1-4 as overlaid polygons.
    Dimensions = 5 summary features.
    """
    dim_feats = {
        "Vertical\nROM":    ("RightHand_PosY_iqr", "LeftHand_PosY_iqr"),
        "Smoothness":       ("RightHand_vel_jerk_rms", "LeftHand_vel_jerk_rms"),
        "Complexity":       ("RightHand_entropy_spectral_entropy",
                             "LeftHand_entropy_spectral_entropy"),
        "Back\nRotation":   ("Back_RotX_mean",),
        "Bilateral\nCorr":  ("hand_lr_correlation_axis1",),
    }

    # resolve one feature per dimension
    resolved = {}
    all_task_cols = set()
    for t in task_names:
        if t in group_data:
            all_task_cols.update(group_data[t].columns)

    for dim, options in dim_feats.items():
        for feat in options:
            if feat in all_task_cols:
                resolved[dim] = feat
                break

    if len(resolved) < 3:
        return None

    dim_names = list(resolved.keys())
    feat_keys = list(resolved.values())
    N = len(dim_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    # global min/max per feature for normalisation
    global_stats = {}
    for feat in feat_keys:
        all_vals = []
        for t in task_names:
            if t in group_data and feat in group_data[t].columns:
                all_vals.extend(group_data[t][feat].dropna().tolist())
        global_stats[feat] = (min(all_vals) if all_vals else 0,
                               max(all_vals) if all_vals else 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                             subplot_kw={"projection": "polar"})
    axes = axes.flatten()

    for si, s in enumerate(SESSIONS):
        ax = axes[si]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_names, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels([], fontsize=6)
        ax.set_title(f"Session {s}", fontsize=10, fontweight="bold", pad=12)

        for vi, (t, color) in enumerate(zip(task_names, VARIANT_PALETTE), start=1):
            if t not in group_data:
                continue
            scores = []
            for feat in feat_keys:
                lo, hi = global_stats[feat]
                if feat in group_data[t].columns and s in group_data[t].index:
                    val = float(group_data[t].loc[s, feat])
                    norm = (val - lo) / (hi - lo + 1e-12) * 100 if pd.notna(val) else 50
                else:
                    norm = 50
                scores.append(norm)
            vals = scores + [scores[0]]
            ax.plot(angles, vals, "o-", color=color, linewidth=1.8,
                    markersize=4, label=f"V{vi}")
            ax.fill(angles, vals, alpha=0.07, color=color)

        if si == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8)

    fig.suptitle(f"{group_name} — per-session radar (V1–V4 overlaid)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def run_group(group_name, task_names):
    s = slug(group_name)
    out  = OUT_DIR / s
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  {group_name}")
    print(f"{'='*65}")

    # check which tasks exist in the CSV
    available = pd.read_csv(IN_CSV)["task_name"].unique().tolist()
    task_names = [t for t in task_names if t in available]
    if not task_names:
        print("  No matching tasks found — skipping.")
        return

    group_data = load_group(task_names)
    print(f"  Loaded variants: {[t for t in task_names if t in group_data]}")

    # summary: durations + n_reps
    df = pd.read_csv(IN_CSV)
    for t in task_names:
        rows = df[df["task_name"] == t].sort_values("session_id")
        durs = rows["task_duration_s"].tolist()
        reps = rows["n_reps"].tolist() if "n_reps" in rows else ["—"] * 4
        print(f"  {t:<45}  dur={[round(d,1) for d in durs]}  reps={reps}")

    # build wide table and save CSV
    wide = build_wide_table(group_data, task_names)
    wide.to_csv(out / "variant_comparison.csv")
    print(f"  Saved variant_comparison.csv  ({len(wide)} features)")

    figs = [
        ("fig1_heatmap",           lambda: fig_heatmap(wide, group_name, task_names)),
        ("fig2_line_grid",         lambda: fig_line_grid(group_data, task_names, group_name)),
        ("fig3_difficulty",        lambda: fig_difficulty_progression(group_data, task_names, group_name)),
        ("fig4_session_radar",     lambda: fig_per_session_radar(group_data, task_names, group_name)),
    ]

    pdf_path = out / "variant_overview.pdf"
    with PdfPages(pdf_path) as pdf:
        for name, fn in figs:
            print(f"  {name} ...", end=" ", flush=True)
            try:
                fig = fn()
                if fig is None:
                    print("skipped")
                    continue
                fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print("done")
            except Exception as e:
                print(f"error: {e}")

    print(f"  -> {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*",
                        help="Group prefixes to process (e.g. 'Right-Hand Head Reach')")
    args = parser.parse_args()

    groups = args.groups if args.groups else list(GROUPS.keys())

    for g in groups:
        if g not in GROUPS:
            # try partial match
            matches = [k for k in GROUPS if g.lower() in k.lower()]
            if not matches:
                print(f"[Warning] Group '{g}' not found.")
                continue
            g = matches[0]
        run_group(g, GROUPS[g])

    print("\nAll comparisons done.")
    print(f"Outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
