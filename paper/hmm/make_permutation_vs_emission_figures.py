#!/usr/bin/env python3
"""
Fig. 3b - Back Washing (T4/P2) only.

Identical to the previous version except that the legend is placed inside the
axes, in the empty region above the bars, instead of above the figure.
All values exact: permutation from results_T4_P2_HMM_variable_length.json;
emission profiles are column means of the normalized state-importance
matrices, eps_d = (1/K) sum_u delta~(u,d), K = 8.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config.paths import get_paper_dir

GROUPS = ["head_pos", "head_rot",
          "left_hand_pos", "left_hand_rot",
          "right_hand_pos", "right_hand_rot"]

XLABELS = ["Head\npos.", "Head\nrot.",
           "L hand\npos.", "L hand\nrot.",
           "R hand\npos.", "R hand\nrot."]

PERM_T4 = dict(head_pos=0.204, head_rot=0.105,
               left_hand_pos=0.169, left_hand_rot=0.196,
               right_hand_pos=0.169, right_hand_rot=0.156)

EMIS_C_T4 = dict(head_pos=0.125, head_rot=0.148,
                 left_hand_pos=0.180, left_hand_rot=0.077,
                 right_hand_pos=0.192, right_hand_rot=0.276)

EMIS_P_T4 = dict(head_pos=0.191, head_rot=0.098,
                 left_hand_pos=0.193, left_hand_rot=0.175,
                 right_hand_pos=0.253, right_hand_rot=0.088)

SHOW_VALUES = False
YMAX = 0.78        # kept identical to Fig. 3a so the two remain comparable

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "hatch.linewidth": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

SERIES = [
    ("#C43D30", "#6E1B12", "///", "Permutation importance (discriminative)"),
    ("#E0E0E0", "#4A4A4A", None,  "Emission importance, control model (structural)"),
    ("#8A8A8A", "#1A1A1A", "---", "Emission importance, RCT model (structural)"),
]

BAR_W = 0.26
X = np.arange(len(GROUPS))
profiles = [PERM_T4, EMIS_C_T4, EMIS_P_T4]

fig, ax = plt.subplots(figsize=(3.5, 3.05))

for i, (prof, (fc, ec, hatch, _)) in enumerate(zip(profiles, SERIES)):
    vals = [prof[g] for g in GROUPS]
    off = (i - 1) * BAR_W
    ax.bar(X + off, vals, BAR_W, facecolor=fc, edgecolor=ec,
           hatch=hatch, linewidth=0.6, zorder=3)
    if SHOW_VALUES:
        for xi, v in zip(X + off, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=5.2, rotation=90, zorder=4)

ax.set_xticks(X)
ax.set_xticklabels(XLABELS)
ax.set_ylim(0, YMAX)
ax.set_yticks(np.arange(0, 0.71, 0.1))
ax.set_xlim(-0.55, len(GROUPS) - 0.45)
ax.set_ylabel("Normalized importance share")
ax.set_title("(b) Back Washing (T4/P2)", pad=4)
ax.yaxis.grid(True, linewidth=0.4, color="#CCCCCC", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for xb in (1.5, 3.5):
    ax.axvline(xb, color="#BBBBBB", linewidth=0.5, linestyle=(0, (2, 2)),
               zorder=1)

# legend inside the axes, occupying the empty area above the bars
handles = [Patch(facecolor=fc, edgecolor=ec, hatch=h, linewidth=0.6, label=lab)
           for fc, ec, h, lab in SERIES]
leg = ax.legend(handles=handles, loc="upper center",
                bbox_to_anchor=(0.5, 0.55),
                bbox_transform=ax.get_yaxis_transform(),
                ncol=1, frameon=True, facecolor="white", edgecolor="none",
                framealpha=1.0, handlelength=1.8, handleheight=1.0,
                handletextpad=0.5, labelspacing=0.45, borderaxespad=0.0,
                borderpad=0.25)
leg.set_zorder(5)

out_dir = get_paper_dir("hmm")
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png", "eps"):
    fig.savefig(out_dir / f"Fig3b_back_washing.{ext}", dpi=600)
print(f"written: {out_dir}/Fig3b_back_washing.pdf / .png / .eps")