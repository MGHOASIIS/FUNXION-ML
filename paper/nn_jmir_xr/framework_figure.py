"""
generate_framework_diagram.py
------------------------------
Generates the XDash three-layer IMU-based MSK assessment framework diagram
as a standalone SVG file.

Usage:
    python generate_framework_diagram.py
    python generate_framework_diagram.py --out my_diagram.svg

Output:
    xdash_framework.svg  (or path specified via --out)

Requirements:
    Python 3.6+  (no external dependencies)
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results/publication/xdash_framework.svg")
args = parser.parse_args()

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE_FILL      = "#E6F1FB"   # c-blue 50
BLUE_STROKE    = "#185FA5"   # c-blue 600
BLUE_TEXT      = "#0C447C"   # c-blue 800

TEAL_FILL      = "#E1F5EE"   # c-teal 50
TEAL_STROKE    = "#0F6E56"   # c-teal 600
TEAL_TEXT      = "#085041"   # c-teal 800
TEAL_TEXT_DARK = "#04342C"   # c-teal 900

CORAL_FILL     = "#FAECE7"   # c-coral 50
CORAL_STROKE   = "#993C1D"   # c-coral 600
CORAL_TEXT     = "#712B13"   # c-coral 800

GRAY_FILL      = "#F1EFE8"   # c-gray 50
GRAY_STROKE    = "#5F5E5A"   # c-gray 600
GRAY_TEXT      = "#444441"   # c-gray 800

RED_STROKE     = "#A32D2D"   # c-red 600

ARROW_COLOR    = "#333333"

W = 680   # viewBox width — do not change

# ── SVG helpers ───────────────────────────────────────────────────────────────
def rect(x, y, w, h, fill, stroke, stroke_w=0.5, rx=8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"{d}/>')

def text(x, y, content, fill, size=12, weight=400, anchor="middle", baseline="central"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'dominant-baseline="{baseline}" '
            f'font-family="Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{content}</text>')

def th(x, y, content, fill, anchor="middle"):
    """Bold 13px heading inside a box."""
    return text(x, y, content, fill, size=13, weight=600, anchor=anchor)

def ts(x, y, content, fill, anchor="middle"):
    """Regular 11px subtitle inside a box."""
    return text(x, y, content, fill, size=11, weight=400, anchor=anchor)

def line(x1, y1, x2, y2, stroke, sw=1.0, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="{marker}"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}" fill="none"{d}{m}/>')

def path(d, stroke, sw=1.0, dash=None, marker=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    m  = f' marker-end="{marker}"' if marker else ""
    return (f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" '
            f'fill="none"{da}{m}/>')

def pill(x, y, w, h, fill, stroke, label, label_fill, sw=0.5):
    rx = h // 2
    return "\n".join([
        rect(x, y, w, h, fill, stroke, sw, rx=rx),
        th(x + w//2, y + h//2, label, label_fill),
    ])

def section_border(x, y, w, h, stroke, label, label_fill):
    return "\n".join([
        rect(x, y, w, h, "none", stroke, stroke_w=1, rx=12, dash="6 3"),
        text(x+16, y+18, label, label_fill, size=12, weight=600, anchor="start"),
    ])

def blue_box(x, y, w, h, title, subtitle=None):
    lines = [
        rect(x, y, w, h, BLUE_FILL, BLUE_STROKE, 0.5),
        th(x+w//2, y + (h//3 if subtitle else h//2), title, BLUE_TEXT),
    ]
    if subtitle:
        lines.append(ts(x+w//2, y + (h*2//3), subtitle, BLUE_STROKE))
    return "\n".join(lines)

def teal_box(x, y, w, h, title, subtitles=None):
    lines = [rect(x, y, w, h, TEAL_FILL, TEAL_STROKE, 0.5)]
    if subtitles:
        step = (h - 20) / (len(subtitles) + 1)
        lines.append(th(x+w//2, y + step, title, TEAL_TEXT))
        for i, s in enumerate(subtitles, 2):
            lines.append(ts(x+w//2, y + step*i, s, TEAL_STROKE))
    else:
        lines.append(th(x+w//2, y+h//2, title, TEAL_TEXT))
    return "\n".join(lines)

def coral_box(x, y, w, h, title, subtitle=None):
    lines = [
        rect(x, y, w, h, CORAL_FILL, CORAL_STROKE, 0.5),
        th(x+w//2, y + (h//3 if subtitle else h//2), title, CORAL_TEXT),
    ]
    if subtitle:
        lines.append(ts(x+w//2, y + h*2//3, subtitle, CORAL_STROKE))
    return "\n".join(lines)

def gray_box(x, y, w, h, title, subtitle=None):
    lines = [
        rect(x, y, w, h, GRAY_FILL, GRAY_STROKE, 0.5, rx=5),
        th(x+w//2, y + (h//3 if subtitle else h//2), title, GRAY_TEXT),
    ]
    if subtitle:
        lines.append(ts(x+w//2, y + h*2//3, subtitle, GRAY_STROKE))
    return "\n".join(lines)

ARROW = "url(#arrow)"
DASH_THIN = "3 2"
DASH_MED  = "5 3"
DASH_THICK = "6 3"

# ── Layout constants ──────────────────────────────────────────────────────────
# Box columns
BW3 = 178   # width of each of the 3 front-end boxes
BX1, BX2, BX3 = 44, 251, 458

# Back-end / output two-column
BW2L, BW2R = 258, 258
BX2L, BX2R = 44, 378

# Y positions (cumulative)
L1_Y      = 24
L1_H      = 288
DEF1_Y    = 72
DEF1_H    = 44
PAPER1_Y  = 144
PAPER1_H  = 96
PILL1_Y   = 280
PILL1_H   = 22
GAP_ARROW = 34

L2_Y      = L1_Y + L1_H + 26   # 338
L2_H      = 252
DEF2_Y    = L2_Y + 46
DEF2_H    = 30
PAPER2_Y  = L2_Y + 102
PAPER2_HL = 44   # left (model)
PAPER2_HR = 96   # right (paradigms)
PILL2_Y   = L2_Y + 224
PILL2_H   = 22

L3_Y      = L2_Y + L2_H + 28   # 618
L3_H      = 254
DEF3_Y    = L3_Y + 44
DEF3_H    = 30
PAPER3_Y  = L3_Y + 98
PAPER3_H  = 118

RES_Y     = L3_Y + L3_H + 20
RES_H     = 66
TOTAL_H   = RES_Y + RES_H + 10

# ── Build SVG ─────────────────────────────────────────────────────────────────
parts = []

parts.append(f'''<svg width="100%" viewBox="0 0 {W} {TOTAL_H}"
  xmlns="http://www.w3.org/2000/svg"
  role="img">
<title>XDash three-layer IMU-based MSK assessment framework</title>
<desc>Three-layer framework showing front end, backend, and output layer
with framework definition (blue) and paper instantiation (teal) rows.</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>''')

# ════════════════════════════════════════════════════════════════════════════
# LAYER 1: FRONT END
# ════════════════════════════════════════════════════════════════════════════
parts.append(f"<!-- LAYER 1 -->")
parts.append(section_border(28, L1_Y, 624, L1_H, BLUE_STROKE,
                             "Layer 1 · Front end", BLUE_STROKE))

# Framework definition row label
parts.append(ts(W//2, L1_Y+40, "Framework definition", "#888880"))

# Blue definition boxes
parts.append(blue_box(BX1, DEF1_Y, BW3, DEF1_H,
                       "XR tasks", "Any validated instrument"))
parts.append(blue_box(BX2, DEF1_Y, BW3, DEF1_H,
                       "IMU sensors", "Body-worn or device-embedded"))
parts.append(blue_box(BX3, DEF1_Y, BW3, DEF1_H,
                       "MSK group", "Any patient population"))

# Dashed down-arrows from blue to teal
for cx in [BX1+BW3//2, BX2+BW3//2, BX3+BW3//2]:
    parts.append(line(cx, DEF1_Y+DEF1_H, cx, PAPER1_Y-4,
                      BLUE_STROKE, 0.8, DASH_THIN, ARROW))

# "This paper" label
parts.append(ts(W//2, PAPER1_Y-6, "This paper", TEAL_STROKE, "middle"))

# Teal paper boxes
parts.append(teal_box(BX1, PAPER1_Y, BW3, PAPER1_H,
                       "DASH-derived XR tasks",
                       ["Jar opening · Key turning",
                        "Cleaning · Back washing",
                        "Cutting · Hammering"]))
parts.append(teal_box(BX2, PAPER1_Y, BW3, PAPER1_H,
                       "3 × 6-DoF IMU",
                       ["XR headset · left hand",
                        "· right hand @ 50 Hz"]))
parts.append(teal_box(BX3, PAPER1_Y, BW3, PAPER1_H,
                       "Shoulder pathology",
                       ["40 patients (RCT & other)",
                        "+ 20 healthy controls"]))

# Converge lines → pill
CX = [BX1+BW3//2, BX2+BW3//2, BX3+BW3//2]
MID_Y = PILL1_Y - 8
for cx in CX:
    parts.append(line(cx, PAPER1_Y+PAPER1_H, cx, MID_Y,
                      TEAL_STROKE, 0.8))
parts.append(line(CX[0], MID_Y, CX[2], MID_Y, TEAL_STROKE, 0.8))
parts.append(line(CX[1], MID_Y, CX[1], PILL1_Y, TEAL_STROKE, 0.8,
                  marker=ARROW))

parts.append(pill(228, PILL1_Y, 224, PILL1_H,
                  BLUE_FILL, BLUE_STROKE, "Data collection", BLUE_TEXT))

# Inter-layer arrow 1
IL1_Y1 = PILL1_Y + PILL1_H
IL1_Y2 = L2_Y - 4
parts.append(line(W//2, IL1_Y1, W//2, IL1_Y2, "#333333", 1.5, marker=ARROW))
parts.append(ts(W//2+16, (IL1_Y1+IL1_Y2)//2+4,
                "Raw IMU &amp; task data", "#888880", "start"))

# ════════════════════════════════════════════════════════════════════════════
# LAYER 2: BACKEND
# ════════════════════════════════════════════════════════════════════════════
parts.append(f"<!-- LAYER 2 -->")
parts.append(section_border(28, L2_Y, 624, L2_H, BLUE_STROKE,
                             "Layer 2 · Backend", BLUE_STROKE))

parts.append(ts(W//2, DEF2_Y-8, "Framework definition", "#888880"))

# Blue definition boxes
parts.append(blue_box(BX2L, DEF2_Y, BW2L, DEF2_H, "Any model architecture"))
parts.append(blue_box(BX2R, DEF2_Y, BW2R, DEF2_H, "Any downstream clinical task"))

# Dashed down-arrows
for cx in [BX2L+BW2L//2, BX2R+BW2R//2]:
    parts.append(line(cx, DEF2_Y+DEF2_H, cx, PAPER2_Y-4,
                      BLUE_STROKE, 0.8, DASH_THIN, ARROW))

parts.append(ts(W//2, PAPER2_Y-6, "This paper", TEAL_STROKE))

# Teal paper boxes
parts.append(teal_box(BX2L, PAPER2_Y, BW2L, PAPER2_HL,
                       "CNN · RNN · Transformer",
                       ["Truncated sequences · LOOCV"]))
parts.append(teal_box(BX2R, PAPER2_Y, BW2R, PAPER2_HR,
                       "Binary classification · 4 paradigms",
                       ["Patients vs controls",
                        "RCT vs controls",
                        "Other conditions vs controls",
                        "RCT vs other conditions"]))

# Converge → pill
L2_PILL_Y = PILL2_Y
L_BOT_L = PAPER2_Y + PAPER2_HL
L_BOT_R = PAPER2_Y + PAPER2_HR
MID2_Y = L2_PILL_Y - 8
CXL = BX2L + BW2L//2
CXR = BX2R + BW2R//2
parts.append(line(CXL, L_BOT_L, CXL, MID2_Y, TEAL_STROKE, 0.8))
parts.append(line(CXR, L_BOT_R, CXR, MID2_Y, TEAL_STROKE, 0.8))
parts.append(line(CXL, MID2_Y, CXR, MID2_Y, TEAL_STROKE, 0.8))
parts.append(line(W//2, MID2_Y, W//2, L2_PILL_Y, TEAL_STROKE, 0.8,
                  marker=ARROW))

parts.append(pill(236, L2_PILL_Y, 208, PILL2_H,
                  BLUE_FILL, BLUE_STROKE,
                  "Analysis &amp; model inference", BLUE_TEXT))

# Inter-layer arrow 2
IL2_Y1 = L2_PILL_Y + PILL2_H
IL2_Y2 = L3_Y - 4
parts.append(line(W//2, IL2_Y1, W//2, IL2_Y2, "#333333", 1.5, marker=ARROW))
parts.append(ts(W//2+16, (IL2_Y1+IL2_Y2)//2+4,
                "Model decisions &amp; features", "#888880", "start"))

# ════════════════════════════════════════════════════════════════════════════
# LAYER 3: OUTPUT
# ════════════════════════════════════════════════════════════════════════════
parts.append(f"<!-- LAYER 3 -->")
parts.append(section_border(28, L3_Y, 624, L3_H, RED_STROKE,
                             "Layer 3 · Output", RED_STROKE))

parts.append(ts(W//2, DEF3_Y-8, "Framework definition", "#888880"))

# Coral framework boxes
parts.append(coral_box(BX2L, DEF3_Y, BW2L, DEF3_H, "Evaluation &amp; explainability"))
parts.append(coral_box(BX2R, DEF3_Y, BW2R, DEF3_H, "Clinically interpretable outputs"))

# Dashed down-arrows
for cx in [BX2L+BW2L//2, BX2R+BW2R//2]:
    parts.append(line(cx, DEF3_Y+DEF3_H, cx, PAPER3_Y-4,
                      CORAL_STROKE, 0.8, DASH_THIN, ARROW))

parts.append(ts(W//2, PAPER3_Y-6, "This paper", TEAL_STROKE))

# Teal output boxes - left
EV_SUBTITLES = [
    "BA · AUC · Recall",
    "AUC confidence intervals",
    "ROC curve · Confusion matrix",
    "Probability density curves",
    "Weight-based feature importance",
]
parts.append(teal_box(BX2L, PAPER3_Y, BW2L, PAPER3_H,
                       "Performance &amp; explainability", EV_SUBTITLES))

# Teal output boxes - right (manual for asterisk line)
CL_SUBTITLES = [
    "DASH task-level classification",
    "PROMs complement (DASH)",
    "Top discriminative channels",
]
parts.append(rect(BX2R, PAPER3_Y, BW2R, PAPER3_H,
                  TEAL_FILL, TEAL_STROKE, 0.5))
step = (PAPER3_H - 20) / (len(CL_SUBTITLES) + 2)
CX_R = BX2R + BW2R//2
parts.append(th(CX_R, PAPER3_Y + step, "Clinical outputs", TEAL_TEXT))
for i, s in enumerate(CL_SUBTITLES, 2):
    parts.append(ts(CX_R, PAPER3_Y + step*(i), s, TEAL_STROKE))
# Asterisk line in white
parts.append(
    f'<text x="{CX_R}" y="{PAPER3_Y + step*(len(CL_SUBTITLES)+2)}" '
    f'text-anchor="middle" dominant-baseline="central" '
    f'font-family="Arial, sans-serif" font-size="12" '
    f'font-weight="600" fill="#FFFFFF">'
    f'* Head movement compensation</text>'
)

# Horizontal arrow between output boxes
MID3_Y = PAPER3_Y + PAPER3_H//2
parts.append(line(BX2L+BW2L, MID3_Y, BX2R, MID3_Y,
                  TEAL_STROKE, 1.0, marker=ARROW))

# ════════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════════
LEG_Y = L3_Y + L3_H + 6
parts.append(f"<!-- LEGEND -->")
swatches = [
    (BLUE_FILL,  BLUE_STROKE,  "Framework definition"),
    (TEAL_FILL,  TEAL_STROKE,  "This paper"),
    (CORAL_FILL, CORAL_STROKE, "Output layer"),
]
lx = 44
for fill, stroke, label in swatches:
    parts.append(rect(lx, LEG_Y, 10, 10, fill, stroke, 0.5, rx=2))
    parts.append(text(lx+14, LEG_Y+5, label, "#888880",
                      size=11, weight=400, anchor="start",
                      baseline="central"))
    lx += len(label)*7 + 30

# Paper finding label
parts.append(
    f'<text x="{lx}" y="{LEG_Y+5}" text-anchor="start" '
    f'dominant-baseline="central" font-family="Arial, sans-serif" '
    f'font-size="11" font-weight="600" fill="{TEAL_STROKE}">'
    f'* Paper finding</text>'
)

# ════════════════════════════════════════════════════════════════════════════
# RESEARCH INSTRUMENT PANEL
# ════════════════════════════════════════════════════════════════════════════
parts.append(f"<!-- RESEARCH INSTRUMENT -->")
# Arrow from output layer down
parts.append(path(f"M {W//2} {L3_Y+L3_H} L {W//2} {RES_Y}",
                  GRAY_STROKE, 1.0, DASH_MED, ARROW))

parts.append(rect(28, RES_Y, 624, RES_H, "none", GRAY_STROKE, 0.5,
                  rx=8, dash="4 3"))
parts.append(text(44, RES_Y+16, "Enables standardised comparison across:",
                  "#888880", size=11, weight=600, anchor="start",
                  baseline="central"))

RES_BOXES = [
    (36,  "Task design",       None),
    (186, "Data collection",   "Device · placement · rate"),
    (358, "MSK population",    None),
    (518, "Model architecture",None),
]
RES_WIDTHS = [138, 160, 148, 126]
for (rx2, rtitle, rsub), rw in zip(RES_BOXES, RES_WIDTHS):
    parts.append(gray_box(rx2, RES_Y+28, rw, 32, rtitle, rsub))

# ════════════════════════════════════════════════════════════════════════════
# CLOSE SVG
# ════════════════════════════════════════════════════════════════════════════
parts.append("</svg>")

svg = "\n".join(parts)

# Write opacity helper — Python f-strings don't handle optional attrs cleanly,
# so patch the two converge lines that need explicit opacity
svg = svg.replace("stroke-width: 0.8\" fill=\"none\"", "stroke-width=\"0.8\" fill=\"none\"")

out = Path(args.out)
out.write_text(svg, encoding="utf-8")
print(f"Saved → {out.resolve()}")
print(f"Open in any browser or vector editor (Inkscape, Illustrator, Figma).")