"""Generate the two repository schematics under docs/:

  1. app_architecture.png   -- the real-time deployment / streaming pipeline.
  2. training_pipeline.png   -- how the three encoders are trained & selected.

Self-contained (matplotlib only; no dataset or checkpoint dependency):

    python docs/make_diagrams.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))

# --- modern, muted palette -------------------------------------------------
INK      = "#0f172a"     # titles
SUBINK   = "#475569"     # sub-labels
EDGE     = "#94a3b8"     # default border
LINE     = "#64748b"     # connectors
NEUTRAL  = "#e2e8f0"
BLUE     = "#dbeafe"
GREEN    = "#dcfce7"
ORANGE   = "#ffedd5"
PURPLE   = "#ede9fe"
ACC_FC   = "#fee2e2"     # highlighted (deployed) card fill
ACC_EC   = "#ef4444"     # highlighted border

SHADOW = [pe.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace="#334155", alpha=0.18)]


def card(ax, x, y, w, h, title, sub=None, fc=NEUTRAL, accent=False,
         title_fs=10.5, sub_fs=8.0):
    r = min(w, h) * 0.22
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={r}",
                         fc=fc, ec=(ACC_EC if accent else EDGE),
                         lw=(2.4 if accent else 1.3))
    box.set_path_effects(SHADOW)
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + h * 0.15, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=INK)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=sub_fs, color=SUBINK)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=INK)


def arrow(ax, p0, p1, color=LINE, lw=1.7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0,
                                 joinstyle="round", capstyle="round"))


def elbow(ax, pts, color=LINE, lw=1.7):
    """Orthogonal multi-segment connector; arrowhead on the final segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                           solid_capstyle="round", solid_joinstyle="round"))
    arrow(ax, pts[-2], pts[-1], color, lw)


def _new(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


# ============================ FIG 1: app architecture ======================
def app_architecture():
    fig, ax = _new((14.0, 3.6), (0, 137), (0, 33))
    steps = [
        ("Video",        "file / camera",      NEUTRAL, False),
        ("ROI",          "YOLO·manual·full",   BLUE,    False),
        ("Optical flow", "8×8 grid → [2,64]",  GREEN,   False),
        ("Normalize",    "80-frame · IQR",     GREEN,   False),
        ("MIL frontend", "conv + K-slot attn", ORANGE,  False),
        ("GRU",          "temporal encoder",   ACC_FC,  True),
        ("Decode",       "peak · thr 0.60",    PURPLE,  False),
    ]
    x0, W, step, y, h = 1.0, 17.0, 19.5, 11.5, 15.0
    for i, (t, s, fc, acc) in enumerate(steps):
        card(ax, x0 + i * step, y, W, h, t, s, fc=fc, accent=acc)
    for i in range(len(steps) - 1):
        arrow(ax, (x0 + i * step + W, y + h / 2), (x0 + (i + 1) * step, y + h / 2))

    # outputs bar (full width) fed by the decode stage
    band_x, band_w = 1.0, 134.0
    card(ax, band_x, 1.0, band_w, 6.2,
         "Outputs      live BPM      ·      breath count      ·      breath-LED      "
         "·      apnea watchdog      ·      video overlay",
         fc="#eef2ff", title_fs=9.5)
    dec_cx = x0 + 6 * step + W / 2
    arrow(ax, (dec_cx, y), (dec_cx, 7.2))

    ax.text(68.5, 31.6, "Real-time deployment pipeline",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(68.5, 28.7, "deploy/pipeline.py  +  deploy/app.py   ·   CPU ≈ 3× real-time @ 10 fps",
            ha="center", va="center", fontsize=8.4, style="italic", color=SUBINK)
    _save(fig, "app_architecture.png")


# ============================ FIG 2: training pipeline =====================
def training_pipeline():
    fig, ax = _new((14.0, 5.6), (0, 140), (0, 100))

    # spine: data prep
    sy, sh = 60, 16
    card(ax, 2,  sy, 24, sh, "Dataset",     "[2,64,T] + labels", fc=NEUTRAL)
    card(ax, 32, sy, 24, sh, "Subject split", "16 infants → 6 grp", fc=BLUE)
    card(ax, 62, sy, 24, sh, "Nested CV",   "6 folds · Optuna",  fc=GREEN)
    arrow(ax, (26, sy + sh / 2), (32, sy + sh / 2))
    arrow(ax, (56, sy + sh / 2), (62, sy + sh / 2))

    # three encoders (shared MIL frontend)
    ex, ew, eh = 95, 25, 13
    enc = [("TCN", "~45K params", ORANGE, False, 84),
           ("GRU  ✓ deployed", "~45K params", ACC_FC, True, 61.5),
           ("Transformer", "~46K params", ORANGE, False, 39)]
    nx = 86, sy + sh / 2                                   # nested CV right edge
    for t, s, fc, acc, ey in enc:
        card(ax, ex, ey, ew, eh, t, s, fc=fc, accent=acc, title_fs=10.0)
        arrow(ax, nx, (ex, ey + eh / 2))
    ax.text(ex + ew / 2, 99, "shared MIL frontend — only the encoder changes",
            ha="center", va="center", fontsize=8.0, style="italic", color=SUBINK)

    # metrics (fan-in)
    mx, mw, my, mh = 124, 14, 60, 16
    card(ax, mx, my, mw, mh, "Metrics", "F1@N\nBPM-MAE", fc=PURPLE, title_fs=10.0, sub_fs=7.6)
    for _, _, _, _, ey in enc:
        arrow(ax, (ex + ew, ey + eh / 2), (mx, my + mh / 2))

    # selection & freeze (bottom tier, left -> right)
    by, bh = 10, 13
    card(ax, 32, by, 30, bh, "GRU selected", "best F1@4 & BPM-MAE", fc=ACC_FC, accent=True)
    card(ax, 72, by, 44, bh, "Retrain on ALL clips", "→ deploy/breath_gru.pt", fc=GREEN)
    arrow(ax, (62, by + bh / 2), (72, by + bh / 2))

    # clean orthogonal elbow: Metrics -> (down) -> (left) -> GRU selected
    elbow(ax, [(mx + mw / 2, my), (mx + mw / 2, 31), (47, 31), (47, by + bh)])

    ax.text(70, 99, "", ha="center")   # spacer keeps top margin tidy
    ax.text(44, 92, "Training & model-selection pipeline",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(44, 87.5, "trainer/nested.py  +  deploy/train_deploy.py",
            ha="center", va="center", fontsize=8.4, style="italic", color=SUBINK)
    _save(fig, "training_pipeline.png")


if __name__ == "__main__":
    app_architecture()
    training_pipeline()
