"""Stage 1 - decoding + peak-timing F1 (the source-of-truth metric).

This module is deliberately model-agnostic and free of any training code: it
turns a per-window score sequence for a clip into discrete predicted peak
frames, and scores them against the GT peak frames. The training harness
(stage 2) imports these functions to evaluate / early-stop on the REAL metric.

Design:
  * decode = per-window threshold, NO neighbour merging. A positive window ->
    one predicted peak placed at the window CENTRE (t0 + test//2). Two adjacent
    positive windows therefore stay TWO peaks (fast-rate preserved).
  * match  = ONE-TO-ONE (greedy by distance) within a tolerance radius.
  * primary tolerance = 4 frames (= ±2 beyond the 5-frame window edge, since the
    peak sits at the centre). Also report exact-match (tol 0) for transparency.

Sanity: feeding the GT itself as the prediction must give F1 = 1.0 at any tol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


DEFAULT_TOLS = (0, 4)
PRIMARY_TOL = 4


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# --------------------------------------------------------------------------- #
# decode: window scores -> predicted peak frames
# --------------------------------------------------------------------------- #
def decode_peaks(scores: Sequence[float], t0s: Sequence[int], test: int,
                 threshold: float, *, from_logits: bool = False,
                 min_distance: int = 0) -> np.ndarray:
    """Per-window threshold -> peak at window centre.

    scores : per-window score (prob, or logit if from_logits)
    t0s    : global start frame of each window's test span
    test   : test-window length (peak placed at t0 + test//2)
    min_distance : if >0, greedy score-ranked non-max suppression -> no two kept
        peaks are closer than `min_distance` frames. Needed when the training
        target is widened (label_tolerance>0) or the window is fine (small test),
        so one breath firing several adjacent windows collapses to ONE peak.
        Safe up to the real inter-breath floor (~9 frames after chunk cutting).
    """
    s = np.asarray(scores, dtype=np.float64)
    if from_logits:
        s = sigmoid(s)
    t0 = np.asarray(t0s, dtype=np.int64)
    centre = t0 + test // 2
    keep = s > threshold
    if not np.any(keep):
        return np.array([], dtype=np.int64)
    c, sc = centre[keep], s[keep]
    if min_distance <= 0:
        return np.sort(c).astype(np.int64)
    chosen: List[int] = []
    for idx in np.argsort(-sc):                 # highest score first
        p = int(c[idx])
        if all(abs(p - q) >= min_distance for q in chosen):
            chosen.append(p)
    return np.sort(np.array(chosen, dtype=np.int64))


# --------------------------------------------------------------------------- #
# match: one-to-one within tolerance
# --------------------------------------------------------------------------- #
def match_counts(pred: Sequence[int], gt: Sequence[int], tol: int) -> Tuple[int, int, int]:
    """Return (tp, fp, fn) with one-to-one greedy-by-distance matching."""
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    if pred.size == 0:
        return 0, 0, int(gt.size)
    if gt.size == 0:
        return 0, int(pred.size), 0

    pairs = []
    for i, p in enumerate(pred):
        d = np.abs(gt - p)
        for j in np.flatnonzero(d <= tol):
            pairs.append((int(d[j]), i, int(j)))
    pairs.sort()

    used_p, used_g, tp = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            tp += 1
    fp = int(pred.size) - tp
    fn = int(gt.size) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# --------------------------------------------------------------------------- #
# aggregate over clips, at several tolerances
# --------------------------------------------------------------------------- #
@dataclass
class ScoreRow:
    tol: int
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def score_clips(preds: Sequence[Sequence[int]], gts: Sequence[Sequence[int]],
                tols: Sequence[int] = DEFAULT_TOLS) -> dict:
    """Micro-averaged P/R/F1 across clips, one row per tolerance.

    preds[i], gts[i] are the predicted / GT peak frames of clip i.
    Returns {tol: ScoreRow}.
    """
    out = {}
    for tol in tols:
        TP = FP = FN = 0
        for pred, gt in zip(preds, gts):
            tp, fp, fn = match_counts(pred, gt, tol)
            TP += tp
            FP += fp
            FN += fn
        p, r, f1 = prf(TP, FP, FN)
        out[tol] = ScoreRow(tol, p, r, f1, TP, FP, FN)
    return out


# --------------------------------------------------------------------------- #
# secondary metric: breathing-rate MAE (BPM)
# --------------------------------------------------------------------------- #
def bpm(peaks: Sequence[int], n_frames: int, fps: float) -> float:
    """Mean breathing rate in breaths-per-minute from peak frames."""
    peaks = np.asarray(peaks, dtype=np.float64)
    if peaks.size >= 2:
        mean_gap = np.mean(np.diff(np.sort(peaks)))      # frames/breath
        return 60.0 * fps / mean_gap if mean_gap > 0 else 0.0
    if n_frames > 0:
        return 60.0 * fps * peaks.size / n_frames
    return 0.0


def bpm_mae(preds: Sequence[Sequence[int]], gts: Sequence[Sequence[int]],
            n_frames: Sequence[int], fps: float) -> float:
    errs = [abs(bpm(p, nf, fps) - bpm(g, nf, fps))
            for p, g, nf in zip(preds, gts, n_frames)]
    return float(np.mean(errs)) if errs else 0.0


def format_scores(rows: dict) -> str:
    lines = ["tol  precision  recall     f1        (tp/fp/fn)"]
    for tol in sorted(rows):
        r = rows[tol]
        lines.append(f"{r.tol:<4d} {r.precision:8.3f}  {r.recall:8.3f}  "
                     f"{r.f1:8.3f}   ({r.tp}/{r.fp}/{r.fn})")
    return "\n".join(lines)
