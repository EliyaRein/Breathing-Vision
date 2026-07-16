"""Label handling: read HDF5 peak impulses and align them to the resampled
motion timeline.

Alignment is done against the *exact* original frame indices that the motion
extractor kept (`selected_idx`), so peaks never drift due to fps resampling.

Note: windowing into training samples is NOT done here. It happens on-the-fly
at train time (see `trainer/`) from the full-length motion tensor + labels, so
context length / stride stay free hyperparameters and nothing goes stale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import h5py
import numpy as np
from scipy.signal import find_peaks

from .config import BuildConfig


@dataclass
class LabelData:
    labels: np.ndarray          # [T] binary, aligned to motion columns
    respiration: np.ndarray     # [T] waveform resampled to motion columns
    n_peaks_orig: int
    n_peaks_aligned: int


def load_hdf5(hdf5_path: str):
    """Return (respiration, impulse_or_None).

    Every AIR-400 hdf5 carries the GT `respiration` waveform; only AIR_125 also
    ships a pre-computed `impulse` peak vector. We read both but never depend on
    `impulse` (see derive_impulse).
    """
    with h5py.File(hdf5_path, "r") as f:
        respiration = np.asarray(f["respiration"][()]).astype(np.float64).ravel()
        impulse = (np.asarray(f["impulse"][()]).astype(np.float64).ravel()
                   if "impulse" in f else None)
    return respiration, impulse


def derive_impulse(respiration: np.ndarray, fps: float, cfg: BuildConfig) -> np.ndarray:
    """Derive a binary peak vector from the GT respiration waveform.

    Structure-aware: the GT is a synthesised waveform where every breath is an
    identical kernel of fixed height H (~0.1995 dataset-wide). We therefore use
    an ABSOLUTE height threshold self-calibrated to H (median of local maxima,
    robust to the ~0.3% overlapping bumps) rather than a range-relative
    prominence (which is inflated by overlaps / truncated edge bumps). Breaths
    whose peak lands on the first/last frame are added explicitly, because
    find_peaks cannot return boundary indices.

    Validated on AIR_125: reproduces the shipped `impulse` on 106/106 clips
    (2646/2646 peaks, 0 false positives), recovering 10 edge breaths the old
    prominence rule missed.
    """
    resp = np.nan_to_num(np.asarray(respiration, dtype=np.float64))
    n = len(resp)
    if n < 2 or resp.max() <= 0:
        return np.zeros(n, dtype=np.float64)
    distance = max(1, int(round(fps * cfg.peak_min_sep_s)))

    cand, props = find_peaks(resp, height=1e-6)
    if len(cand) == 0:
        return np.zeros(n, dtype=np.float64)
    H = float(np.median(props["peak_heights"]))          # kernel height (~0.1995)
    thr = min(cfg.peak_height_frac * H, cfg.peak_height_cap)

    idx, _ = find_peaks(resp, height=thr, distance=distance)
    idx = list(idx)

    # Boundary breaths: only when the edge sample is at ~full kernel height (the
    # peak actually lands on the boundary frame), not a partial flank of a bump
    # whose true peak lies outside the recording (which `impulse` omits).
    edge_thr = cfg.peak_edge_frac * H
    if resp[0] >= edge_thr and resp[0] >= resp[1] and not any(p < distance for p in idx):
        idx = [0] + idx
    if resp[-1] >= edge_thr and resp[-1] >= resp[-2] and not any(p > n - 1 - distance for p in idx):
        idx = idx + [n - 1]

    impulse = np.zeros(n, dtype=np.float64)
    impulse[np.asarray(sorted(idx), dtype=int)] = 1.0
    return impulse


def resp_frame_scale(n_resp: int, n_frames: int | None, fps: float) -> float:
    """Frames per respiration-sample (respiration-index -> video-frame).

    Two AIR-400 conventions, distinguished by the recovered sensor rate:
      * per-frame respiration (AIR_125): len(resp) == n_frames, rate == fps
        -> scale 1.0 (resp index already equals frame index).
      * fixed-rate respiration (AIR_175): resp sampled at a constant sensor rate
        (~10 Hz) anchored at t=0; the video may carry a few extra trailing frames
        beyond the annotation window. A 15 fps video vs 10 Hz resp needs scale
        1.5 (the S07 case).

    We recover the CLEAN sensor rate by rounding `L*fps/n_frames`, so the handful
    of unannotated trailing frames don't inflate the scale. Using the raw ratio
    `n_frames/L` instead injects a ~1.3% stretch that measurably *lowers* the
    motion<->GT correlation on the 10 fps clips (verified empirically).
    """
    if not n_frames or n_resp < 2:
        return 1.0
    resp_rate = round(n_resp * fps / n_frames)          # clean sensor rate: 10/15/30
    return (fps / resp_rate) if resp_rate > 0 else 1.0


def align_labels(impulse: np.ndarray, respiration: np.ndarray,
                 selected_idx: List[int], fps: float, cfg: BuildConfig,
                 n_frames: int | None = None) -> LabelData:
    """Align GT peaks (respiration-sample units) to the motion columns.

    The `respiration`/`impulse` arrays are sampled on the annotation timeline,
    while `selected_idx` are ORIGINAL VIDEO frame indices at the video fps. When
    the two rates differ (e.g. AIR175_S07: 15 fps video vs 10 Hz respiration) a
    respiration index is NOT a video-frame index. We map every peak to the video
    frame axis via `resp_frame_scale` before matching it to `selected_idx`;
    otherwise labels get compressed toward the start (the S07 bug).
    """
    sel = np.asarray(selected_idx, dtype=np.int64)
    t_out = len(sel)
    labels = np.zeros(t_out, dtype=np.float32)

    L = len(respiration)
    scale = resp_frame_scale(L, n_frames, fps)

    peaks = np.flatnonzero(impulse > 0.5)
    spacing = max(1.0, fps / cfg.target_fps)   # original frames per output column
    aligned = 0
    for p in peaks:
        fp = p * scale                          # peak position on the video-frame axis
        k = int(np.argmin(np.abs(sel - fp)))
        if abs(sel[k] - fp) <= spacing:         # peak falls within coverage
            labels[k] = 1.0
            aligned += 1

    # respiration resampled onto the kept video frames (map frame -> resp index)
    resp_idx = np.clip(np.round(sel / scale).astype(np.int64), 0, L - 1)
    resp_out = respiration[resp_idx].astype(np.float32)

    return LabelData(labels, resp_out, int(len(peaks)), aligned)
