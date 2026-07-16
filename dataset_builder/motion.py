"""Motion extraction: seed many feature points across the ROI, track them with
sparse Lucas-Kanade, and keep the *raw per-point tracks* (the lossless
substrate). A cell-mean tensor is then *derived* from those tracks so the two
dataset variants (raw / used) are guaranteed consistent.

Raw substrate (per video):
    tracks   [P, 2, T]  per-point (dx, dy) displacement vs the previous kept
                        frame, in pixels.
    valid    [P, T]     per-point validity (1 = LK tracked it this frame).
    init_xy  [P, 2]     initial point position, normalised to the ROI box [0,1].
    cell_id  [P]        which 8x8 cell each point was seeded in.

Used representation (derived from the raw tracks):
    tensor   [3, n_cells, T]  channels (dx, dy, valid_mask); each cell is the
                        MEAN displacement over its valid points (no position).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import BuildConfig, CH_DX, CH_DY, CH_VALID


@dataclass
class MotionResult:
    tensor: np.ndarray          # [3, n_cells, T] cell-mean (dx, dy, valid)
    tracks: np.ndarray          # [P, 2, T] raw per-point displacement (px)
    valid: np.ndarray           # [P, T] per-point validity mask
    init_xy: np.ndarray         # [P, 2] initial coords normalised to ROI [0,1]
    cell_id: np.ndarray         # [P] seed cell index per point
    selected_idx: List[int]     # original frame index for each output column
    init_points: np.ndarray     # [P, 2] initial coords (full-frame) for QC
    cells: List[Tuple[int, int, int, int]]
    valid_ratio: float          # fraction of valid samples in the cell tensor


def cells_from_box(box: Tuple[int, int, int, int], grid: int) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    xs = np.linspace(x1, x2, grid + 1).round().astype(int)
    ys = np.linspace(y1, y2, grid + 1).round().astype(int)
    cells = []
    for j in range(grid):          # row (y) outer -> reading order, top-to-bottom
        for i in range(grid):      # col (x)
            cells.append((xs[i], ys[j], xs[i + 1], ys[j + 1]))
    return cells


def _seed_points(gray: np.ndarray, box, cfg: BuildConfig):
    """Seed up to `points_per_cell` features in each cell; return (points[P,1,2],
    cell_id[P]). Cells with no texture simply contribute no points."""
    cells = cells_from_box(box, cfg.grid)
    all_pts, all_cell = [], []
    for c, (cx1, cy1, cx2, cy2) in enumerate(cells):
        region = gray[cy1:cy2, cx1:cx2]
        if region.shape[0] <= 5 or region.shape[1] <= 5:
            continue
        corners = cv2.goodFeaturesToTrack(
            region,
            maxCorners=cfg.points_per_cell,
            qualityLevel=cfg.seed_quality,
            minDistance=cfg.seed_min_distance,
        )
        if corners is None or len(corners) == 0:
            continue
        pts = corners.reshape(-1, 2).astype(np.float32) + np.array([cx1, cy1], np.float32)
        all_pts.append(pts)
        all_cell.append(np.full(len(pts), c, dtype=np.int64))
    if not all_pts:
        return None, None
    pts = np.concatenate(all_pts, axis=0)
    cell_id = np.concatenate(all_cell, axis=0)
    return pts.reshape(-1, 1, 2), cell_id


def _select_indices(first_idx: int, n_frames: int, fps: float, cfg: BuildConfig) -> List[int]:
    span = n_frames - first_idx
    t_out = int(span * cfg.target_fps / fps)
    targets: List[int] = []
    for k in range(t_out):
        idx = first_idx + int(round(k * fps / cfg.target_fps))
        if idx >= n_frames:
            break
        if not targets or idx > targets[-1]:   # keep strictly increasing
            targets.append(idx)
    return targets


def cell_mean_tensor(tracks: np.ndarray, valid: np.ndarray, cell_id: np.ndarray,
                     n_cells: int) -> np.ndarray:
    """Derive the [3, n_cells, T] cell-mean tensor from raw per-point tracks.

    Each cell = mean (dx, dy) over its *valid* points at each frame; the valid
    channel is 1 whenever the cell has at least one tracked point. Averaging is
    done in raw pixel units (normalisation is deferred to training), so the
    per-point SNR is preserved.
    """
    _, _, T = tracks.shape
    tensor = np.zeros((3, n_cells, T), dtype=np.float32)
    for c in range(n_cells):
        m = np.flatnonzero(cell_id == c)
        if m.size == 0:
            continue
        v = valid[m]                              # [k, T]
        cnt = v.sum(axis=0)                       # [T]
        has = cnt > 0
        if not has.any():
            continue
        dx = (tracks[m, 0, :] * v).sum(axis=0)
        dy = (tracks[m, 1, :] * v).sum(axis=0)
        tensor[CH_DX, c, has] = dx[has] / cnt[has]
        tensor[CH_DY, c, has] = dy[has] / cnt[has]
        tensor[CH_VALID, c, has] = 1.0
    return tensor


def extract_motion(video_path: str, box, first_idx: int, fps: float,
                   n_frames: int, cfg: BuildConfig) -> MotionResult:
    cells = cells_from_box(box, cfg.grid)           # full-frame cells (for QC)
    n_cells = len(cells)
    targets = _select_indices(first_idx, n_frames, fps, cfg)
    t_out = len(targets)

    # Optical flow runs on the ROI crop only: the LK image pyramid is then built
    # over a few hundred K pixels instead of the full 1920x1080 frame (~8x faster).
    x1, y1, x2, y2 = box
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    crop_box = (0, 0, x2 - x1, y2 - y1)
    offset = np.array([x1, y1], np.float32)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)

    lk_params = dict(
        winSize=cfg.lk_win,
        maxLevel=cfg.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    prev_gray = None
    prev_pts = None          # [P, 1, 2] in crop coordinates
    cell_id = None           # [P]
    tracks = None            # [P, 2, T]
    valid = None             # [P, T]
    init_points = np.zeros((0, 2), np.float32)
    init_xy = np.zeros((0, 2), np.float32)
    cur = first_idx
    ti = 0

    while ti < t_out and cur < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if cur == targets[ti]:
            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_pts, cell_id = _seed_points(gray, crop_box, cfg)
                if prev_pts is None:                 # no features at all in ROI
                    cap.release()
                    raise RuntimeError("no trackable features in ROI")
                P = prev_pts.shape[0]
                tracks = np.zeros((P, 2, t_out), dtype=np.float32)
                valid = np.zeros((P, t_out), dtype=np.float32)
                crop_xy = prev_pts[:, 0, :].copy()               # crop coords
                init_points = crop_xy + offset                   # full-frame (QC)
                init_xy = crop_xy / np.array([crop_w, crop_h], np.float32)
                valid[:, ti] = 1.0                               # t0: seeded, no disp
            else:
                new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None, **lk_params)
                st = st.reshape(-1).astype(bool)
                disp = (new_pts - prev_pts).reshape(-1, 2)       # [P, 2]
                tracks[st, 0, ti] = disp[st, 0]
                tracks[st, 1, ti] = disp[st, 1]
                valid[st, ti] = 1.0
                prev_pts[st] = new_pts[st]                       # advance tracked only
            prev_gray = gray
            ti += 1
        cur += 1

    cap.release()

    if tracks is None:
        raise RuntimeError("no frames processed")

    # trim if the video ended early
    if ti < t_out:
        tracks = tracks[:, :, :ti]
        valid = valid[:, :ti]
        targets = targets[:ti]

    tensor = cell_mean_tensor(tracks, valid, cell_id, n_cells)
    valid_ratio = float(tensor[CH_VALID].mean()) if tensor.size else 0.0

    return MotionResult(
        tensor=tensor,
        tracks=tracks,
        valid=valid,
        init_xy=init_xy.astype(np.float32),
        cell_id=cell_id,
        selected_idx=targets,
        init_points=init_points.astype(np.float32),
        cells=cells,
        valid_ratio=valid_ratio,
    )
