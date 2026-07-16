"""ROI detection: find the first non-blank frame, then a person box via YOLO.

The box is aggregated (median) over several frames sampled across the video,
which is far more robust than a single frame when the baby is partly covered.
Falls back to the whole frame only when no frame yields a person.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .config import BuildConfig


@dataclass
class RoiResult:
    box: tuple          # (x1, y1, x2, y2) ints
    source: str         # "yolo" | "full_frame"
    confidence: float   # mean conf of aggregated boxes, or 0.0 for fallback
    first_frame_idx: int
    frame_bgr: np.ndarray  # the first real frame (for QC)
    n_detections: int = 0  # how many sampled frames contributed a box


def find_first_real_frame(cap: cv2.VideoCapture, cfg: BuildConfig) -> tuple[int, Optional[np.ndarray]]:
    """Scan leading frames and return (index, frame) of the first non-blank one."""
    idx = 0
    while idx < cfg.blank_search_limit:
        ret, frame = cap.read()
        if not ret:
            break
        if float(frame.std()) >= cfg.blank_std_thresh:
            return idx, frame
        idx += 1
    # all leading frames blank (or video shorter than limit): fall back to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    return 0, (frame if ret else None)


def _largest_person_box(result, person_class: int):
    best_box, best_conf, best_area = None, 0.0, -1.0
    for b in result.boxes:
        if int(b.cls[0]) != person_class:
            continue
        conf = float(b.conf[0])
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        # prefer the largest person; ties broken by confidence
        if area > best_area or (area == best_area and conf > best_conf):
            best_box, best_conf, best_area = (x1, y1, x2, y2), conf, area
    return best_box, best_conf


def detect_roi(video_path: str, yolo_model, cfg: BuildConfig) -> RoiResult:
    cap = cv2.VideoCapture(video_path)
    first_idx, frame0 = find_first_real_frame(cap, cfg)
    if frame0 is None:
        cap.release()
        raise RuntimeError("could not read any frame from video")

    h, w = frame0.shape[:2]
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_idx = np.unique(
        np.linspace(first_idx, max(first_idx, n_frames - 1),
                    cfg.roi_n_samples).round().astype(int))

    boxes: List[tuple] = []
    confs: List[float] = []
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        result = yolo_model(frame, verbose=False, conf=cfg.roi_sample_conf)[0]
        box, conf = _largest_person_box(result, cfg.yolo_person_class)
        if box is not None and conf >= cfg.roi_keep_conf:
            boxes.append(box)
            confs.append(conf)
    cap.release()

    if not boxes:
        return RoiResult((0, 0, w, h), "full_frame", 0.0, first_idx, frame0, 0)

    med = np.median(np.asarray(boxes), axis=0)
    x1 = max(0, int(round(med[0])))
    y1 = max(0, int(round(med[1])))
    x2 = min(w, int(round(med[2])))
    y2 = min(h, int(round(med[3])))
    return RoiResult((x1, y1, x2, y2), "yolo", float(np.mean(confs)),
                     first_idx, frame0, len(boxes))
