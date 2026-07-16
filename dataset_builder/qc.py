"""Quality-control outputs for manual review:
  1. qc_roi.png  - ROI box + grid + seeded points on the first real frame.
  2. motion_full.xlsx - the full (un-chunked) cell-mean tensor + labels +
     respiration (written for the 'used' variant only).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .motion import MotionResult
from .labels import LabelData


def save_roi_image(path: str, frame_bgr: np.ndarray, box, cells, init_points,
                   source: str, confidence: float) -> None:
    img = frame_bgr.copy()
    x1, y1, x2, y2 = box

    # grid cells (thin gray)
    for (cx1, cy1, cx2, cy2) in cells:
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (180, 180, 180), 1)

    # ROI box (green)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # tracked points (red dots) - many now, so keep them small
    if init_points is not None:
        for px, py in init_points:
            cv2.circle(img, (int(round(px)), int(round(py))), 2, (0, 0, 255), -1)

    label = f"ROI: {source} ({confidence:.2f})" if source == "yolo" else "ROI: full frame (fallback)"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(path, img)


def save_motion_excel(path: str, motion: MotionResult, labels: LabelData,
                      target_fps: float) -> None:
    tensor = motion.tensor              # [3, n_cells, T]
    _, n_cells, t = tensor.shape

    cols = {
        "frame_out": np.arange(t),
        "orig_frame_idx": np.asarray(motion.selected_idx[:t]),
        "time_s": np.arange(t) / target_fps,
    }
    for c in range(n_cells):
        cols[f"cell{c:02d}_dx"] = tensor[0, c, :]
        cols[f"cell{c:02d}_dy"] = tensor[1, c, :]
        cols[f"cell{c:02d}_valid"] = tensor[2, c, :]

    cols["label"] = labels.labels[:t]
    cols["respiration"] = labels.respiration[:t]

    df = pd.DataFrame(cols)
    df.to_excel(path, index=False)
