"""Central configuration for the AIR-400 dataset builder.

All tunables live here so the pipeline stays reproducible and easy to audit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

# Repo root (…/Breathing-Vision), so default weight paths are portable across machines.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class BuildConfig:
    # --- temporal ---
    target_fps: float = 10.0          # all videos are resampled to this rate
    # NOTE: windowing hyper-params (context/test-window/stride/label_tolerance)
    # are NOT here anymore. Windows are cut on-the-fly at train time from the
    # full-length motion tensor + labels; see `trainer/config.py`.

    # --- peak derivation (labels) ---
    # Peaks are derived from the GT `respiration` waveform (present in every
    # hdf5), not from the optional `impulse` field (only AIR_125 has it). On
    # AIR_125 the derived peaks reproduce `impulse` EXACTLY (106/106 clips,
    # 2646/2646 peaks, 0 false positives), so this unifies AIR_125 + AIR_175.
    #
    # The `respiration` GT is a *synthesised* waveform: every breath is rendered
    # as an identical kernel of fixed height (~0.1995 across the whole dataset,
    # verified on 8795 peaks). We exploit that structure instead of a fragile
    # range-relative prominence: an ABSOLUTE height threshold self-calibrated to
    # the kernel height (median of local maxima -> immune to overlaps/edges
    # inflating the range), plus explicit boundary handling for breaths whose
    # peak lands on the first/last frame (which find_peaks structurally drops).
    peak_min_sep_s: float = 0.5       # min seconds between peaks (~<=120 bpm)
    peak_height_frac: float = 0.5     # detect threshold = frac * median kernel height
    peak_height_cap: float = 0.15     # ceiling on the threshold (< a single bump)
    peak_edge_frac: float = 0.9       # boundary breath must reach frac * kernel height

    # --- spatial / ROI ---
    grid: int = 8                     # grid -> grid*grid = 64 cells
    # Portable default; if the file is absent, ultralytics auto-downloads it by name.
    yolo_weights: str = os.path.join(_REPO_ROOT, "yolov8m.pt")
    yolo_person_class: int = 0        # COCO 'person'
    # ROI is aggregated over several frames (like AIR-400) instead of a single frame,
    # because the baby is often partly covered and a single frame can miss the detection.
    roi_n_samples: int = 5            # frames sampled across the video for detection
    roi_sample_conf: float = 0.10     # low conf passed to YOLO so weak hits are returned
    roi_keep_conf: float = 0.25       # only boxes >= this confidence are aggregated
    blank_std_thresh: float = 10.0    # frame.std() below this == blank frame
    blank_search_limit: int = 60      # how many leading frames to scan for first real frame

    # --- optical flow ---
    # Up to `points_per_cell` feature points are seeded *per cell* (uniform
    # coverage, no single textured cell hogs the budget) and tracked. The raw
    # per-point tracks are the lossless substrate; the "used" representation
    # reduces each small (8x8) cell to the MEAN displacement of its valid
    # points. Within a small cell points are near-co-located -> same phase, so
    # averaging denoises (~sqrt(ppc)) without cancelling anti-phase signal.
    # 8 grid x 8 ppc = up to 512 raw points / 64 cell-means (the "8x8x8" config).
    points_per_cell: int = 8
    seed_quality: float = 0.01
    seed_min_distance: int = 4
    lk_win: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3

    @property
    def n_cells(self) -> int:
        return self.grid * self.grid

    @property
    def channels(self) -> int:
        return 3  # dx, dy, valid_mask


# channel indices inside the motion tensor [C, n_cells, T]
CH_DX = 0
CH_DY = 1
CH_VALID = 2

DEFAULT = BuildConfig()
