"""Windowing + data configuration for training.

These are the hyper-parameters that used to be (wrongly) baked into the dataset
at build time. They now live here and are applied on-the-fly, so changing the
context length or stride never requires rebuilding anything.

Layout of one context window:

      |<------------------ context = 80 ------------------>|
      |<-------- past = 63 -------->|<test=7>|<-future=10->|
      ^g0                           ^t0      ^t0+7         ^t0+17
                         |----- label region (test) -----|

  * `t0` is the GLOBAL frame index where the 7-frame test window starts.
  * the label is 1 iff any GT peak falls in [t0 - tol, t0 + test + tol).
  * `label_tolerance` is 0 for training: the scoring tolerance is a *scoring*
    radius only, it never widens the training target.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowConfig:
    context: int = 80          # total frames fed to the model
    past: int = 63             # frames before the test window
    test: int = 7              # frames the label refers to (the target)
    future: int = 10           # frames after the test window (near-causal peek)
    stride: int = 7            # hop between consecutive test windows
    label_tolerance: int = 0   # train-time label widening (0 = off; scoring tol is separate)
    min_distance: int = 0      # decode-time NMS: min frames between kept peaks (0 = off)

    def __post_init__(self) -> None:
        if self.past + self.test + self.future != self.context:
            raise ValueError(
                f"past+test+future ({self.past}+{self.test}+{self.future}) "
                f"must equal context ({self.context})")
        for k in ("context", "past", "test", "future", "stride"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be positive")
        if self.label_tolerance < 0:
            raise ValueError("label_tolerance must be >= 0")
        if self.min_distance < 0:
            raise ValueError("min_distance must be >= 0")


@dataclass(frozen=True)
class DataConfig:
    # where the "used" (cell-mean) clips live: <root>/<clip>/motion_cellmean.npz
    # Portable: set the BV_DATASET env var, else default to a path relative to CWD.
    root: str = os.environ.get("BV_DATASET", os.path.join("dataset_out", "dataset_used"))
    norm: str = "robust_window_noctr"
    eps: float = 1e-6
