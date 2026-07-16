"""Training-time utilities for Breathing-Vision.

Windowing, normalization and the MIL DataLoader live here. Nothing in this
package touches the raw dataset on disk — windows are cut on-the-fly from the
full-length motion tensors + labels, so context length / stride / tolerance
stay free hyper-parameters (see `trainer.config.WindowConfig`).
"""
from .config import WindowConfig, DataConfig
from .dataset import ClipWindowDataset, subject_groups, list_clips, make_fold_loaders

__all__ = [
    "WindowConfig", "DataConfig",
    "ClipWindowDataset", "subject_groups", "list_clips", "make_fold_loaders",
]
