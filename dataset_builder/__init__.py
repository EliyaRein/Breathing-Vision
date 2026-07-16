"""AIR-400 dataset builder.

Turns a folder tree of mp4 + matching hdf5 files into per-video training
tensors (sparse optical-flow motion + binary chunk labels), plus QC artifacts.
"""
from .config import BuildConfig, DEFAULT

__all__ = ["BuildConfig", "DEFAULT"]
