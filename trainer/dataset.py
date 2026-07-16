"""On-the-fly windowing MIL dataset.

Each clip on disk is stored full-length as `motion_cellmean.npz` (tensor
`[3, 64, T]` = dx, dy, valid) + `labels_10fps.npy` (`[T]` binary). We keep the
clips in memory (normalised once) and cut fixed 80-frame context windows lazily
in `__getitem__`, padding with zeros at the clip edges. One sample is a **bag**
of 64 cell-instances, each a `[2, 80]` (dx, dy) motion snippet, plus a
per-frame validity mask `[64, 80]` (0..1). No cell-level gating is applied: a
cell contributes on exactly the frames where it was tracked (a point that is
good for only part of the window still counts on those frames), and how to use
the mask in pooling is left to the model.

Nothing here is cached to disk: change `WindowConfig` and the samples change,
no rebuild needed.
"""
from __future__ import annotations

import os
import warnings
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset_builder.config import CH_DX, CH_DY, CH_VALID
from .config import DataConfig, WindowConfig


def list_clips(root: str) -> List[str]:
    """All clip folders under `root` that carry a motion tensor + labels."""
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if (os.path.isfile(os.path.join(d, "motion_cellmean.npz")) and
                os.path.isfile(os.path.join(d, "labels_10fps.npy"))):
            out.append(name)
    return out


def subject_groups(names: Sequence[str]) -> np.ndarray:
    """Group key = <set>_<subject>, e.g. AIR125_S01_003 -> 'AIR125_S01'.

    Splits MUST be by baby (subject), never by clip, so no subject leaks between
    train and test.
    """
    return np.array(["_".join(n.split("_")[:2]) for n in names])


# robust-normalization guards (see _robust_normalize): floor the IQR divisor at
# this fraction of the clip's typical cell-IQR, and clip the result.
NORM_FLOOR_FRAC = 0.5
NORM_CLIP = 10.0


def _robust_stats(motion: np.ndarray, valid_mask: np.ndarray,
                  eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the robust (median, scale) divisor for a motion tensor.

    motion: [2, 64, T]; valid_mask: [64, T] bool. Returns med, scale as [2,64,1]
    so they can be cached once and applied cheaply (see `_apply_norm`).
    """
    vm = valid_mask[None]                      # [1, 64, T] broadcast over channels
    masked = np.where(vm, motion, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(masked, axis=2, keepdims=True)          # [2, 64, 1]
        q75 = np.nanpercentile(masked, 75, axis=2, keepdims=True)
        q25 = np.nanpercentile(masked, 25, axis=2, keepdims=True)
    iqr = q75 - q25
    med = np.nan_to_num(med, nan=0.0)
    iqr = np.nan_to_num(iqr, nan=0.0)

    # Robust per-channel floor on the divisor. A near-static (noise) cell has a
    # tiny IQR; dividing by it amplifies pure jitter by orders of magnitude
    # (observed std ~ 1e4) and DROWNS the real breathing cells. We floor the
    # divisor at a fraction of the TYPICAL (median) cell IQR, so a static cell
    # stays near 0 instead of exploding, while genuinely moving cells keep their
    # own scale. A final clip caps any residual outlier.
    pos_iqr = np.where(iqr > eps, iqr, np.nan)                # [2,64,1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        typ = np.nanmedian(pos_iqr, axis=1, keepdims=True)   # [2,1,1] per channel
    typ = np.nan_to_num(typ, nan=1.0)
    typ = np.where(typ > eps, typ, 1.0)
    floor = np.maximum(NORM_FLOOR_FRAC * typ, eps)           # [2,1,1]
    scale = np.maximum(iqr, floor)
    return med.astype(np.float32), scale.astype(np.float32)


def _apply_norm(motion: np.ndarray, med: np.ndarray, scale: np.ndarray,
                valid_mask: np.ndarray) -> np.ndarray:
    """(motion - med) / scale, clipped, with invalid frames zeroed."""
    out = (motion - med) / scale
    out = np.clip(out, -NORM_CLIP, NORM_CLIP)
    out = np.where(valid_mask[None], out, 0.0).astype(np.float32)
    return out


def _robust_stats_batch(motion: np.ndarray, valid_mask: np.ndarray,
                        eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised `_robust_stats` over a batch of windows.

    motion: [B, 2, 64, T]; valid_mask: [B, 64, T] bool. Returns med, scale each
    [B, 2, 64, 1]. Computing the quantiles for many windows in one nanpercentile
    call is far cheaper than a Python loop of single-window calls.
    """
    vm = valid_mask[:, None]                              # [B,1,64,T] over channels
    masked = np.where(vm, motion, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q = np.nanpercentile(masked, [25, 50, 75], axis=3)   # [3,B,2,64]
    q25, med, q75 = q[0], q[1], q[2]                      # each [B,2,64]
    iqr = np.nan_to_num(q75 - q25, nan=0.0)
    med = np.nan_to_num(med, nan=0.0)

    pos_iqr = np.where(iqr > eps, iqr, np.nan)            # [B,2,64]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        typ = np.nanmedian(pos_iqr, axis=2)              # [B,2] per channel
    typ = np.nan_to_num(typ, nan=1.0)
    typ = np.where(typ > eps, typ, 1.0)[:, :, None]      # [B,2,1]
    floor = np.maximum(NORM_FLOOR_FRAC * typ, eps)       # [B,2,1]
    scale = np.maximum(iqr, floor)                       # [B,2,64]
    return (med[..., None].astype(np.float32),
            scale[..., None].astype(np.float32))


def _robust_normalize(motion: np.ndarray, valid_mask: np.ndarray,
                      eps: float) -> np.ndarray:
    """Per-cell, per-channel (x - median) / IQR over the *valid* frames.

    motion: [2, 64, T]; valid_mask: [64, T] bool. Invalid frames are set to 0
    after normalising, so padded/dead frames contribute nothing.
    """
    med, scale = _robust_stats(motion, valid_mask, eps)
    return _apply_norm(motion, med, scale, valid_mask)


class ClipWindowDataset(Dataset):
    """Bag-of-cells windows cut on-the-fly from full-length clips."""

    def __init__(self, names: Sequence[str], dcfg: DataConfig, wcfg: WindowConfig):
        self.dcfg = dcfg
        self.wcfg = wcfg
        self.names = list(names)

        self._motion: List[np.ndarray] = []    # per clip: [2, 64, T] normalised
        self._valid: List[np.ndarray] = []     # per clip: [64, T] valid ratio
        self._labels: List[np.ndarray] = []    # per clip: [T]
        self.index: List[Tuple[int, int]] = []  # (clip_idx, t0)
        # per-window robust norm is streaming-faithful but must NOT be recomputed
        # every epoch (nanmedian/nanpercentile per overlapping window is ~60x the
        # training cost). We cache each window's (median, scale) ONCE here and just
        # apply them in __getitem__.
        #   robust_window        -> centred: (x - median) / IQR
        #   robust_window_noctr  -> IQR only (no centring). The motion is a VELOCITY
        #     signal (~zero-mean; a breath extremum sits AT a zero-crossing), and the
        #     per-window median is empirically ~0 (histogram: 99% of active cells
        #     |median|<0.1 px/frame), so subtracting it is a near no-op that only
        #     risks nudging the zero-crossing. Dropping it keeps the crossing exact.
        self._perwin = dcfg.norm in ("robust_window", "robust_window_noctr")
        self._center = (dcfg.norm != "robust_window_noctr")
        self._win_med: List[np.ndarray] = []   # per index: [2, 64, 1]
        self._win_scale: List[np.ndarray] = []  # per index: [2, 64, 1]

        for ci, name in enumerate(self.names):
            d = os.path.join(dcfg.root, name)
            tensor = np.load(os.path.join(d, "motion_cellmean.npz"))["tensor"]
            labels = np.load(os.path.join(d, "labels_10fps.npy")).astype(np.float32)
            T = tensor.shape[2]

            motion = tensor[[CH_DX, CH_DY]].astype(np.float32)   # [2, 64, T]
            valid = tensor[CH_VALID].astype(np.float32)          # [64, T]
            if dcfg.norm == "robust":
                motion = _robust_normalize(motion, valid > 0, dcfg.eps)
            elif dcfg.norm in ("robust_window", "robust_window_noctr"):
                pass  # normalized PER context-window in __getitem__ (streaming-faithful:
                      # each 80-frame tensor is self-normalized, so a noisy segment of the
                      # clip can't rescale a calm segment elsewhere). Store raw here.
            elif dcfg.norm != "none":
                raise ValueError(f"unknown norm '{dcfg.norm}'")

            self._motion.append(motion)
            self._valid.append(valid)
            self._labels.append(labels)

            # test-window start positions fully inside the clip
            for t0 in range(0, T - wcfg.test + 1, wcfg.stride):
                self.index.append((ci, t0))

        if self._perwin:
            self._precompute_winstats()

    def _precompute_winstats(self, chunk: int = 256) -> None:
        """Cache each window's (median, scale) once, computed in vectorised
        chunks (see `_robust_stats_batch`). Padded edges are invalid -> excluded."""
        w = self.wcfg
        n = len(self.index)
        self._win_med = [None] * n      # type: ignore[list-item]
        self._win_scale = [None] * n    # type: ignore[list-item]
        for start in range(0, n, chunk):
            ks = range(start, min(n, start + chunk))
            b = len(ks)
            ms = np.zeros((b, 2, 64, w.context), np.float32)
            vs = np.zeros((b, 64, w.context), np.float32)
            for j, k in enumerate(ks):
                ci, t0 = self.index[k]
                g0 = t0 - w.past
                ms[j] = self._slice(self._motion[ci], g0, g0 + w.context)
                vs[j] = self._slice(self._valid[ci], g0, g0 + w.context)
            med, scale = _robust_stats_batch(ms, vs > 0, self.dcfg.eps)
            if not self._center:
                med = np.zeros_like(med)     # IQR-only: don't shift the zero-crossing
            for j, k in enumerate(ks):
                self._win_med[k] = med[j]
                self._win_scale[k] = scale[j]

    def __len__(self) -> int:
        return len(self.index)

    def _slice(self, arr: np.ndarray, g0: int, g1: int) -> np.ndarray:
        """Slice arr[..., g0:g1] with zero padding past either edge."""
        T = arr.shape[-1]
        width = g1 - g0
        out = np.zeros(arr.shape[:-1] + (width,), dtype=arr.dtype)
        s0, s1 = max(0, g0), min(T, g1)
        if s1 > s0:
            out[..., s0 - g0:s1 - g0] = arr[..., s0:s1]
        return out

    def __getitem__(self, i: int) -> dict:
        w = self.wcfg
        ci, t0 = self.index[i]
        g0 = t0 - w.past                 # global context start
        g1 = g0 + w.context              # global context end (= t0 + test + future)

        motion = self._slice(self._motion[ci], g0, g1)   # [2, 64, C]
        valid = self._slice(self._valid[ci], g0, g1)     # [64, C]

        # per-window (streaming-faithful) robust norm: apply the median/scale that
        # were computed ONCE (in __init__) from THIS window's valid frames only.
        if self._perwin:
            motion = _apply_norm(motion, self._win_med[i], self._win_scale[i],
                                 valid > 0)

        # bag: cells first, then (channel, time) -> [64, 2, C]
        x = np.transpose(motion, (1, 0, 2))

        labels = self._labels[ci]
        lo = max(0, t0 - w.label_tolerance)
        hi = min(labels.shape[0], t0 + w.test + w.label_tolerance)
        y = np.float32(labels[lo:hi].sum() > 0)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),          # [64, 2, C]
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),  # [64, C] per-frame validity (0..1)
            "y": torch.tensor(y),                                    # scalar
            "clip": torch.tensor(ci, dtype=torch.long),
            "t0": torch.tensor(t0, dtype=torch.long),
        }

    def pos_weight(self) -> float:
        """neg/pos ratio over all windows — handy as BCE `pos_weight`."""
        pos = sum(int(self._labels[ci][max(0, t0 - self.wcfg.label_tolerance):
                                       t0 + self.wcfg.test + self.wcfg.label_tolerance].sum() > 0)
                  for ci, t0 in self.index)
        neg = len(self.index) - pos
        return float(neg) / float(max(pos, 1))


def make_fold_loaders(dcfg: DataConfig, wcfg: WindowConfig, *, n_splits: int = 5,
                      fold: int = 0, batch_size: int = 256, num_workers: int = 0,
                      shuffle_train: bool = True):
    """Build (train_loader, val_loader, (train_names, val_names)) for one
    subject-grouped fold."""
    from sklearn.model_selection import GroupKFold

    names = list_clips(dcfg.root)
    groups = subject_groups(names)
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(names, groups=groups))
    tr_idx, va_idx = splits[fold]
    tr_names = [names[i] for i in tr_idx]
    va_names = [names[i] for i in va_idx]

    tr_ds = ClipWindowDataset(tr_names, dcfg, wcfg)
    va_ds = ClipWindowDataset(va_names, dcfg, wcfg)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle_train,
                       num_workers=num_workers, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, drop_last=False)
    return tr_ld, va_ld, (tr_names, va_names)


def make_tvt_loaders(dcfg: DataConfig, wcfg: WindowConfig, *, n_splits: int = 6,
                     fold: int = 0, batch_size: int = 256, num_workers: int = 0,
                     shuffle_train: bool = True):
    """Build (train, val, test) loaders for one subject-grouped fold, matching the
    AIR-400 protocol (6-fold, subject-wise, three disjoint splits per fold).

    `GroupKFold(n_splits)` partitions subjects into `n_splits` disjoint test
    groups (each subject is the held-out test exactly once). For fold `i`:
        test  = group i          (reported on -- untouched by tuning)
        val   = group (i+1) % n  (model/threshold/HP selection)
        train = the remaining groups
    All three are subject-disjoint, so no baby leaks across splits and the HP
    search (on val) never sees the test subjects. Returns loaders + name lists.
    """
    from sklearn.model_selection import GroupKFold

    names = list_clips(dcfg.root)
    groups = subject_groups(names)
    splitter = GroupKFold(n_splits=n_splits)
    # test_idx of split k = the k-th disjoint held-out group (partition of all)
    test_groups = [te for _, te in splitter.split(names, groups=groups)]

    te_idx = test_groups[fold]
    va_idx = test_groups[(fold + 1) % n_splits]
    holdout = set(te_idx.tolist()) | set(va_idx.tolist())
    tr_idx = np.array([i for i in range(len(names)) if i not in holdout])

    tr_names = [names[i] for i in tr_idx]
    va_names = [names[i] for i in va_idx]
    te_names = [names[i] for i in te_idx]

    def _loader(nm, shuffle):
        ds = ClipWindowDataset(nm, dcfg, wcfg)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=False)

    tr_ld = _loader(tr_names, shuffle_train)
    va_ld = _loader(va_names, False)
    te_ld = _loader(te_names, False)
    return tr_ld, va_ld, te_ld, (tr_names, va_names, te_names)
