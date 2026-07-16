"""Stage 2 - training harness (imports the stage-1 metric as the real objective).

One place to train any of the three models on a subject-grouped fold and select
on the SOURCE-OF-TRUTH metric (peak-timing F1@4), never on the proxy loss:

    loss    : BCEWithLogitsLoss(pos_weight = neg/pos)  -- per-window presence.
    optim   : AdamW(lr, weight_decay).
    select  : after every epoch, run the full stage-1 decode on val at the decode
              threshold (0.60) -> F1@4. Early-stop + checkpoint on that F1, not on
              loss. The same threshold is used for the test decode.
    split   : GroupKFold by subject (no baby leaks across train/val).

Training treats labels as hard 0/1 (no label smoothing / no ignore-band). The
scoring tolerance (±4) is the only place timing slack lives.

This file is a LIBRARY only: it exposes `TrainConfig` + `train_fold` for the 
nested-CV protocol in `trainer.nested`.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import ModelConfig, build_model, count_params
from . import metrics as M
from .config import DataConfig, WindowConfig
from .dataset import ClipWindowDataset, make_tvt_loaders

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DECODE_THRESHOLD = 0.60


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Training knobs for one fold. Only the encoder (`model`), the four
    Optuna-tuned optimisation HPs (lr/weight_decay/dropout/warmup_frac) and
    operational settings vary. The decode threshold is `DECODE_THRESHOLD` (0.60)
    -- see `evaluate`."""
    model: str = "tcn"                 # tcn | gru | transformer
    pool: str = "attention"            
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    epochs: int = 30
    patience: int = 6                  # early-stop patience (epochs w/o F1 gain)
    max_grad_norm: float = 1.0         # gradient clipping (0 = off); tames TF divergence
    warmup_frac: float = 0.05          # linear LR warmup frac (a DEFAULT; tuned per-model by Optuna)
    use_scheduler: bool = True         # linear-warmup + cosine-decay
    min_lr_frac: float = 0.0           # cosine floor as a fraction of base lr
    batch_size: int = 256
    n_splits: int = 6                  # AIR-400 protocol: 6-fold subject-wise
    fold: int = 0
    ckpt_tag: str = ""                 # optional suffix to keep variant ckpts distinct
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_config(tc: TrainConfig, wcfg: WindowConfig) -> ModelConfig:
    # Only the encoder, dropout and the readout region depend on tc/wcfg.
    return ModelConfig(name=tc.model, pool=tc.pool, dropout=tc.dropout,
                       readout_start=wcfg.past, readout_len=wcfg.test)


def build_scheduler(opt, total_steps: int, warmup_frac: float, min_lr_frac: float):
    """Linear warmup then cosine decay to `min_lr_frac`*lr, stepped per batch.

    NOTE on fairness: a single fixed schedule is NOT automatically fair across
    architectures (a transformer usually wants more warmup / lower lr than a
    TCN/GRU). So `lr` and `warmup_frac` are per-model hyper-parameters, tuned by
    Optuna over a SHARED search space + SHARED budget -> each model is compared at
    its OWN optimum. This default schedule is only for first single runs. Warmup
    is what the course flags as essential for transformers.
    """
    import math

    warmup = max(1, int(warmup_frac * total_steps)) if warmup_frac > 0 else 0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if total_steps <= warmup:
            return 1.0
        prog = (step - warmup) / (total_steps - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        return min_lr_frac + (1.0 - min_lr_frac) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# --------------------------------------------------------------------------- #
# evaluation = the REAL metric (stage-1 decode on val)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device,
                   *, with_labels: bool = False):
    """Run the model over a loader, return per-clip {scores, t0s} + gts + nframes.

    Uses the loader's dataset to recover GT peak frames and clip lengths, so the
    decode is done on complete per-clip window sequences.

    with_labels=True also returns two FLAT arrays (y_flat, logit_flat) over every
    window, for window-level confusion-matrix stats (never changes the default
    4-tuple used by existing callers).
    """
    model.eval()
    ds: ClipWindowDataset = loader.dataset  # type: ignore[assignment]
    n_clips = len(ds.names)
    scores: List[List[float]] = [[] for _ in range(n_clips)]
    t0s: List[List[int]] = [[] for _ in range(n_clips)]
    y_flat: List[np.ndarray] = []
    s_flat: List[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        valid = batch["valid"].to(device)
        logit = model(x, valid).squeeze(1).float().cpu().numpy()   # [B]
        cis = batch["clip"].numpy()
        tts = batch["t0"].numpy()
        if with_labels:
            y_flat.append(batch["y"].numpy().astype(np.float32))
            s_flat.append(logit)
        for s, ci, t0 in zip(logit, cis, tts):
            scores[ci].append(float(s))
            t0s[ci].append(int(t0))

    gts = [np.flatnonzero(ds._labels[c]).astype(np.int64) for c in range(n_clips)]
    nframes = [int(ds._labels[c].shape[0]) for c in range(n_clips)]
    if with_labels:
        yf = np.concatenate(y_flat) if y_flat else np.zeros(0, np.float32)
        sf = np.concatenate(s_flat) if s_flat else np.zeros(0, np.float32)
        return scores, t0s, gts, nframes, yf, sf
    return scores, t0s, gts, nframes


def evaluate(model: nn.Module, loader: DataLoader, wcfg: WindowConfig,
             device: torch.device, fps: float = 10.0,
             threshold: float = DECODE_THRESHOLD) -> Dict:
    """Decode logits -> F1@{0,4} + BPM MAE at the decode `threshold` (0.60), the
    same value for the val and test splits."""
    scores, t0s, gts, nframes = collect_logits(model, loader, device)
    threshold = float(threshold)
    preds = [M.decode_peaks(s, t, wcfg.test, threshold, from_logits=True,
                            min_distance=wcfg.min_distance)
             for s, t in zip(scores, t0s)]
    rows = M.score_clips(preds, gts, M.DEFAULT_TOLS)
    return {"threshold": threshold, "f1_primary": rows[M.PRIMARY_TOL].f1,
            "rows": rows, "bpm_mae": M.bpm_mae(preds, gts, nframes, fps)}


# --------------------------------------------------------------------------- #
# training one fold
# --------------------------------------------------------------------------- #
def train_fold(tc: TrainConfig, dcfg: DataConfig, wcfg: WindowConfig, *,
               loaders=None, verbose: bool = True, eval_test: bool = True) -> Dict:
    """Train one fold. The model is selected on VAL (F1@4 at the decode
    threshold); the final numbers are ALSO reported on the untouched TEST split
    (AIR-400 protocol).

    `loaders`, if given, may be either the legacy 2-tuple
    (train, val, (tr_names, va_names)) -- then test == val (smoke only) -- or the
    3-way (train, val, test, (tr_names, va_names, te_names)).

    `eval_test=False` skips the test decode entirely -- use it for DEVELOPMENT
    (screens/ablations) so architecture choices are made on VAL only and the test
    is never peeked at before the single final measurement.
    """
    set_seed(tc.seed)
    device = resolve_device(tc.device)
    os.makedirs(CKPT_DIR, exist_ok=True)

    if loaders is None:
        tr_ld, va_ld, te_ld, (tr_names, va_names, te_names) = make_tvt_loaders(
            dcfg, wcfg, n_splits=tc.n_splits, fold=tc.fold,
            batch_size=tc.batch_size, num_workers=tc.num_workers)
    elif len(loaders) == 4:
        tr_ld, va_ld, te_ld, (tr_names, va_names, te_names) = loaders
    else:  # legacy 2-way (smoke): no separate test
        tr_ld, va_ld, (tr_names, va_names) = loaders
        te_ld, te_names = va_ld, va_names

    pos_w = tr_ld.dataset.pos_weight()  # type: ignore[attr-defined]
    model = build_model(model_config(tc, wcfg)).to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    steps_per_epoch = max(1, len(tr_ld))
    sched = (build_scheduler(opt, tc.epochs * steps_per_epoch, tc.warmup_frac,
                             tc.min_lr_frac) if tc.use_scheduler else None)

    if verbose:
        print(f"[{tc.model}/{tc.pool}] params={n_params}  pos_weight={pos_w:.2f}  "
              f"train={len(tr_names)} clips / {len(tr_ld.dataset)} win  "
              f"val={len(va_names)} clips / {len(va_ld.dataset)} win  "
              f"test={len(te_names)} clips / {len(te_ld.dataset)} win  dev={device}")

    _tag = f"_{tc.ckpt_tag}" if tc.ckpt_tag else ""
    ckpt = os.path.join(CKPT_DIR, f"{tc.model}_{tc.pool}{_tag}_fold{tc.fold}.pt")
    best_f1, best_epoch, best_eval = -1.0, -1, None
    history = []

    for epoch in range(1, tc.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for batch in tr_ld:
            x = batch["x"].to(device)
            valid = batch["valid"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad()
            logit = model(x, valid).squeeze(1)
            loss = loss_fn(logit, y)
            loss.backward()
            if tc.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
            opt.step()
            if sched is not None:
                sched.step()
            run_loss += float(loss.item())
            nb += 1

        ev = evaluate(model, va_ld, wcfg, device)   # F1@4 at the decode threshold
        f1 = ev["f1_primary"]
        history.append({"epoch": epoch, "loss": run_loss / max(nb, 1),
                        "f1@4": f1, "thr": ev["threshold"]})
        if verbose:
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  e{epoch:02d}  loss={run_loss / max(nb, 1):.4f}  "
                  f"F1@4={f1:.3f} (thr={ev['threshold']:.2f})  "
                  f"lr={cur_lr:.2e}  {time.time() - t0:.1f}s")

        if f1 > best_f1:
            best_f1, best_epoch, best_eval = f1, epoch, ev
            torch.save({"state_dict": model.state_dict(),
                        "cfg": asdict(tc), "epoch": epoch,
                        "f1_primary": f1, "threshold": ev["threshold"]}, ckpt)
        elif epoch - best_epoch >= tc.patience:
            if verbose:
                print(f"  early-stop @ e{epoch} (best F1@4={best_f1:.3f} @e{best_epoch})")
            break

    # ---- final report on the UNTOUCHED test split, using the val-selected model
    test_eval = None
    if eval_test and best_eval is not None and te_ld is not va_ld:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["state_dict"])
        thr = float(state["threshold"])              # decode threshold (0.60)
        scores, t0s, gts, nframes = collect_logits(model, te_ld, device)
        preds = [M.decode_peaks(s, t, wcfg.test, thr, from_logits=True,
                                min_distance=wcfg.min_distance)
                 for s, t in zip(scores, t0s)]
        rows = M.score_clips(preds, gts, M.DEFAULT_TOLS)
        test_eval = {"threshold": thr, "rows": rows,
                     "f1_primary": rows[M.PRIMARY_TOL].f1,
                     "bpm_mae": M.bpm_mae(preds, gts, nframes, 10.0)}

    if verbose and best_eval is not None:
        print(f"\nBEST {tc.model}/{tc.pool} fold{tc.fold}: "
              f"VAL F1@4={best_f1:.3f} @e{best_epoch}  (ckpt: {ckpt})")
        print("[val]  " + M.format_scores(best_eval["rows"]))
        print(f"[val]  BPM MAE: {best_eval['bpm_mae']:.3f}")
        if test_eval is not None:
            print(f"[test] F1@4={test_eval['f1_primary']:.3f} (thr={test_eval['threshold']:.2f})")
            print("[test] " + M.format_scores(test_eval["rows"]))
            print(f"[test] BPM MAE: {test_eval['bpm_mae']:.3f}")

    return {"best_f1": best_f1, "best_epoch": best_epoch, "best_eval": best_eval,
            "test_eval": test_eval,
            "test_f1": (test_eval["f1_primary"] if test_eval else None),
            "history": history, "n_params": n_params, "ckpt": ckpt}


# This module is a library: nested.py drives the pipeline.
