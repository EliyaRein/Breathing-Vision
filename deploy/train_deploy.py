"""Train the DEPLOYMENT model on ALL clips and freeze it to deploy/breath_gru.pt.

This is the single production model behind the streaming app. It reproduces the
winning nested-CV recipe (validated leak-free at F1@4=0.705, native F1@3=0.604):

    encoder   : GRU
    window    : context=80, past=63, test=7, future=10, stride=7, tol=0
    norm      : robust_window_noctr  (per-80-frame-context IQR only, no centring
                -> velocity-faithful, causal/streaming)
    decode    : per-window threshold 0.60, peak at window centre, min_distance=0

Unlike the CV folds there is NO held-out split here: the app ships one model, so
we spend every subject on the weights. Threshold and epoch budget come straight
from the nested run's val-based choices. For HP, the inner Optuna converged to 3
distinct points across the 6 folds; we take the moderate-lr point (see HP below)
because all-data training here has no val split to early-stop on, which makes the
aggressive-lr points riskier. Epoch budget is a fixed number (no early-stop
possible without a val split), sized to the folds' observed convergence.

The saved .pt is self-describing: it carries the model cfg, window cfg, norm
mode, decode threshold and fps, so deploy/pipeline.py rebuilds an identical model
and preprocessing without hard-coding anything.

Run:
    python -m deploy.train_deploy                 # defaults (recommended)
    python -m deploy.train_deploy --epochs 22
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import ModelConfig, build_model, count_params
from trainer import metrics as M
from trainer.config import DataConfig, WindowConfig
from trainer.dataset import ClipWindowDataset, list_clips
from trainer.harness import build_scheduler, resolve_device, set_seed

OUT_PATH = os.path.join(os.path.dirname(__file__), "breath_gru.pt")

WCFG = WindowConfig()            # (context=80, past=63, test=7, future=10, stride=7, tol=0)
NORM = "robust_window_noctr"     # IQR only, no centring (velocity-faithful; median~0)
FPS = 10.0

# HP: the inner-Optuna search produced 3 distinct points across the 6 pw folds.
# We take the moderate-lr point (folds 4 & 5) rather than the aggressive-lr ones
# (lr~7e-3, folds 0/1/2/3): all-data training has no val split to early-stop on,
# so the moderate lr=3.2e-4 is the safe choice under a fixed epoch budget.
# The pick is low-stakes: inner-trial val-F1 differences were small (per-fold
# std ~0.02-0.06) and convergence (patience 5) landed around epoch 10.
HP = {"lr": 3.198118496726709e-04, "weight_decay": 7.793746250885181e-05,
      "dropout": 0.21879360563134626, "warmup_frac": 0.2675319002346239}

# Decode operating point: the same threshold used in the nested evaluation
# (trainer.harness.DECODE_THRESHOLD). Kept as a literal here so the deployment
# recipe is self-contained.
DECODE_THR = 0.60


def make_model_cfg() -> ModelConfig:
    return ModelConfig(name="gru", pool="attention", dropout=HP["dropout"],
                       readout_start=WCFG.past, readout_len=WCFG.test)


def train(epochs: int, batch_size: int, device_name: str, seed: int) -> None:
    set_seed(seed)
    device = resolve_device(device_name)
    dcfg = DataConfig(norm=NORM)

    names = list_clips(dcfg.root)
    print(f"[deploy] training on ALL {len(names)} clips  dev={device}")
    t_build = time.time()
    ds = ClipWindowDataset(names, dcfg, wcfg=WCFG)     # per-window precompute here
    print(f"[deploy] dataset ready: {len(ds)} windows in {time.time()-t_build:.1f}s")
    ld = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    pos_w = ds.pos_weight()
    model = build_model(make_model_cfg()).to(device)
    print(f"[deploy] params={count_params(model)}  pos_weight={pos_w:.2f}")
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"],
                            weight_decay=HP["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    steps = max(1, len(ld))
    sched = build_scheduler(opt, epochs * steps, HP["warmup_frac"], 0.0)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        run, nb = 0.0, 0
        for batch in ld:
            x = batch["x"].to(device)
            valid = batch["valid"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad()
            logit = model(x, valid).squeeze(1)
            loss = loss_fn(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.item())
            nb += 1
        print(f"  e{epoch:02d}  loss={run/max(nb,1):.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.1f}s", flush=True)

    # quick TRAIN-set sanity decode (not a generalisation estimate -- just proves
    # the frozen model + threshold produce sane peaks before we ship it).
    _sanity_decode(model, ld, device)

    ckpt = {
        "state_dict": model.state_dict(),
        "model_cfg": vars(make_model_cfg()),
        "wcfg": vars(WCFG),
        "norm": NORM,
        "decode_threshold": DECODE_THR,
        "fps": FPS,
        "hp": HP,
        "n_clips": len(names),
        "epochs": epochs,
        "grid": 8,               # 8x8 = 64 cells (matches dataset_builder)
    }
    torch.save(ckpt, OUT_PATH)
    print(f"[deploy] saved -> {OUT_PATH}")


@torch.no_grad()
def _sanity_decode(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    model.eval()
    ds: ClipWindowDataset = loader.dataset  # type: ignore[assignment]
    n = len(ds.names)
    scores = [[] for _ in range(n)]
    t0s = [[] for _ in range(n)]
    for batch in loader:
        logit = model(batch["x"].to(device),
                      batch["valid"].to(device)).squeeze(1).float().cpu().numpy()
        for s, ci, t0 in zip(logit, batch["clip"].numpy(), batch["t0"].numpy()):
            scores[ci].append(float(s)); t0s[ci].append(int(t0))
    gts = [np.flatnonzero(ds._labels[c]).astype(np.int64) for c in range(n)]
    preds = [M.decode_peaks(s, t, WCFG.test, DECODE_THR, from_logits=True,
                            min_distance=WCFG.min_distance)
             for s, t in zip(scores, t0s)]
    rows = M.score_clips(preds, gts, [M.PRIMARY_TOL])
    print(f"[deploy] TRAIN-set sanity F1@4={rows[M.PRIMARY_TOL].f1:.3f} "
          f"(memorisation upper-bound, not a test score)")


def main() -> None:
    try:
        import sys
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20,
                    help="fixed epoch budget (no val split to early-stop on)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.epochs, args.batch_size, args.device, args.seed)


if __name__ == "__main__":
    main()
