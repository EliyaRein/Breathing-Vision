"""Nested cross-validation -- ZERO test leakage, including HP tuning.

A flat scheme would tune ONE global HP set on a few folds' val, which reappear as
other folds' test -> a small but real HP leak. Here every outer test fold gets
its OWN inner Optuna that only ever sees that fold's train+val; the test group is
untouched until the single final measurement.

Per outer fold i (test = G_i, val = G_{i+1}, train = the rest):
  1. INNER TUNE: Optuna trains on train_i and scores on val_i only (test_i is
     never even built -- we pass a 2-tuple so train_fold sets te==va). Best HP is
     chosen on val_i.
  2. FINAL: retrain with that HP, early-stop on val_i, then decode test_i EXACTLY
     ONCE at the decode threshold (0.60). G_i never influenced training, selection,
     or HP.
Report = mean +/- std of the 6 outer test scores.

Cost ~ 6x a flat run, so keep the inner budget small. Loaders are built once per
fold and reused across all trials.

Run:
  python -m trainer.nested --trials 10               # all three encoders (default)
  python -m trainer.nested --model gru --trials 10   # a single encoder
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DataConfig, WindowConfig
from .dataset import make_tvt_loaders, ClipWindowDataset
from .harness import (TrainConfig, train_fold, CKPT_DIR, resolve_device, set_seed,
                      model_config, build_scheduler, collect_logits, DECODE_THRESHOLD)
from . import metrics as M
from models import build_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def suggest_hp(trial) -> Dict:
    """The 4 optimisation HPs tuned per-fold by Optuna (identical space + budget
    for the three encoders -> a fair best-vs-best comparison). Everything else in
    the pipeline is fixed. lr's upper bound is safe thanks to grad-clip + warmup +
    pre-LN transformer."""
    return {
        "lr": trial.suggest_float("lr", 3e-5, 8e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 3e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "warmup_frac": trial.suggest_float("warmup_frac", 0.0, 0.3),
    }


def refit_trainval_test(tc: TrainConfig, dcfg, wcfg, tr_names, va_names, te_ld,
                        epochs: int, threshold: float, device) -> Dict:
    """Refit on TRAIN+VAL united (~82%) for a set epoch budget, then decode TEST
    once at the decode `threshold` (0.60).

    This is the textbook nested-CV final step: the inner val already set the
    stopping point `epochs` WITHOUT the test. Now we spend all of (train+val) on the
    weights so the measured model sees the max data that still leaves G_i untouched.
    No early-stop here (no held-out left); epochs come from the val-selected best
    epoch.
    """
    set_seed(tc.seed)
    ds = ClipWindowDataset(list(tr_names) + list(va_names), dcfg, wcfg)
    ld = DataLoader(ds, batch_size=tc.batch_size, shuffle=True,
                    num_workers=tc.num_workers, drop_last=False)
    pos_w = ds.pos_weight()
    model = build_model(model_config(tc, wcfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    steps = max(1, len(ld))
    epochs = max(1, int(epochs))
    sched = (build_scheduler(opt, epochs * steps, tc.warmup_frac, tc.min_lr_frac)
             if tc.use_scheduler else None)

    model.train()
    for _ in range(epochs):
        for batch in ld:
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

    scores, t0s, gts, nframes, y_flat, s_flat = collect_logits(
        model, te_ld, device, with_labels=True)
    preds = [M.decode_peaks(s, t, wcfg.test, threshold, from_logits=True,
                            min_distance=wcfg.min_distance)
             for s, t in zip(scores, t0s)]
    rows = M.score_clips(preds, gts, M.DEFAULT_TOLS)
    # window-level confusion matrix at the decode threshold + breath counts
    p_flat = (1.0 / (1.0 + np.exp(-s_flat))) >= threshold
    yb = y_flat >= 0.5
    win_cm = {"tp": int(np.sum(p_flat & yb)), "tn": int(np.sum(~p_flat & ~yb)),
              "fp": int(np.sum(p_flat & ~yb)), "fn": int(np.sum(~p_flat & yb))}
    return {"threshold": threshold, "rows": rows,
            "f1_primary": rows[M.PRIMARY_TOL].f1,
            "bpm_mae": M.bpm_mae(preds, gts, nframes, 10.0),
            "refit_epochs": epochs,
            "win_cm": win_cm, "n_clips": len(gts),
            "pred_breaths": int(sum(len(p) for p in preds)),
            "gt_breaths": int(sum(len(g) for g in gts))}


def tune_one_fold(model, dcfg, wcfg, fold, *, trials, epochs, patience,
                  batch_size, n_splits, device, seed=0, final_epochs=30,
                  refit_trainval=True, ckpt_tag=None):
    """Optuna on (train_i, val_i) only; returns best HP + the final test eval."""
    import optuna
    ckpt_tag = ckpt_tag or f"nested_{model}"

    tr, va, te, names = make_tvt_loaders(dcfg, wcfg, n_splits=n_splits, fold=fold,
                                         batch_size=batch_size)
    tr_names, va_names, te_names = names
    tune_loaders = (tr, va, (tr_names, va_names))     # 2-tuple -> test NOT built

    def objective(trial):
        hp = suggest_hp(trial)
        tc = TrainConfig(model=model, fold=fold, epochs=epochs, patience=patience,
                         batch_size=batch_size, n_splits=n_splits, seed=seed,
                         device=device, **hp)
        res = train_fold(tc, dcfg, wcfg, loaders=tune_loaders, verbose=False)
        return res["best_f1"]                          # VAL_i only

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner())             # single inner split -> no pruning
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_hp = study.best_params

    # FINAL step 1: train on TRAIN, select best_epoch on VAL (test untouched); 
    # When we refit on train+val next (the default), TEST is decoded there EXACTLY ONCE,
    # so skip the redundant test decode here (eval_test=False). Only the non-refit path
    # needs train_fold's own test_eval.
    tc = TrainConfig(model=model, fold=fold, epochs=final_epochs, patience=6,
                     batch_size=batch_size, n_splits=n_splits, seed=seed,
                     device=device, ckpt_tag=ckpt_tag, **best_hp)
    res = train_fold(tc, dcfg, wcfg, loaders=(tr, va, te, names), verbose=False,
                     eval_test=not refit_trainval)

    # FINAL step 2 (default): refit on TRAIN+VAL united (~82%) at the val-selected
    # epoch (decode threshold 0.60), then decode TEST once. Maximises data the model sees
    # while G_i stays fully untouched (train_fold above never trained on test/val weights).
    if refit_trainval:
        dev = resolve_device(device)
        te_eval = refit_trainval_test(
            tc, dcfg, wcfg, tr_names, va_names, te,
            epochs=res["best_epoch"], threshold=DECODE_THRESHOLD,
            device=dev)
    else:
        te_eval = res["test_eval"]

    return {"fold": fold, "hp": best_hp, "val_f1": res["best_f1"],
            "threshold": te_eval["threshold"],   # decode threshold (0.60)
            "refit": "trainval" if refit_trainval else "train",
            "test_f1": te_eval["f1_primary"],
            "test_per_tol": {int(t): te_eval["rows"][t].f1 for t in M.DEFAULT_TOLS},
            "test_mae": te_eval["bpm_mae"],
            "win_cm": te_eval.get("win_cm"), "n_clips": te_eval.get("n_clips"),
            "pred_breaths": te_eval.get("pred_breaths"),
            "gt_breaths": te_eval.get("gt_breaths"),
            "test_subjects": sorted(set(
                "_".join(n.split("_")[:2]) for n in te_names))}


def nested_cv(model: str, *, trials=10, epochs=20, patience=5,
              batch_size=256, n_splits=6, device="auto", tag="", final_epochs=30,
              folds_subset=None, ckpt_tag=None) -> Dict:
    dcfg = DataConfig()
    wcfg = WindowConfig()
    runtag = tag or f"nested_{model}"
    outdir = os.path.join(RESULTS_DIR, runtag)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{model}.json")

    fold_ids = folds_subset if folds_subset is not None else list(range(n_splits))
    folds: List[Dict] = []
    for f in fold_ids:
        print(f"\n### nested {model} outer fold {f}/{n_splits-1} "
              f"(inner Optuna {trials} trials on train+val only) ###")
        r = tune_one_fold(model, dcfg, wcfg, f, trials=trials, epochs=epochs,
                          patience=patience, batch_size=batch_size,
                          n_splits=n_splits, device=device, final_epochs=final_epochs,
                          ckpt_tag=ckpt_tag)
        print(f"  fold{f}: test F1@4={r['test_f1']:.3f}  val={r['val_f1']:.3f}  "
              f"thr={r['threshold']:.2f}  hp={r['hp']}  test={r['test_subjects']}")
        cm = r.get("win_cm")
        if cm:
            nc = max(1, r.get("n_clips") or 1)
            print(f"         win/clip: TP={cm['tp']/nc:.1f} TN={cm['tn']/nc:.1f} "
                  f"FP={cm['fp']/nc:.1f} FN={cm['fn']/nc:.1f}  |  breaths/clip "
                  f"pred={r['pred_breaths']/nc:.1f} gt={r['gt_breaths']/nc:.1f}")
        folds.append(r)
        with open(out, "w") as fh:
            json.dump({"model": model, "folds": folds}, fh, indent=2)

    tf = np.array([x["test_f1"] for x in folds])
    per_tol = {t: np.array([x["test_per_tol"][t] for x in folds])
               for t in M.DEFAULT_TOLS}
    mae = np.array([x["test_mae"] for x in folds])
    print(f"\n########## NESTED {model} ({n_splits}-fold, leak-free) ##########")
    for t in M.DEFAULT_TOLS:
        print(f"TEST F1@{t}: {per_tol[t].mean():.3f} +/- {per_tol[t].std():.3f}")
    print(f"TEST BPM MAE: {mae.mean():.3f} +/- {mae.std():.3f}")
    print(f"decode threshold: {DECODE_THRESHOLD:.2f}")
    if all(x.get("win_cm") for x in folds):
        TP = sum(x["win_cm"]["tp"] for x in folds)
        TN = sum(x["win_cm"]["tn"] for x in folds)
        FP = sum(x["win_cm"]["fp"] for x in folds)
        FN = sum(x["win_cm"]["fn"] for x in folds)
        NC = max(1, sum(x["n_clips"] for x in folds))
        PB = sum(x["pred_breaths"] for x in folds)
        GB = sum(x["gt_breaths"] for x in folds)
        wp = TP / (TP + FP) if TP + FP else 0.0
        wr = TP / (TP + FN) if TP + FN else 0.0
        print(f"TEST win/clip (avg): TP={TP/NC:.1f} TN={TN/NC:.1f} "
              f"FP={FP/NC:.1f} FN={FN/NC:.1f}  (window P={wp:.3f} R={wr:.3f})")
        print(f"TEST breaths/clip (avg): detected={PB/NC:.1f}  actual={GB/NC:.1f}")
    print(f"saved -> {out}")
    return {"model": model,
            "test_f1_mean": float(tf.mean()), "test_f1_std": float(tf.std())}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="Nested CV of the frozen final pipeline; only the encoder "
                    "(--model) varies. Default 'all' runs the three in sequence.")
    ap.add_argument("--model", default="all",
                    choices=["all", "tcn", "gru", "transformer"],
                    help="encoder to evaluate; 'all' runs tcn+gru+transformer")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--final-epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="subset of outer folds (default all) -- for quick smokes")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt-tag", default=None,
                    help="checkpoint tag (default nested_<model>); set to avoid clobber")
    args = ap.parse_args()

    models = ["tcn", "gru", "transformer"] if args.model == "all" else [args.model]
    for m in models:
        # keep checkpoints distinct per encoder when a shared tag is given
        ckpt_tag = (f"{args.ckpt_tag}_{m}" if args.ckpt_tag and len(models) > 1
                    else args.ckpt_tag)
        nested_cv(m, trials=args.trials, epochs=args.epochs,
                  patience=args.patience, batch_size=args.batch_size,
                  n_splits=args.n_splits, device=args.device, tag=args.tag,
                  final_epochs=args.final_epochs, folds_subset=args.folds,
                  ckpt_tag=ckpt_tag)


if __name__ == "__main__":
    main()
