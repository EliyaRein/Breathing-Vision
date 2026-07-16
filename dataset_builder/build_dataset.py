"""Dataset generator for AIR-400.

Usage (run from the Breathing-Vision folder):
    python -m dataset_builder.build_dataset --input "C:\\...\\AIR_400" \
        --output "C:\\...\\dataset_out" --sets AIR_125 --subjects S01

Input layout expected:
    <input>/<set>/<subject>/NNN.mp4  + NNN.hdf5   (same stem)

Two dataset variants are written side by side under <output>, both derived from
a single optical-flow pass so they stay consistent:

  dataset_raw/<name>/   (lossless substrate; not fed to the model directly)
      tracks.npz        - tracks[P,2,T], valid[P,T], init_xy[P,2], cell_id[P]
      labels_10fps.npy, roi.json, meta.json, qc_roi.png

  dataset_used/<name>/  (what we feed: 8x8 cell-mean, no position coords)
      motion_cellmean.npz - tensor[3,64,T]
      labels_10fps.npy, roi.json, meta.json, qc_roi.png, motion_full.xlsx

Plus a top-level summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import traceback
from typing import List, Optional

import cv2
import numpy as np

from .config import BuildConfig, DEFAULT
from .roi import detect_roi
from .motion import extract_motion
from .labels import load_hdf5, derive_impulse, align_labels, LabelData
from .qc import save_roi_image, save_motion_excel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Our hand-reviewed labels shipped in the repo (see labels/README.md).
CURATED_LABELS = os.path.join(_REPO_ROOT, "labels", "breath_labels_10fps.npz")

SUBJECT_RE = re.compile(r"^S\d+$")

# Minimum resampled length for a clip to be usable. Windowing is on-the-fly and
# pads short contexts, so this only rejects degenerate/near-empty clips.
MIN_FRAMES = 40

SUMMARY_FIELDS = [
    "name", "set", "subject", "video", "status",
    "orig_fps", "n_frames", "first_frame_idx", "t_out",
    "roi_source", "roi_conf", "valid_ratio", "n_points",
    "n_peaks_orig", "n_peaks_aligned", "labels_source", "error",
]


def discover_videos(input_root: str, sets: Optional[List[str]],
                    subjects: Optional[List[str]]):
    """Yield (set_name, subject, video_stem, mp4_path, hdf5_path)."""
    set_names = sets if sets else [
        d for d in sorted(os.listdir(input_root))
        if os.path.isdir(os.path.join(input_root, d))
    ]
    for set_name in set_names:
        set_dir = os.path.join(input_root, set_name)
        if not os.path.isdir(set_dir):
            continue
        subj_dirs = [
            d for d in sorted(os.listdir(set_dir))
            if os.path.isdir(os.path.join(set_dir, d)) and SUBJECT_RE.match(d)
        ]
        if subjects:
            subj_dirs = [d for d in subj_dirs if d in subjects]
        for subject in subj_dirs:
            subj_dir = os.path.join(set_dir, subject)
            for fn in sorted(os.listdir(subj_dir)):
                if not fn.lower().endswith(".mp4"):
                    continue
                stem = os.path.splitext(fn)[0]
                mp4 = os.path.join(subj_dir, fn)
                # AIR_125 keeps the .hdf5 next to the .mp4; AIR_400 stores it under
                # a per-subject `out/` folder. Accept either layout.
                hdf5 = os.path.join(subj_dir, stem + ".hdf5")
                if not os.path.exists(hdf5):
                    hdf5 = os.path.join(subj_dir, "out", stem + ".hdf5")
                if os.path.exists(hdf5):
                    yield set_name, subject, stem, mp4, hdf5


def process_video(mp4_path, hdf5_path, raw_dir, used_dir, yolo_model, cfg: BuildConfig,
                  name: str = "", labels_mode: str = "derive", curated=None) -> dict:
    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps <= 0:
        raise RuntimeError("invalid fps")

    roi = detect_roi(mp4_path, yolo_model, cfg)

    motion = extract_motion(mp4_path, roi.box, roi.first_frame_idx, fps, n_frames, cfg)
    if motion.tensor.shape[2] < MIN_FRAMES:
        raise RuntimeError(f"too few frames after resample ({motion.tensor.shape[2]})")

    respiration, _impulse_file = load_hdf5(hdf5_path)
    # respiration may be sampled at a different rate than the video (e.g. 10 Hz
    # respiration vs 15 fps video in AIR175_S07); recover the clean sensor rate
    # for peak min-sep and pass n_frames so align_labels maps resp index -> frame.
    resp_fps = round(len(respiration) * fps / n_frames) if n_frames else fps
    impulse = derive_impulse(respiration, resp_fps, cfg)
    label_data = align_labels(impulse, respiration, motion.selected_idx, fps, cfg,
                              n_frames=n_frames)

    # Label source: 'derive' keeps the AIR-400-derived labels computed above;
    # 'curated' swaps in our hand-reviewed vector. The respiration waveform (for
    # QC) is always the derived one. In curated mode we FAIL LOUDLY rather than
    # silently mixing in derived / artificially re-aligned labels, so a built
    # dataset is never a silent mix of sources.
    labels_source = "derived"
    if labels_mode == "curated":
        if curated is None or name not in curated:
            raise RuntimeError(
                f"--labels curated: no curated label vector for '{name}' "
                f"(use --labels derive to compute it from the AIR-400 hdf5)")
        t_out = motion.tensor.shape[2]
        cur = np.asarray(curated[name]).ravel()
        if len(cur) != t_out:
            raise RuntimeError(
                f"--labels curated: label length {len(cur)} != motion length "
                f"{t_out} for '{name}' (build/label mismatch -- do not force-align)")
        uniq = np.unique(cur)
        if not np.all(np.isin(uniq, (0, 1))):
            raise RuntimeError(
                f"--labels curated: labels for '{name}' are not binary (values "
                f"{uniq[:5]}...)")
        label_data = LabelData(cur.astype(np.float32), label_data.respiration,
                               label_data.n_peaks_orig, int(cur.sum()))
        labels_source = "curated"

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(used_dir, exist_ok=True)

    selected_idx = np.asarray(motion.selected_idx)
    box = np.asarray(roi.box)
    n_points = int(motion.tracks.shape[0])

    roi_meta = {"box": list(roi.box), "source": roi.source,
                "confidence": roi.confidence, "n_detections": roi.n_detections,
                "first_frame_idx": roi.first_frame_idx}

    meta = {
        "orig_fps": fps, "n_frames": n_frames,
        "first_frame_idx": roi.first_frame_idx,
        "t_out": int(motion.tensor.shape[2]),
        "roi_source": roi.source, "roi_conf": roi.confidence,
        "valid_ratio": motion.valid_ratio, "n_points": n_points,
        "n_peaks_orig": label_data.n_peaks_orig,
        "n_peaks_aligned": label_data.n_peaks_aligned,
        "labels_source": labels_source,
        "target_fps": cfg.target_fps,
        "grid": cfg.grid, "points_per_cell": cfg.points_per_cell,
    }

    # --- raw substrate: per-point tracks (lossless; not fed directly) ---------
    np.savez_compressed(
        os.path.join(raw_dir, "tracks.npz"),
        tracks=motion.tracks,            # [P, 2, T] raw dx/dy (px)
        valid=motion.valid,              # [P, T]
        init_xy=motion.init_xy,          # [P, 2] normalised to ROI [0,1]
        cell_id=motion.cell_id,          # [P] seed 8x8 cell index
        selected_idx=selected_idx,
        box=box,
    )
    np.save(os.path.join(raw_dir, "labels_10fps.npy"), label_data.labels)
    save_roi_image(os.path.join(raw_dir, "qc_roi.png"), roi.frame_bgr, roi.box,
                   motion.cells, motion.init_points, roi.source, roi.confidence)
    with open(os.path.join(raw_dir, "roi.json"), "w") as f:
        json.dump(roi_meta, f, indent=2)
    with open(os.path.join(raw_dir, "meta.json"), "w") as f:
        json.dump({**meta, "variant": "raw"}, f, indent=2)

    # --- used representation: 8x8 cell-mean, no position coords ---------------
    np.savez_compressed(
        os.path.join(used_dir, "motion_cellmean.npz"),
        tensor=motion.tensor,            # [3, 64, T] (dx, dy, valid)
        selected_idx=selected_idx,
        box=box,
    )
    np.save(os.path.join(used_dir, "labels_10fps.npy"), label_data.labels)
    save_roi_image(os.path.join(used_dir, "qc_roi.png"), roi.frame_bgr, roi.box,
                   motion.cells, motion.init_points, roi.source, roi.confidence)
    with open(os.path.join(used_dir, "roi.json"), "w") as f:
        json.dump(roi_meta, f, indent=2)
    with open(os.path.join(used_dir, "meta.json"), "w") as f:
        json.dump({**meta, "variant": "used"}, f, indent=2)

    # Excel is a convenience QC artifact (used variant only); never let a
    # locked/open file (Excel holding the handle) abort the video and lose the
    # real training tensors.
    try:
        save_motion_excel(os.path.join(used_dir, "motion_full.xlsx"), motion,
                          label_data, cfg.target_fps)
    except Exception as e:  # optional QC artifact: never fail a good clip over it
        print(f"   [warn] motion_full.xlsx not written ({e}) - skipped")

    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description="AIR-400 dataset builder")
    p.add_argument("--input", required=True, help="root folder (contains AIR_125 / AIR_175)")
    p.add_argument("--output", required=True, help="output folder for the dataset")
    p.add_argument("--sets", nargs="*", default=None, help="e.g. AIR_125 AIR_175")
    p.add_argument("--subjects", nargs="*", default=None, help="e.g. S01 S02")
    p.add_argument("--limit", type=int, default=0, help="max videos (0 = all)")
    p.add_argument("--overwrite", action="store_true", help="rebuild existing outputs")
    p.add_argument("--labels", choices=["curated", "derive"], default="curated",
                   help="'curated' (default) = use our hand-reviewed labels shipped "
                        "in labels/ (reproduces the report; STRICT -- a clip fails "
                        "if its label is missing/mismatched/non-binary, never "
                        "silently derived); 'derive' = compute labels from the "
                        "AIR-400 hdf5.")
    p.add_argument("--labels-file", default=CURATED_LABELS,
                   help="path to the curated .npz (default: labels/breath_labels_10fps.npz)")
    args = p.parse_args(argv)

    cfg = DEFAULT

    curated = None
    if args.labels == "curated":
        if not os.path.exists(args.labels_file):
            sys.exit(f"[error] --labels curated but the label file was not found:\n"
                     f"        {args.labels_file}\n"
                     f"        Pass --labels-file <path>, or use --labels derive to "
                     f"compute labels from the AIR-400 hdf5.")
        curated = dict(np.load(args.labels_file))
        print(f"[init] curated labels: {args.labels_file} ({len(curated)} clips)")

    from ultralytics import YOLO
    print(f"[init] loading YOLO: {cfg.yolo_weights}")
    yolo_model = YOLO(cfg.yolo_weights)

    raw_root = os.path.join(args.output, "dataset_raw")
    used_root = os.path.join(args.output, "dataset_used")
    os.makedirs(raw_root, exist_ok=True)
    os.makedirs(used_root, exist_ok=True)

    summary_path = os.path.join(args.output, "summary.csv")
    new_summary = not os.path.exists(summary_path)
    summary_f = open(summary_path, "a", newline="")
    writer = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDS)
    if new_summary:
        writer.writeheader()

    videos = list(discover_videos(args.input, args.sets, args.subjects))
    if args.limit:
        videos = videos[:args.limit]
    total = len(videos)
    print(f"[init] {total} videos to process")

    # Full curated build must cover EXACTLY the shipped label corpus: a silently
    # skipped clip (missing hdf5 / wrong folder layout) would train on < the full
    # set. Only enforce for an unfiltered curated run.
    if args.labels == "curated" and not (args.subjects or args.sets or args.limit):
        discovered = {f"{s.replace('_', '')}_{sub}_{st}"
                      for (s, sub, st, _m, _h) in videos}
        expected = set(curated.keys())
        missing, extra = expected - discovered, discovered - expected
        if missing or extra:
            summary_f.close()
            sys.exit(
                f"[error] full curated build does not match the label corpus "
                f"({len(expected)} expected, {len(discovered)} discovered):\n"
                f"        missing {len(missing)}: {sorted(missing)[:8]}\n"
                f"        extra   {len(extra)}: {sorted(extra)[:8]}\n"
                f"        Check the --input layout / hdf5 presence, or build a "
                f"--subjects subset / use --labels derive.")

    requested_src = "curated" if args.labels == "curated" else "derived"
    ok = fail = 0
    t0 = time.time()
    for i, (set_name, subject, stem, mp4, hdf5) in enumerate(videos, 1):
        name = f"{set_name.replace('_', '')}_{subject}_{stem}"
        raw_dir = os.path.join(raw_root, name)
        used_dir = os.path.join(used_root, name)
        done_marker = os.path.join(used_dir, "meta.json")
        if os.path.exists(done_marker) and not args.overwrite:
            try:
                with open(done_marker, encoding="utf-8") as fh:
                    prev_src = json.load(fh).get("labels_source")
            except Exception:
                prev_src = None
            # Only skip when the EXISTING clip already uses the requested label
            # source; otherwise rebuild it so a source switch actually takes effect.
            if prev_src == requested_src:
                print(f"[{i}/{total}] {name} -- skip (exists, labels={prev_src})")
                continue
            print(f"[{i}/{total}] {name} -- rebuild (labels {prev_src} -> {requested_src})")

        # About to (re)build this clip: clear any previous outputs first, so a
        # mid-build failure leaves NO stale files (which the trainer would happily
        # pick up as an old/different label source) rather than a silent mix.
        for d in (used_dir, raw_dir):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

        row = {k: "" for k in SUMMARY_FIELDS}
        row.update(name=name, set=set_name, subject=subject, video=stem)
        try:
            meta = process_video(mp4, hdf5, raw_dir, used_dir, yolo_model, cfg,
                                  name=name, labels_mode=args.labels, curated=curated)
            row.update(status="ok", **{k: meta[k] for k in (
                "orig_fps", "n_frames", "first_frame_idx", "t_out",
                "roi_source", "roi_conf", "valid_ratio", "n_points",
                "n_peaks_orig", "n_peaks_aligned", "labels_source")})
            ok += 1
            print(f"[{i}/{total}] {name} -- OK  roi={meta['roi_source']} "
                  f"labels={meta['labels_source']} "
                  f"peaks={meta['n_peaks_aligned']}/{meta['n_peaks_orig']} "
                  f"valid={meta['valid_ratio']:.2f}")
        except Exception as e:  # noqa: BLE001 - keep batch alive
            # remove any partial outputs so the trainer never picks up a clip that
            # failed mid-build (it only checks for the two tensor/label files).
            shutil.rmtree(used_dir, ignore_errors=True)
            shutil.rmtree(raw_dir, ignore_errors=True)
            row.update(status="FAILED", error=str(e))
            fail += 1
            print(f"[{i}/{total}] {name} -- FAILED: {e}")
            traceback.print_exc()
        writer.writerow(row)
        summary_f.flush()

    summary_f.close()
    dt = time.time() - t0
    print(f"\n[done] ok={ok} failed={fail} total={total} in {dt:.1f}s "
          f"({dt / max(1, ok + fail):.1f}s/video)")
    print(f"[done] summary: {summary_path}")
    if fail:
        print(f"[done] {fail} clip(s) FAILED -- exiting non-zero")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
