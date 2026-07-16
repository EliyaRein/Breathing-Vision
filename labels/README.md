# Curated breath labels

Our **final, hand-reviewed** breath-event labels for the AIR-400 clips used in
this project. Raw AIR-400 respiration signals are noisy and their timing
sometimes drifts; we corrected the derived labels by hand (velocity-peak /
zero-crossing convention, fixed a per-subject compression bug, etc.) with the
interactive editor. These files are the ground truth behind every result in the
report, so we ship them for full reproducibility.

## Files

| File | What |
|---|---|
| `breath_labels_10fps.npz` | All **381 clips** as arrays keyed by clip name. Each array is a per-frame binary vector on the **10-fps motion timeline**: `1` = a labelled breath (exhale peak) at that frame, `0` otherwise (`uint8`). |
| `manual_label_edits.csv` | Global provenance log of every manual add/remove edit. |
| `labels_manifest.json` | Summary: clip count, which clips were manually edited, totals. |

- 381 clips · 198 manually edited · 8 976 breath labels total.

## Usage

```python
import numpy as np

d = np.load("labels/breath_labels_10fps.npz")
print(len(d.files), "clips")               # 381
y = d["AIR125_S01_001"]                     # per-frame 0/1 vector, 10 fps
breath_frames = np.flatnonzero(y)           # frame indices of labelled breaths
```

The timeline matches the motion tensors produced by `dataset_builder` (same
first-real-frame seek and 10-fps sampling), so once you rebuild the dataset from
AIR-400 these labels align frame-for-frame with each clip's motion.

## Source & credit

These labels are **derived from** the AIR-400 respiration annotations of Song,
Bishnoi et al., *"Overcoming Small Data Limitations in Video-Based Infant
Respiration Estimation"* (WACV 2026) — <https://github.com/michaelwwan/air-400>
(MIT License). The hand-review and corrections above are our own contribution.
