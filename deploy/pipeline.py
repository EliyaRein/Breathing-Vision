"""Streaming inference pipeline for the deployment app.

Turns a live sequence of 10-fps frames (already cropped-region-aware) into breath
events, reproducing EXACTLY the training-time preprocessing so the frozen model
sees the same distribution it was trained on:

    frame ->  incremental Lucas-Kanade optical flow on the ROI crop
          ->  per-cell mean displacement  (8x8 = 64 cells, dx/dy)   [2, 64]
          ->  rolling 80-frame context buffer
          ->  per-window robust norm  (IQR only, no centring)       [64, 2, 80]
          ->  GRU model  ->  logit  ->  sigmoid  ->  prob
          ->  threshold 0.60  ->  breath at the test-window centre

Causality / latency: scoring the test window that starts at frame t0 needs the
context [t0-63, t0+17), i.e. up to frame t0+16. The breath is reported at the
window CENTRE t0+3, so an event is emitted ~13 frames (~1.3 s) after it occurs --
this is the deliberate `future=10` peek, the only non-causal part (accepted).

Two classes:
    StreamingMotion    -- seed once, track per frame, emit [2,64]+valid[64].
    StreamingPipeline  -- buffers motion, normalises per window, runs the model,
                          emits breath-centre frame indices as they are decoded.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from dataset_builder.config import BuildConfig, CH_DX, CH_DY
from dataset_builder.motion import _seed_points
from models import ModelConfig, build_model
from trainer.config import WindowConfig
from trainer.dataset import _robust_stats, _apply_norm

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "breath_gru.pt")


# --------------------------------------------------------------------------- #
# incremental optical flow  (mirrors dataset_builder/motion.extract_motion)
# --------------------------------------------------------------------------- #
class StreamingMotion:
    """Seed feature points once inside the ROI, then track them frame-by-frame
    with sparse LK, emitting an 8x8 cell-mean (dx, dy) + valid mask per frame.

    Faithful to the offline builder: points are seeded ONCE (on the first frame)
    and only tracked afterwards; a point that loses tracking stays lost (its cell
    goes invalid once it has no live points), exactly as in training.
    """

    def __init__(self, box: Tuple[int, int, int, int], cfg: Optional[BuildConfig] = None):
        self.cfg = cfg or BuildConfig()
        self.box = tuple(int(v) for v in box)
        self.n_cells = self.cfg.n_cells
        x1, y1, x2, y2 = self.box
        self._crop_box = (0, 0, x2 - x1, y2 - y1)
        self._lk = dict(
            winSize=self.cfg.lk_win, maxLevel=self.cfg.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None      # [P,1,2] crop coords
        self._cell_id: Optional[np.ndarray] = None       # [P]
        self.seeded = False

    def _gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    def push(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (motion[2,64], valid[64]) for this frame."""
        gray = self._gray(frame_bgr)
        motion = np.zeros((2, self.n_cells), np.float32)
        valid = np.zeros(self.n_cells, np.float32)

        if not self.seeded:
            pts, cell_id = _seed_points(gray, self._crop_box, self.cfg)
            if pts is None:
                # no trackable features yet: emit empty, retry seeding next frame
                self._prev_gray = gray
                return motion, valid
            self._prev_pts, self._cell_id = pts, cell_id
            self._prev_gray = gray
            self.seeded = True
            # first frame: seeded, no displacement -> valid cells get 1, motion 0
            for c in np.unique(cell_id):
                valid[c] = 1.0
            return motion, valid

        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk)
        st = st.reshape(-1).astype(bool)
        disp = (new_pts - self._prev_pts).reshape(-1, 2)      # [P,2]
        self._prev_pts[st] = new_pts[st]                      # advance tracked only
        self._prev_gray = gray

        # cell-mean over the cells' valid (tracked-this-frame) points
        cid = self._cell_id
        for c in range(self.n_cells):
            m = np.flatnonzero((cid == c) & st)
            if m.size == 0:
                continue
            motion[CH_DX, c] = disp[m, 0].mean()
            motion[CH_DY, c] = disp[m, 1].mean()
            valid[c] = 1.0
        return motion, valid


# --------------------------------------------------------------------------- #
# full streaming pipeline
# --------------------------------------------------------------------------- #
class StreamingPipeline:
    def __init__(self, box, ckpt_path: str = DEFAULT_CKPT, device: str = "auto"):
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        ck = torch.load(ckpt_path, map_location=self.device)
        self.wcfg = WindowConfig(**ck["wcfg"])
        self.norm = ck["norm"]                      # "robust_window_noctr"
        self.center = self.norm != "robust_window_noctr"
        self.thr = float(ck["decode_threshold"])
        self.fps = float(ck["fps"])
        self.eps = 1e-6
        self.model = build_model(ModelConfig(**ck["model_cfg"])).to(self.device).eval()
        self.model.load_state_dict(ck["state_dict"])

        self.motion_src = StreamingMotion(box)
        self._motion: List[np.ndarray] = []         # per frame [2,64]
        self._valid: List[np.ndarray] = []          # per frame [64]
        self.n = 0                                  # frames pushed
        # Score from the very first window (t0=0). Early windows have a
        # zero-padded left context, exactly as during training (which also emitted
        # windows at t0=0,stride,...), so analysing the first seconds is faithful
        # to the trained distribution; the padded region is masked in the norm/pool.
        self._next_t0 = 0
        # decode latency (frames): a breath at centre c is only decidable once the
        # window that contains it has its future context, at n = t0+test+future.
        # centre = t0 + test//2  ->  latency = test + future - test//2.
        self.latency = self.wcfg.test + self.wcfg.future - self.wcfg.test // 2
        self.breaths: List[int] = []                # decoded breath-centre frames
        self.last_prob: float = 0.0

    # -- streaming API -------------------------------------------------------
    def push(self, frame_bgr: np.ndarray) -> List[int]:
        """Feed one frame; return the (possibly empty) list of NEW breath-centre
        frame indices decoded as a result."""
        m, v = self.motion_src.push(frame_bgr)
        self._motion.append(m)
        self._valid.append(v)
        self.n += 1

        new: List[int] = []
        w = self.wcfg
        # a window at t0 needs frames up to t0 + test + future - 1
        while self._next_t0 + w.test + w.future <= self.n:
            prob = self._score_window(self._next_t0)
            self.last_prob = prob
            if prob > self.thr:
                centre = self._next_t0 + w.test // 2
                self.breaths.append(centre)
                new.append(centre)
            self._next_t0 += w.stride
        return new

    # -- internals -----------------------------------------------------------
    def _slice(self, seq: List[np.ndarray], g0: int, g1: int) -> np.ndarray:
        """Stack seq[g0:g1] along a new last axis with zero-padding past edges.
        seq elements are [2,64] or [64]; returns [...,(g1-g0)]."""
        width = g1 - g0
        shape = seq[0].shape + (width,)
        out = np.zeros(shape, np.float32)
        s0, s1 = max(0, g0), min(self.n, g1)
        for k in range(s0, s1):
            out[..., k - g0] = seq[k]
        return out

    @torch.no_grad()
    def _score_window(self, t0: int) -> float:
        w = self.wcfg
        g0 = t0 - w.past
        g1 = g0 + w.context
        motion = self._slice(self._motion, g0, g1)      # [2,64,80]
        valid = self._slice(self._valid, g0, g1)        # [64,80]
        vmask = valid > 0

        med, scale = _robust_stats(motion, vmask, self.eps)
        if not self.center:
            med = np.zeros_like(med)
        motion = _apply_norm(motion, med, scale, vmask)  # [2,64,80]

        x = np.transpose(motion, (1, 0, 2))[None]        # [1,64,2,80]
        vx = valid[None]                                 # [1,64,80]
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        vt = torch.from_numpy(np.ascontiguousarray(vx)).to(self.device)
        logit = self.model(xt, vt).squeeze().float().item()
        return 1.0 / (1.0 + np.exp(-logit))

    # -- summary -------------------------------------------------------------
    def current_bpm(self) -> float:
        """Instantaneous rate from the mean inter-breath gap so far (bpm)."""
        b = np.asarray(self.breaths, dtype=np.float64)
        if b.size >= 2:
            gap = np.mean(np.diff(np.sort(b)))
            return 60.0 * self.fps / gap if gap > 0 else 0.0
        if self.n > 0:
            return 60.0 * self.fps * b.size / self.n
        return 0.0

    def bpm_upto(self, d: int) -> float:
        """Rate using only breaths whose centre <= frame d (for the delayed
        display, so the on-screen BPM matches the frame actually shown)."""
        b = np.asarray([x for x in self.breaths if x <= d], dtype=np.float64)
        if b.size >= 2:
            gap = np.mean(np.diff(np.sort(b)))
            return 60.0 * self.fps / gap if gap > 0 else 0.0
        if d >= 0:
            return 60.0 * self.fps * b.size / (d + 1)
        return 0.0

    def summary(self) -> dict:
        return {"n_frames": self.n, "duration_s": self.n / self.fps,
                "n_breaths": len(self.breaths), "bpm": self.current_bpm(),
                "breath_frames": list(self.breaths)}
