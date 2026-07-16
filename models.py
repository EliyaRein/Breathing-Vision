"""The three encoders under comparison: TCN, GRU, Transformer.

Architecture (early axis-level selection):

    input  (B, P0, 2, T) motion + (B, P0, T) valid   P0 = 64 cells
      -> split axes: instance = (cell, axis) -> P = 128 single-channel signals
         each instance = [signal, valid]                 [B*P, 2, T]
      -> SHARED dilated conv STEM (RF ~= a breath cycle,  [B*P, stem, T]
         so each axis-signal is turned into a half phase-
         canonical "reversal-ish" feature BEFORE selection)
      -> SHARED multi-slot SELECTION over the 128        [B, K*stem, T]
         instances: K content-dependent, non-negative
         (softmax) attention distributions, static per
         window. Collapses the INSTANCE axis (keeps time)
         into K fused channels -> one multivariate signal.
      -> per-signal temporal ENCODER (the only part that  [B, d, T]
         differs: TCN / GRU / Transformer). Runs ONCE per
         sample now (not x128), so it can be wider.
      -> SHARED readout at the DECISION region (the 7      [B, d]
         test frames) — NOT a blind mean over 80.
      -> SHARED linear head                               [B, 1] logit

Why this shape (design decisions):
  * axis-level instances: for a given cell one axis is often informative while
    the other is pure noise; splitting lets the selection drop the noisy axis
    (impossible when dx,dy are bundled through a fixed shared stem).
  * selection BEFORE the heavy encoder: most of the 128 signals are noise; a
    learned, aggressive, per-clip attention keeps the few informative ones. It
    is content-dependent (a signal wins by its content, not its index), so cell
    16's dx can win in one clip and cell 60's dy in another.
  * K slots (multi-slot): a single non-negative weighted sum can attenuate
    signals that are out of phase; K slots let different phase/region groups
    live in different fused channels. Exact anti-phase and arbitrary phase
    offsets are ultimately handled by the temporal encoder (which turns an
    oscillation into an event marker) + the loss.
  * readout at the test window: the label is about the 7 test frames only, so
    averaging the encoder output over all 80 context frames dilutes the signal
    ~11x and collapses training to a trivial prior. We read out at the decision
    region instead (the encoder has already seen the full context there).

The `valid` mask is used as (1) a per-instance input channel and (2) to mask
padded frames out of the score pooling and fully-empty instances out of the
softmax. No instance is dropped for being valid only part of the window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    name: str = "tcn"                 # tcn | gru | transformer
    in_ch: int = 2                    # per axis-instance: signal + valid
    stem_ch: int = 24                 # shared conv-stem width
    d: int = 32                       # encoder embedding dim (identical head)
    dropout: float = 0.1
    stem_norm: str = "group"          # shared conv-stem normalization
    stem_mode: str = "conv"           # dilated conv stem (RF~17)
    stem_rf: str = "short"            # receptive field ~17 frames (~one breath cycle)
    # --- selection ---
    n_slots: int = 4                  # K fused channels-groups
    att_dim: int = 32                 # attention hidden
    select_score: str = "meanstd"     # slot scorer sees time mean+std
    pool: str = "attention"           # kept for run-naming; selection is always multi-slot
    # --- readout (the test frames inside the 80-frame context) ---
    readout_start: int = 63           # = WindowConfig.past
    readout_len: int = 7              # = WindowConfig.test
    # --- TCN ---
    tcn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    tcn_kernel: int = 3
    # --- GRU ---
    gru_hidden: int = 50              # param-matched to TCN (~45K)
    gru_layers: int = 2
    # --- Transformer ---
    tf_layers: int = 2
    tf_heads: int = 4
    tf_ff: int = 224                  # param-matched to TCN (~45K)

    @property
    def enc_in(self) -> int:
        """Channels entering the temporal encoder = K fused signals x stem width."""
        return self.n_slots * self.stem_ch


# --------------------------------------------------------------------------- #
# shared frontend: stem + multi-slot selection
# --------------------------------------------------------------------------- #
class ConvStem(nn.Module):
    """Shared per-instance temporal featurizer.

    Dilated 1d stack with receptive field ~17 frames (@10fps ~= one breath
    cycle) so a raw oscillation starts to become an event-aligned feature
    BEFORE cross-instance selection — that is what lets a plain non-negative
    sum reinforce (rather than cancel) instances at arbitrary phase offsets.
    """

    def __init__(self, in_ch: int, ch: int, dropout: float, norm: str = "none",
                 mode: str = "conv", rf: str = "short"):
        super().__init__()

        def _norm():
            # GroupNorm is per-sample -> does NOT mix padded/empty instances across
            # the batch (safer than BatchNorm given our valid-mask); BatchNorm is
            # the classic tut08 option. Returns [] for 'none' so the module layout
            # is IDENTICAL to the original stem (old checkpoints stay loadable).
            if norm == "group":
                return [nn.GroupNorm(num_groups=min(4, ch), num_channels=ch)]
            if norm == "batch":
                return [nn.BatchNorm1d(ch)]
            return []

        if mode == "proj":
            # ABLATION: no temporal featurizer. A single pointwise (kernel=1, RF=1)
            # lift in_ch->ch, so the selector/encoder dims stay valid but the stem
            # contributes NO cross-frame processing -> isolates ConvStem's value.
            convs = [(nn.Conv1d(in_ch, ch, kernel_size=1), 1, 1)]
        else:
            # 'short' RF~17 (~one breath cycle). 'long' RF~33 adds a dilation-8 tap
            # (~two cycles) so the selector sees more oscillation context before it
            # scores each instance. padding = dilation*(k-1)/2 keeps length ('same').
            convs = [
                (nn.Conv1d(in_ch, ch, kernel_size=5, padding=2), 5, 1),
                (nn.Conv1d(ch, ch, kernel_size=3, padding=2, dilation=2), 3, 2),
                (nn.Conv1d(ch, ch, kernel_size=3, padding=4, dilation=4), 3, 4),
            ]
            if rf == "long":
                convs.append(
                    (nn.Conv1d(ch, ch, kernel_size=3, padding=8, dilation=8), 3, 8))
        layers = []
        for conv, _, _ in convs:
            layers += [conv, *_norm(), nn.GELU()]
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [BP, in, T] -> [BP, ch, T]
        return self.net(x)


class MultiSlotSelect(nn.Module):
    """Collapse the instance axis with K content-dependent attention slots.

    Static per-window: each instance gets ONE weight per slot for the whole
    window, computed from its time-pooled stem embedding (gated-attention,
    Ilse et al. 2018, extended to K heads). softmax is over the P instances, so
    weights are non-negative and select/average instances; K parallel slots keep
    room for distinct phase/region groups. Time is preserved.
    """

    def __init__(self, stem_ch: int, n_slots: int, att_dim: int,
                 score: str = "mean"):
        super().__init__()
        self.K = n_slots
        self.score = score
        # scorer sees time-MEAN (blind to oscillation), time-STD (oscillation
        # energy -> distinguishes a breathing instance from a flat/noisy one), or
        # both. A diffuse near-uniform selection on the mean-only variant was the
        # measured failure mode (see inspect_selection.py).
        in_dim = stem_ch * (2 if score == "meanstd" else 1)
        self.V = nn.Linear(in_dim, att_dim)
        self.U = nn.Linear(in_dim, att_dim)
        self.w = nn.Linear(att_dim, n_slots)

    def _pool(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        cnt = m.sum(-1).clamp(min=1.0)                          # [B,P,1]
        mean = (h * m).sum(-1) / cnt                            # [B,P,C]
        if self.score == "mean":
            return mean
        var = (((h - mean.unsqueeze(-1)) ** 2) * m).sum(-1) / cnt
        std = torch.sqrt(var.clamp(min=1e-8))                   # [B,P,C]
        return std if self.score == "std" else torch.cat([mean, std], dim=-1)

    def forward(self, h: torch.Tensor, tmask: torch.Tensor):
        # h [B, P, C, T]; tmask [B, P, T] (1 = real frame)
        neg = torch.finfo(h.dtype).min
        m = tmask.unsqueeze(2)                                   # [B,P,1,T]
        e = self._pool(h, m)                                     # [B,P,C or 2C]
        a = self.w(torch.tanh(self.V(e)) * torch.sigmoid(self.U(e)))   # [B,P,K]
        imask = (tmask.sum(-1) > 0).unsqueeze(-1)               # [B,P,1] instance has data
        a = a.masked_fill(~imask, neg)
        a = torch.softmax(a, dim=1)                             # over instances -> [B,P,K]
        fused = torch.einsum("bpk,bpct->bkct", a, h)           # [B,K,C,T]
        B, K, C, T = fused.shape
        return fused.reshape(B, K * C, T), a                    # [B,K*C,T], [B,P,K]


# --------------------------------------------------------------------------- #
# encoders (only these differ) -- all map [B, enc_in, T] -> [B, d, T]
# --------------------------------------------------------------------------- #
class _Chomp(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[..., :-self.n] if self.n > 0 else x


class _TemporalBlock(nn.Module):
    """Dilated causal residual block (Bai et al. 2018)."""

    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.chomp = _Chomp(pad)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        y = self.drop(self.act(self.chomp(self.conv1(x))))
        y = self.drop(self.act(self.chomp(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return self.act(y + res)


class TCNEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        blocks, c_in = [], cfg.enc_in
        for dil in cfg.tcn_dilations:
            blocks.append(_TemporalBlock(c_in, cfg.d, cfg.tcn_kernel, dil, cfg.dropout))
            c_in = cfg.d
        self.net = nn.Sequential(*blocks)

    def forward(self, x):                 # [B, enc_in, T] -> [B, d, T]
        return self.net(x)


class GRUEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gru = nn.GRU(cfg.enc_in, cfg.gru_hidden, cfg.gru_layers,
                          batch_first=True, dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0)
        self.proj = nn.Linear(cfg.gru_hidden, cfg.d)

    def forward(self, x):                 # [B, enc_in, T] -> [B, d, T]
        y, _ = self.gru(x.transpose(1, 2))       # [B, T, hidden]
        y = self.proj(y)                          # [B, T, d]
        return y.transpose(1, 2)                  # [B, d, T]


class _PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d]

    def forward(self, x):                 # [B, T, d]
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.inproj = nn.Linear(cfg.enc_in, cfg.d)
        self.posenc = _PositionalEncoding(cfg.d)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d, nhead=cfg.tf_heads, dim_feedforward=cfg.tf_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
            norm_first=True)   # pre-LN: stable at init (post-LN NaNs w/o warmup)
        self.tf = nn.TransformerEncoder(layer, num_layers=cfg.tf_layers)

    def forward(self, x):                 # [B, enc_in, T] -> [B, d, T]
        y = self.posenc(self.inproj(x.transpose(1, 2)))    # [B, T, d]
        y = self.tf(y)
        return y.transpose(1, 2)                            # [B, d, T]


# --------------------------------------------------------------------------- #
# full model
# --------------------------------------------------------------------------- #
_ENCODERS = {"tcn": TCNEncoder, "gru": GRUEncoder, "transformer": TransformerEncoder}


class MILModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = ConvStem(cfg.in_ch, cfg.stem_ch, cfg.dropout, cfg.stem_norm,
                             cfg.stem_mode, cfg.stem_rf)
        self.select = MultiSlotSelect(cfg.stem_ch, cfg.n_slots, cfg.att_dim,
                                      cfg.select_score)
        self.encoder = _ENCODERS[cfg.name](cfg)
        self.head = nn.Linear(cfg.d, 1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                return_attn: bool = False):
        # x [B, P0, 2, T] (P0=64 cells); valid [B, P0, T]
        B, P0, _, T = x.shape
        P = P0 * 2                                             # axis-level instances

        # split axes -> single-channel instances, carry per-instance valid
        sig = x.reshape(B, P, 1, T)                            # (c0dx, c0dy, c1dx, ...)
        vmask = valid.repeat_interleave(2, dim=1)             # [B, P, T]
        inst = torch.cat([sig, vmask.unsqueeze(2)], dim=2)    # [B, P, 2, T]

        h = self.stem(inst.reshape(B * P, self.cfg.in_ch, T))  # [BP, stem, T]
        h = h.reshape(B, P, self.cfg.stem_ch, T)               # [B, P, stem, T]

        fused, attn = self.select(h, vmask)                    # [B, K*stem, T], [B,P,K]
        seq = self.encoder(fused)                              # [B, d, T]

        s = self.cfg.readout_start
        emb = seq[..., s:s + self.cfg.readout_len].mean(dim=-1)  # [B, d] decision region
        logit = self.head(emb)                                  # [B, 1]
        return (logit, attn) if return_attn else logit


def build_model(cfg: ModelConfig) -> MILModel:
    return MILModel(cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
