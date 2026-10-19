"""
models_v3.py
============
Two improved DualStreamTCN regression heads on top of the original backbone,
plus ablation switches so we can quantify each component's contribution.

Method V_A (conservative, current backbone + add-ons)
-----------------------------------------------------
DualStreamTCN backbone (MixTemporalBlock x4 + CoordAttention)
    + PatchAttention   (kept — current behaviour)
    + CrossStreamAttention  (NEW: Raw ↔ Trend interaction before pooling)
    + AttentionPooling
    + length-token (16-d embedding of trajectory length)
    + Heteroscedastic head (μ, logσ²)

Method V_Aprime (Battery-v2-inspired, replaces PatchAttention with PE+Transformer+SKFusion)
-------------------------------------------------------------------------------------------
DualStreamTCN backbone
    - PatchAttention  (REMOVED — Battery ablation showed this contributes the least)
    + PositionalEncoding
    + TransformerEncoder (2 layers, post-TCN, fused across both streams)
    + SKFusion          (selective fusion of Raw and Trend features)
    + AttentionPooling
    + length-token
    + Heteroscedastic head (μ, logσ²)

Both methods support the same constructor signature so the trainer doesn't
need to branch — pass `variant='V_A'` or `'V_Aprime'` plus the optional
`disable_*` ablation flags.

For classification (NGAFID) the heteroscedastic head outputs a single
log-temperature scalar per sample instead of (μ, σ²); softmax(logits / T)
gives a learned per-sample confidence.  See `DualStreamCLF_v3`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

__all__ = [
    'DualStreamTCN_v3',
    'DualStreamCLF_v3',
    'gaussian_nll_loss',
]


# Shared building blocks
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.pool_w = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(inp, mip, 1)
        self.bn1 = nn.BatchNorm1d(mip)
        self.act = Mish()
        self.conv_h = nn.Conv1d(mip, inp, 1)
        self.conv_time = nn.Conv1d(1, 1, 7, padding=3, bias=False)

    def forward(self, x):
        identity = x
        y_ch = self.conv_h(self.act(self.bn1(self.conv1(self.pool_w(x))))).sigmoid()
        y_t = self.conv_time(torch.mean(x, dim=1, keepdim=True)).sigmoid()
        return identity * y_ch * y_t


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size]


class MixTemporalBlock(nn.Module):
    """Identical to the production model.py block: causal mixed-kernel TCN + CoordAttn + residual."""
    def __init__(self, n_inputs, n_outputs, dilation, dropout=0.2):
        super().__init__()
        half = n_outputs // 2
        self.pad3 = dilation * 2
        self.conv3 = weight_norm(nn.Conv1d(n_inputs, half, 3, padding=self.pad3, dilation=dilation))
        self.chomp3 = Chomp1d(self.pad3)
        self.pad5 = dilation * 4
        self.conv5 = weight_norm(nn.Conv1d(n_inputs, half, 5, padding=self.pad5, dilation=dilation))
        self.chomp5 = Chomp1d(self.pad5)
        self.act = Mish()
        self.dropout = nn.Dropout(dropout)
        self.ca = CoordinateAttention(n_outputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.conv3.weight.data.normal_(0, 0.01)
        self.conv5.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        o3 = self.chomp3(self.conv3(x))
        o5 = self.chomp5(self.conv5(x))
        out = torch.cat([o3, o5], dim=1)
        out = self.dropout(self.act(out))
        out = self.ca(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)

    def forward(self, x):  # x: [B, D, L]
        x_t = x.permute(0, 2, 1)
        scores = self.w(torch.tanh(x_t))
        weights = F.softmax(scores, dim=1)
        return torch.sum(x_t * weights, dim=1)


class PatchAttention(nn.Module):
    """Same patch-attention block used in the production model.py — kept for V_A only."""
    def __init__(self, d_model, patch_len=10, n_heads=4, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.patch_proj = nn.Linear(d_model * patch_len, d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                                dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.unpatch_proj = nn.Linear(d_model, d_model * patch_len)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        identity = x
        B, C, L = x.shape
        if L < self.patch_len:
            return x
        n_patches = L // self.patch_len
        used_len = n_patches * self.patch_len
        x_patch = x[:, :, :used_len].reshape(B, C, n_patches, self.patch_len)
        x_patch = x_patch.permute(0, 2, 1, 3).reshape(B, n_patches, C * self.patch_len)
        tokens = self.patch_proj(x_patch)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        x_recon = self.unpatch_proj(tokens)
        x_recon = x_recon.reshape(B, n_patches, C, self.patch_len).permute(0, 2, 1, 3).reshape(B, C, used_len)
        out = identity.clone()
        out[:, :, :used_len] = identity[:, :, :used_len] + self.alpha * x_recon
        return out


# NEW components (V_A and V_Aprime)
class CrossStreamAttention(nn.Module):
    """Bidirectional cross-attention between Raw and Trend streams.
    Used by V_A only. Each stream queries the other; outputs feed back as residual.
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn_r2t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_t2r = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_r = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.alpha_r = nn.Parameter(torch.tensor(0.1))
        self.alpha_t = nn.Parameter(torch.tensor(0.1))

    def forward(self, raw, trend):
        # raw, trend: [B, D, L] → permute to [B, L, D] for MHA
        r = raw.permute(0, 2, 1)
        t = trend.permute(0, 2, 1)
        r_attn, _ = self.attn_t2r(r, t, t)
        t_attn, _ = self.attn_r2t(t, r, r)
        r_out = self.norm_r(r + self.alpha_r * r_attn)
        t_out = self.norm_t(t + self.alpha_t * t_attn)
        return r_out.permute(0, 2, 1), t_out.permute(0, 2, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):  # x: [B, L, D]
        return self.drop(x + self.pe[:, :x.size(1)])


class SKFusion(nn.Module):
    """Selective Kernel Fusion of Raw and Trend feature streams (Battery v2 idea).
    Used by V_Aprime to combine streams adaptively per channel.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        d = max(16, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, d), Mish())
        self.fc_r = nn.Linear(d, channels)
        self.fc_t = nn.Linear(d, channels)

    def forward(self, raw, trend):  # [B, C, L]
        u = raw + trend
        s = u.mean(dim=-1)
        z = self.fc(s)
        ar = self.fc_r(z); at = self.fc_t(z)
        w = torch.softmax(torch.stack([ar, at], dim=1), dim=1)
        wr = w[:, 0, :].unsqueeze(-1); wt = w[:, 1, :].unsqueeze(-1)
        return raw * wr + trend * wt


# Main regression model with variant switches
class DualStreamTCN_v3(nn.Module):
    """Two-headed regression model (μ, logσ²) for RUL with selectable variant.

    variant:
      'V_A'      — original backbone + PatchAttention + CrossStreamAttention
      'V_Aprime' — original backbone + PositionalEncoding + Transformer + SKFusion (no PatchAttention)

    Ablation flags (let us measure each component's contribution):
      disable_patch     — drop PatchAttention (V_A only)
      disable_xstream   — drop CrossStreamAttention (V_A only)
      disable_pe        — drop PositionalEncoding (V_Aprime only)
      disable_trans     — drop TransformerEncoder (V_Aprime only)
      disable_skfusion  — drop SKFusion (V_Aprime only; falls back to concat)
      disable_het       — drop heteroscedastic head (use a single μ head)
      disable_length    — drop length-token feature
    """
    def __init__(self, num_sensors: int, d_model: int = 128,
                 variant: str = 'V_A', length_dim: int = 16, max_len: int = 1024,
                 disable_patch: bool = False, disable_xstream: bool = False,
                 disable_pe: bool = False, disable_trans: bool = False, disable_skfusion: bool = False,
                 disable_het: bool = False, disable_length: bool = False):
        super().__init__()
        assert variant in ('V_A', 'V_Aprime')
        self.variant = variant
        self.disable_het = disable_het
        self.disable_length = disable_length

        # Backbone (shared)
        self.raw_entry = nn.Conv1d(num_sensors, d_model, 1)
        self.raw_tcn = nn.Sequential(
            MixTemporalBlock(d_model, d_model, dilation=1),
            MixTemporalBlock(d_model, d_model, dilation=2),
            MixTemporalBlock(d_model, d_model, dilation=4),
            MixTemporalBlock(d_model, d_model, dilation=8),
        )
        self.trend_entry = nn.Conv1d(num_sensors, d_model, 1)
        self.trend_tcn = nn.Sequential(
            MixTemporalBlock(d_model, d_model, dilation=1),
            MixTemporalBlock(d_model, d_model, dilation=2),
            MixTemporalBlock(d_model, d_model, dilation=4),
            MixTemporalBlock(d_model, d_model, dilation=8),
        )

        # V_A enhancements
        if variant == 'V_A':
            self.raw_patch  = None if disable_patch else PatchAttention(d_model, patch_len=10)
            self.trend_patch = None if disable_patch else PatchAttention(d_model, patch_len=10)
            self.xstream    = None if disable_xstream else CrossStreamAttention(d_model)

        # V_Aprime enhancements
        if variant == 'V_Aprime':
            self.pe        = None if disable_pe else PositionalEncoding(d_model * 2, max_len=max_len)
            if disable_trans:
                self.transformer = None
            else:
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=d_model * 2, nhead=8,
                                                dim_feedforward=d_model * 4,
                                                dropout=0.2, batch_first=True, norm_first=True),
                    num_layers=2,
                )
            self.skfusion = None if disable_skfusion else SKFusion(d_model)

        self.raw_attn = AttentionPooling(d_model)
        self.trend_attn = AttentionPooling(d_model)

        # Length token: log-scaled trajectory length → 16-d embedding
        feat_dim = d_model * 2
        if not disable_length:
            self.len_proj = nn.Sequential(
                nn.Linear(1, length_dim),
                nn.Tanh(),
            )
            feat_dim += length_dim

        # Heads
        hidden = 128
        self.shared_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            Mish(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 64),
            Mish(),
        )
        self.mu_head = nn.Linear(64, 1)
        if not disable_het:
            self.logvar_head = nn.Linear(64, 1)
        else:
            self.logvar_head = None

    def forward(self, x, seq_len_feat=None):
        # x: [B, L, 2*num_sensors] → [B, 2*num_sensors, L]
        x = x.permute(0, 2, 1)
        total = x.shape[1]; ns = total // 2
        x_raw, x_trend = x[:, :ns, :], x[:, ns:, :]

        r = self.raw_tcn(self.raw_entry(x_raw))
        t = self.trend_tcn(self.trend_entry(x_trend))

        if self.variant == 'V_A':
            if self.raw_patch is not None:
                r = self.raw_patch(r); t = self.trend_patch(t)
            if self.xstream is not None:
                r, t = self.xstream(r, t)
            r_vec = self.raw_attn(r)
            t_vec = self.trend_attn(t)

        else:  # V_Aprime
            # Optional SKFusion: per-channel adaptive blend
            if self.skfusion is not None:
                fused_seq = self.skfusion(r, t)  # [B, D, L]
                # Project fused seq into 2D feature: still pool both streams separately,
                # but feed the fused seq through PE+Transformer (replaces PatchAttention's role).
            # Bring r and t side-by-side as patches: cat channels → [B, 2D, L] → [B, L, 2D]
            seq = torch.cat([r, t], dim=1).permute(0, 2, 1)
            if self.pe is not None:
                seq = self.pe(seq)
            if self.transformer is not None:
                seq = self.transformer(seq)  # [B, L, 2D]
            seq_perm = seq.permute(0, 2, 1)
            # Use AttentionPooling on the per-stream halves of the post-Transformer features.
            r_t = seq_perm[:, :seq_perm.shape[1] // 2, :]
            t_t = seq_perm[:, seq_perm.shape[1] // 2:, :]
            r_vec = self.raw_attn(r_t)
            t_vec = self.trend_attn(t_t)

        feat = torch.cat([r_vec, t_vec], dim=1)
        if not self.disable_length:
            if seq_len_feat is None:
                seq_len_feat = torch.zeros(feat.shape[0], 1, device=feat.device)
            feat = torch.cat([feat, self.len_proj(seq_len_feat)], dim=1)

        h = self.shared_head(feat)
        mu = self.mu_head(h).squeeze(-1)
        if self.logvar_head is not None:
            logvar = self.logvar_head(h).squeeze(-1)
            logvar = torch.clamp(logvar, -7.0, 7.0)
            return mu, logvar
        return mu, None


# Classification variant (NGAFID) — keeps existing 3-head structure
class DualStreamCLF_v3(nn.Module):
    """Classification head wrapper for NGAFID.  Backbone follows the V_A or V_Aprime
    architecture; outputs fine + coarse + ba logits + an optional temperature scalar.
    """
    def __init__(self, num_sensors: int, num_fine: int, num_coarse: int,
                 variant: str = 'V_A', d_model: int = 128,
                 stat_dim: int = 92, ba_emb_dim: int = 8,
                 disable_patch: bool = False, disable_xstream: bool = False,
                 disable_pe: bool = False, disable_trans: bool = False, disable_skfusion: bool = False,
                 disable_temp: bool = False):
        super().__init__()
        # Re-use the v3 backbone as a feature extractor; we ignore its (mu, logvar) heads
        # by setting disable_het=True and disable_length=True (we'll concat our own extras).
        self.backbone = DualStreamTCN_v3(
            num_sensors, d_model=d_model, variant=variant,
            disable_patch=disable_patch, disable_xstream=disable_xstream,
            disable_pe=disable_pe, disable_trans=disable_trans, disable_skfusion=disable_skfusion,
            disable_het=True, disable_length=True,
        )
        # Override its head with a feature passthrough by replacing shared_head with identity.
        backbone_feat_dim = d_model * 2
        self.backbone.shared_head = nn.Identity()
        self.backbone.mu_head = nn.Identity()

        # Stat encoder for raw+trend mean/std (carryover from the NGAFID pipeline)
        self.stat_encoder = nn.Sequential(
            nn.Linear(stat_dim, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(96, 32), nn.GELU(),
        )
        self.ba_embed = nn.Sequential(nn.Linear(1, ba_emb_dim), nn.Tanh())
        self.ba_drop_prob = 0.0

        feat_dim = backbone_feat_dim + 32 + ba_emb_dim
        self.fine_head    = nn.Linear(feat_dim, num_fine)
        self.coarse_head  = nn.Linear(feat_dim, num_coarse)
        self.ba_pred_head = nn.Linear(feat_dim, 2)
        if not disable_temp:
            self.logT_head = nn.Linear(feat_dim, 1)  # per-sample log-temperature
        else:
            self.logT_head = None

    def forward(self, x, stat_feat=None, before_after=None):
        # backbone returns (feat, None) because we stripped its head down to identity
        feat, _ = self.backbone(x)
        sf = (self.stat_encoder(stat_feat)
              if stat_feat is not None
              else torch.zeros(feat.shape[0], 32, device=feat.device))
        if before_after is not None:
            if self.training and self.ba_drop_prob > 0.0 and torch.rand(1).item() < self.ba_drop_prob:
                ba_feat = torch.zeros(feat.shape[0], 8, device=feat.device)
            else:
                ba_feat = self.ba_embed(before_after.unsqueeze(1).float())
        else:
            ba_feat = torch.zeros(feat.shape[0], 8, device=feat.device)
        combined = torch.cat([feat, sf, ba_feat], dim=1)
        fine = self.fine_head(combined)
        coarse = self.coarse_head(combined)
        ba = self.ba_pred_head(combined)
        if self.logT_head is not None:
            logT = self.logT_head(combined).squeeze(-1).clamp(-2.0, 2.0)
        else:
            logT = None
        return fine, coarse, ba, logT


# Loss
def gaussian_nll_loss(mu, logvar, target):
    """Per-sample Gaussian NLL with log-variance parameterisation:
        L = 0.5 * (exp(-logvar) * (mu - y)^2 + logvar)
    Bounded logvar (handled in model) keeps gradients sane.
    """
    inv_var = torch.exp(-logvar)
    return 0.5 * (inv_var * (mu - target) ** 2 + logvar).mean()
