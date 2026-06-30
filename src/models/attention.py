"""
Hybrid Attention Neck for Badger.

This implements the key mathematical insight that makes D-FINE/RT-DETR beat YOLO:
  - CNNs: receptive field grows O(depth) — linear, local
  - ViT attention: global receptive field from layer 1 — O(N²) complexity
  - Swin window attention: O(N) complexity with cross-window connections

We combine window-based self-attention with global tokens to get:
  - CNN-like efficiency (linear in spatial size)
  - ViT-like global context (attention across the full feature map)
  - Cross-scale fusion via the existing PA-FPN pathway

The math (Vaswani et al., 2017):
  Attention(Q,K,V) = softmax(QK^T / √d_k) × V

For window attention, this is computed independently within non-overlapping
windows of size M×M, reducing complexity from O(H²W²) to O(M²HW).

Reference papers:
  - Liu et al., "Swin Transformer" (arXiv:2103.14030)
  - Dosovitskiy et al., "ViT" (arXiv:2010.11929)
  - Zhu et al., "Deformable DETR" (arXiv:2010.04159)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Conv, make_divisible


# =============================================================================
# Window-based Multi-Head Self-Attention
# =============================================================================

class WindowAttention(nn.Module):
    """
    Multi-head self-attention within local windows.

    For an input of shape (H, W, C), we partition into windows of size
    M×M and compute attention independently within each window.

    Complexity: O(M² × H/M × W/M × C) = O(M² × num_windows × C)
    vs. global attention: O(H²W² × C)

    With typical values M=7, H=W=80: global would need 80⁴ = 40M operations
    per head; windowed needs only 7² × (80/7)² ≈ 12K operations — 3000× less.
    """

    def __init__(self, dim, num_heads=8, window_size=7, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/√d_k from the attention formula

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Linear projections for Q, K, V — combined for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, C) — where N = H*W, already partitioned into windows
            mask: optional attention mask for shifted windows

        Returns:
            (B, N, C) — attended features
        """
        B, N, C = x.shape

        # QKV projection: (B, N, C) → (B, N, 3*C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        # QK^T / √d_k: (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask for shifted windows (if provided)
        if mask is not None:
            # mask shape: (num_windows, window_size², window_size²)
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # softmax(QK^T / √d_k)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # softmax(...) × V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# =============================================================================
# Swin Transformer Block
# =============================================================================

class SwinBlock(nn.Module):
    """
    Swin Transformer block: window attention + MLP, with residual connections.

    Uses shifted window partitioning on alternating blocks to enable
    cross-window connections without explicit communication.

    Block 1 (W-MSA): regular window partitioning
    Block 2 (SW-MSA): windows shifted by (M/2, M/2) pixels

    The shift creates connections between adjacent windows, giving the
    next layer a receptive field spanning multiple windows — very clever.
    """

    def __init__(self, dim, num_heads=8, window_size=7, shift_size=0,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # Layer Norm before attention (Pre-LN — more stable than Post-LN)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, dropout=dropout)

        # MLP: dim → dim*mlp_ratio → dim  (with GELU — standard for transformers)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),  # Smooth activation, better for transformers than ReLU
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def _window_partition(self, x, window_size):
        """
        Partition feature map into non-overlapping windows.

        Input:  (B, H, W, C)
        Output: (B * num_windows, window_size², C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size * window_size, C)
        return windows

    def _window_reverse(self, windows, window_size, H, W):
        """
        Reverse window partition back to feature map.
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size,
                        window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C) — flattened spatial features
            H, W: spatial dimensions

        Returns:
            (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = self._window_partition(shifted_x, self.window_size)
        # Attention within windows
        attn_windows = self.attn(x_windows)
        # Merge windows back
        shifted_x = self._window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual connection — the "1" in ∂L/∂x = ∂L/∂y(1 + ∂F/∂x)
        x = shortcut + x

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


# =============================================================================
# Global Token Mixer — cross-scale attention
# =============================================================================

class GlobalTokenMixer(nn.Module):
    """
    Mixes information across feature scales using global tokens.

    This is inspired by the "encoder" in DETR-style detectors, but much simpler:
      - Pool a small set of "global tokens" from each feature scale
      - Apply cross-attention between scales via these tokens
      - Scatter the tokens back to enrich each scale

    The math is the same attention formula, but Q comes from one scale and
    K,V from another — this is cross-attention, the key operation that lets
    D-FINE integrate context across the entire image.
    """

    def __init__(self, channels, num_tokens=64, num_heads=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.channels = channels

        # Learnable global tokens
        self.tokens = nn.Parameter(torch.randn(3, num_tokens, channels) * 0.02)

        # Cross-attention: each scale's tokens attend to all others
        self.cross_attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )

        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, features):
        """
        Args:
            features: list of [B, C, H_i, W_i] — P3, P4, P5

        Returns:
            list of [B, C, H_i, W_i] — enriched features
        """
        B = features[0].shape[0]
        device = features[0].device

        # Expand global tokens per batch
        tokens = self.tokens.unsqueeze(1).expand(-1, B, -1, -1)  # (3, B, num_tokens, C)

        # Pool spatial features into token space per scale
        pooled = []
        for i, feat in enumerate(features):
            # Adaptive average pool each scale to num_tokens
            p = F.adaptive_avg_pool2d(feat, (8, 8))  # (B, C, 8, 8)
            p = p.flatten(2).transpose(1, 2)  # (B, 64, C)
            pooled.append(p)

        # Cross-attention: each scale's pooled features attend to global tokens
        for i in range(3):
            # Query: this scale's tokens, Key/Value: global tokens
            attn_out, _ = self.cross_attn(
                pooled[i],                 # Q: scale-specific
                tokens[i % 3],             # K: global tokens
                tokens[(i + 1) % 3],       # V: different global tokens (cross-scale)
                need_weights=False
            )
            pooled[i] = self.norm(pooled[i] + attn_out)
            pooled[i] = pooled[i] + self.mlp(pooled[i])

        # Scatter enriched tokens back (broadcast to spatial dimensions)
        outputs = []
        for i, feat in enumerate(features):
            _, _, H, W = feat.shape
            # Broadcast token info spatially
            token_info = pooled[i].transpose(1, 2).view(B, -1, 8, 8)
            token_info = F.interpolate(token_info, size=(H, W), mode='bilinear', align_corners=False)
            outputs.append(feat + token_info)

        return outputs


# =============================================================================
# Hybrid Attention Neck — combines CNN FPN with Swin attention
# =============================================================================

class AttentionNeck(nn.Module):
    """
    Hybrid neck: PA-FPN for cross-scale fusion + Swin attention for
    long-range dependencies + Global Token Mixer for cross-scale reasoning.

    This is our key architectural improvement over YOLOv8's neck.
    It combines three mathematical principles:
      1. CNN FPN: efficient multi-scale feature fusion (VGG insight)
      2. Window attention: O(N) global context (Swin insight)
      3. Cross-scale attention: DETR-style global reasoning

    The idea: use CNN operations for what they're best at (local feature
    extraction), and attention for what it's best at (global reasoning).
    """

    def __init__(self, in_channels, width_multiple=0.5, depth_multiple=0.33,
                 use_attention=True, use_global_tokens=True):
        super().__init__()
        from .neck import PAFPN

        # Standard PA-FPN for efficient cross-scale fusion
        self.fpn = PAFPN(in_channels, width_multiple, depth_multiple)
        self.out_channels = self.fpn.out_channels

        reduce_ch = self.out_channels[0]

        # Swin attention blocks — applied after FPN on each scale
        self.use_attention = use_attention
        if use_attention:
            self.attn_blocks = nn.ModuleList([
                nn.ModuleList([
                    SwinBlock(reduce_ch, num_heads=8, window_size=7, shift_size=0),
                    SwinBlock(reduce_ch, num_heads=8, window_size=7, shift_size=3),
                ]) for _ in range(3)  # One pair per scale (W-MSA + SW-MSA)
            ])

        # Global token mixer for cross-scale reasoning
        self.use_global_tokens = use_global_tokens
        if use_global_tokens:
            self.global_mixer = GlobalTokenMixer(reduce_ch)

    def forward(self, features):
        """
        Args:
            features: [P3, P4, P5] from backbone

        Returns:
            [N3, N4, N5] — attention-enriched fused features
        """
        # Standard FPN fusion
        fpn_features = self.fpn(features)

        # Apply Swin attention per scale
        if self.use_attention:
            attended = []
            for i, feat in enumerate(fpn_features):
                B, C, H, W = feat.shape

                # Pad spatial dims to be divisible by window_size
                ws = self.attn_blocks[i][0].window_size
                pad_h = (ws - H % ws) % ws
                pad_w = (ws - W % ws) % ws
                if pad_h > 0 or pad_w > 0:
                    feat = F.pad(feat, (0, pad_w, 0, pad_h))
                    H_pad, W_pad = H + pad_h, W + pad_w
                else:
                    H_pad, W_pad = H, W

                # Reshape to (B, H*W, C) for attention
                x = feat.flatten(2).transpose(1, 2)
                # W-MSA block
                x = self.attn_blocks[i][0](x, H_pad, W_pad)
                # SW-MSA block (shifted windows)
                x = self.attn_blocks[i][1](x, H_pad, W_pad)
                # Back to (B, C, H_pad, W_pad)
                x = x.transpose(1, 2).view(B, C, H_pad, W_pad)

                # Crop back to original size
                if pad_h > 0 or pad_w > 0:
                    x = x[:, :, :H, :W]

                attended.append(x)
            fpn_features = attended

        # Cross-scale global mixing
        if self.use_global_tokens:
            fpn_features = self.global_mixer(fpn_features)

        return fpn_features
