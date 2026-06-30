"""
Differential Attention (DA)

A noise-reducing attention mechanism that computes the difference between
two independent softmax attention distributions.

Mathematically:
    Attn_diff = softmax(Q1 @ K1^T / sqrt(d)) - λ · softmax(Q2 @ K2^T / sqrt(d))

where Q is split into two halves (Q1, Q2), with Q2 normalized via GroupNorm.
The differential operation suppresses common-mode noise, yielding cleaner
attention maps.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DifferentialAttention(nn.Module):
    """
    Differential Attention for visual feature refinement.

    Splits query heads into two groups, normalizes the second group, and
    computes the difference between their attention scores. This suppresses
    redundant attention patterns and emphasizes distinctive visual cues.
    """

    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # head_dim must be even for query splitting into two groups
        assert self.head_dim % 2 == 0, \
            f"head_dim {self.head_dim} must be divisible by 2 for query splitting"

        # QKV projection
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)

        # Learnable lambda parameter (scalar) to balance the two softmax groups
        # Initialized to 0.8 as described in the paper
        self.lambda_param = nn.Parameter(torch.tensor(0.8))

        # Output projection
        self.proj = nn.Conv2d(dim, dim, 1)
        self.attn_drop = nn.Dropout(attn_drop)

        # GroupNorm for Q2 normalization
        group_norm_channels = num_heads * (self.head_dim // 2)
        group_norm_channels = max(1, group_norm_channels)
        num_groups = min(2, group_norm_channels)
        self.group_norm = nn.GroupNorm(num_groups, group_norm_channels)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape to multi-head format: [B, num_heads, head_dim, H*W]
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        # Split queries into two groups for differential attention
        # q1, q2: [B, num_heads, head_dim//2, H*W]
        q1, q2 = torch.chunk(q, 2, dim=2)

        # Normalize the second query group via GroupNorm
        q2_reshaped = q2.reshape(B, self.num_heads * (self.head_dim // 2), H, W)
        q2_normalized = self.group_norm(q2_reshaped)
        q2 = q2_normalized.reshape(B, self.num_heads, self.head_dim // 2, H * W)

        # Split keys and values to match query halves
        k1, k2 = torch.chunk(k, 2, dim=2)

        # Compute attention scores for both groups
        # q1: [B, num_heads, H*W, head_dim//2], k1: [B, num_heads, head_dim//2, H*W]
        attn1 = torch.matmul(q1.transpose(-1, -2), k1) * self.scale
        attn2 = torch.matmul(q2.transpose(-1, -2), k2) * self.scale

        # Softmax normalization
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # Differential operation: A_diff = A1 - λ·A2 (λ applied after softmax)
        differential_attn = attn1 - self.lambda_param * attn2
        differential_attn = self.attn_drop(differential_attn)

        # Apply to values
        v1, v2 = torch.chunk(v, 2, dim=2)
        output1 = torch.matmul(differential_attn, v1.transpose(-1, -2))
        output2 = torch.matmul(differential_attn, v2.transpose(-1, -2))

        # Merge output halves
        x = torch.cat([output1, output2], dim=2)  # [B, num_heads, head_dim, H*W]
        x = x.reshape(B, self.dim, H, W)

        # Output projection
        x = self.proj(x)
        return x
