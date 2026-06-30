"""
Helper utilities for the TransXNet backbone.
Provides ConvModule, activation/normalization builders without requiring mmcv.
"""

import torch
from torch import nn


def build_activation_layer(act_cfg):
    """Build activation layer from config dict (mmcv-compatible)."""
    if act_cfg is None:
        return nn.Identity()
    act_type = act_cfg.get('type', 'ReLU')
    if act_type == 'GELU':
        return nn.GELU()
    if act_type == 'SiLU':
        return nn.SiLU()
    if act_type == 'ReLU':
        return nn.ReLU(inplace=act_cfg.get('inplace', True))
    raise ValueError(f"Unsupported activation type: {act_type}")


def build_norm_layer(norm_cfg, num_features):
    """Build normalization layer from config dict (mmcv-compatible)."""
    norm_type = norm_cfg.get('type', 'BN2d')
    if norm_type in ('BN', 'BN2d'):
        layer = nn.BatchNorm2d(num_features)
    elif norm_type == 'GN':
        layer = nn.GroupNorm(norm_cfg.get('num_groups', 1), num_features)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    return norm_type, layer


class ConvModule(nn.Module):
    """Conv + Norm + Act block (mmcv-compatible fallback)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias,
        )
        self.norm = build_norm_layer(norm_cfg, out_channels)[1] if norm_cfg is not None else nn.Identity()
        self.act = build_activation_layer(act_cfg) if act_cfg is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
