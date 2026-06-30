"""
Mamba Text Encoder

State Space Model (Mamba) for sequence modeling, based on:

  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces,"
  arXiv:2312.00752, 2023.
"""

import math
from dataclasses import dataclass
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MambaConfig:
    """Configuration for a Mamba block / sequence model."""
    d_model: int                  # Hidden dimension
    n_layers: int                 # Number of Mamba blocks
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16             # SSM state dimension (N)
    expand_factor: int = 2        # Inner dim expansion (E)
    d_conv: int = 4               # Conv1d kernel size
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True            # Parallel scan mode

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E * D
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


# ═══════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class MambaBlock(nn.Module):
    """
    Single Mamba block.

    Architecture (simplified):
      input → RMSNorm → Linear(proj) → Conv1d → SiLU →
        Selective SSM (discretized A, B, C, Δ) → RMSNorm → output

    The selective SSM uses input-dependent Δ, B, C parameters to
    achieve content-aware sequence processing.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Input projection: D → 2*ED (split into two branches)
        self.in_proj = nn.Linear(
            config.d_model, 2 * config.d_inner, bias=config.bias)

        # 1D depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner, out_channels=config.d_inner,
            kernel_size=config.d_conv, bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1)

        # Project to input-dependent Δ, B, C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # Project Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # Δ initialization
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(config.d_inner) *
            (math.log(config.dt_max) - math.log(config.dt_min)) +
            math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias = nn.Parameter(inv_dt)

        # A matrix (state transition) — low-rank parameterization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))

        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        # x: [B, L, D]
        residual = x
        x = RMSNorm(self.config.d_model)(x)

        # Project and split
        x_and_res = self.in_proj(x)  # [B, L, 2*ED]
        x, res = x_and_res.split(
            [self.config.d_inner, self.config.d_inner], dim=-1)

        # Convolution
        x = x.transpose(1, 2)  # [B, ED, L]
        x = self.conv1d(x)[:, :, :x.size(-1)]  # trim padding
        x = F.silu(x)
        x = self._selective_scan(x)
        x = x.transpose(1, 2)  # [B, L, ED]

        # Gated output
        x = F.silu(res) * x
        x = self.out_proj(x)
        return x + residual

    def _selective_scan(self, x):
        """Selective SSM scan (discretized state-space update)."""
        B, D, L = x.shape
        x_flat = x.transpose(1, 2).reshape(B * L, D)

        # Input-dependent parameters
        delta_bc = self.x_proj(x_flat)  # [B*L, dt_rank + 2*d_state]
        delta, B_ssm, C_ssm = delta_bc.split(
            [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)

        # Δ projection with softplus
        delta = self.dt_proj(delta)  # [B*L, D]
        delta = F.softplus(delta)

        # Discretize A
        A = -torch.exp(self.A_log.float())  # [D, d_state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [B*L, D, d_state]
        deltaB = delta.unsqueeze(-1) * B_ssm.unsqueeze(1)  # [B*L, D, d_state]

        # Reshape for scan
        deltaA = deltaA.reshape(B, L, D, self.config.d_state)
        deltaB = deltaB.reshape(B, L, D, self.config.d_state)
        C_ssm = C_ssm.reshape(B, L, D, self.config.d_state)

        # Parallel selective scan
        x_reshaped = x.unsqueeze(-1)  # [B, D, L, 1]
        h = torch.zeros(B, D, self.config.d_state, device=x.device)
        outputs = []

        for t in range(L):
            h = deltaA[:, t] * h + deltaB[:, t] * x_reshaped[:, :, t]
            y = (h * C_ssm[:, t]).sum(dim=-1)  # [B, D]
            outputs.append(y)

        y = torch.stack(outputs, dim=-1)  # [B, D, L]
        y = y + x * self.D.unsqueeze(0).unsqueeze(-1)
        return y  # [B, D, L]


class MambaSequenceModel(nn.Module):
    """
    Stacked Mamba blocks forming a text encoder.

    Args:
        config: MambaConfig with d_model, n_layers, etc.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        """x: [B, L, d_model] → [B, L, d_model]"""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
