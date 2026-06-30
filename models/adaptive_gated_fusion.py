"""
Adaptive Gated Fusion (AGF)

A learnable fusion module that adaptively balances image and text modalities
on a per-sample basis, preserving both cross-modal agreement and discrepancy.

Fusion mechanism:
    1. Interaction vector: c = [f_I; f_T; f_I ⊙ f_T; |f_I - f_T|]  ∈ R^{4d}
       - f_I ⊙ f_T  captures cross-modal agreement (element-wise product)
       - |f_I - f_T| encodes cross-modal discrepancy

    2. Adaptive gate: g = σ(W₂ · ReLU(W₁ · c + b₁) + b₂)  ∈ (0, 1)^d

    3. Fused output:  f_fused = g ⊙ f_I + (1 - g) ⊙ f_T

The gate g learns to emphasize the more reliable modality for each sample,
while the interaction vector c provides rich cross-modal context.
"""

import torch
from torch import nn


class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive Gated Fusion for multimodal representation.

    Args:
        dim: Feature dimension of each modality (f_I and f_T).
        hidden_ratio: Expansion ratio for the hidden layer (default 2).
    """

    def __init__(self, dim, hidden_ratio=2):
        super().__init__()
        hidden_dim = dim * hidden_ratio
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),  # c = [f_I; f_T; f_I⊙f_T; |f_I-f_T|]
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),                     # g ∈ (0, 1)^d
        )

    def forward(self, f_I, f_T):
        """
        Args:
            f_I: Image features  [B, dim]
            f_T: Text features   [B, dim]
        Returns:
            Fused representation [B, dim]
        """
        # Interaction vector: agreement (⊙) + discrepancy (|·|)
        c = torch.cat([f_I, f_T, f_I * f_T, torch.abs(f_I - f_T)], dim=-1)
        g = self.gate(c)                       # Adaptive gate
        return g * f_I + (1 - g) * f_T         # Weighted fusion
