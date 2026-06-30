"""
Multimodal Rumor Detection — Model Architecture

  differential_attention  — Differential Attention (DA)
  mudd                    — Multiway Dynamic Dense Connection (MUDD)
  adaptive_gated_fusion   — Adaptive Gated Fusion (AGF)
  mamba_encoder           — Mamba SSM text encoder
  transxnet_backbone      — Enhanced TransXNet backbone with MUDD/DA
  rumor_detector          — Full MultimodalRumorDetector
  helpers                 — ConvModule and utility layers
"""

from .differential_attention import DifferentialAttention
from .mudd import MUDDConnection
from .adaptive_gated_fusion import AdaptiveGatedFusion
from .mamba_encoder import MambaConfig, MambaBlock, MambaSequenceModel, RMSNorm
from .transxnet_backbone import (
    TransXNet,
    create_transxnet_t,
    create_transxnet_s,
    create_transxnet_b,
)
from .rumor_detector import MultimodalRumorDetector
