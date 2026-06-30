"""
Multimodal Rumor Detector — Full Model Architecture

Complete architecture for multimodal rumor detection. Image and text
branches are processed independently, fused via Adaptive Gated Fusion (AGF),
and classified by a lightweight MLP head.

  Image branch:  Lightweight TransXNet-T (modified)
                 → MUDD (stages 3-4) + Differential Attention (stages 3-4)
                 → AdaptiveAvgPool2d → Linear(448→128)

  Text branch:   4-layer Mamba SSM
                 → AdaptiveAvgPool1d → Linear(128→128)

  Fusion:        Adaptive Gated Fusion (AGF)
                 c = [f_I; f_T; f_I ⊙ f_T; |f_I - f_T|]  ∈ R^{4d}
                 g = σ(W₂ · ReLU(W₁ · c + b₁) + b₂)      ∈ (0,1)^d
                 f_fused = g ⊙ f_I + (1 - g) ⊙ f_T       ∈ R^d

  Classifier:    Linear(128→64) → ReLU → Dropout → Linear(64→2)
"""

import torch
from torch import nn

from .transxnet_backbone import create_transxnet_t
from .adaptive_gated_fusion import AdaptiveGatedFusion
from .mamba_encoder import MambaConfig, MambaSequenceModel


class MultimodalRumorDetector(nn.Module):
    """
    Multimodal Rumor Detection Model.

    Args:
        vocab_size:     Vocabulary size (default 10000).
        embedding_dim:  Token embedding & Mamba hidden dimension (default 128).
        max_seq_length: Maximum text sequence length (default 128).
        num_classes:    Number of output classes (default 2).
        img_size:       Input image resolution (default 256).
        feat_dim:       Feature dimension after projection (default 128).
        dropout:        Dropout rate in classifier (default 0.5).
    """

    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=128,
        max_seq_length=128,
        num_classes=2,
        img_size=256,
        feat_dim=128,
        dropout=0.5,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length

        # ── Image branch ──────────────────────────────────────────
        # Lightweight TransXNet-T with MUDD (stages 3-4) + DA (stages 3-4)
        self.image_backbone = create_transxnet_t(
            img_size=img_size,
            use_mudd=True,
            use_differential_attn=True,
            da_stages=[2, 3],
        )

        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_proj = nn.Linear(448, feat_dim)

        # ── Text branch ───────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        mamba_config = MambaConfig(
            d_model=embedding_dim,
            n_layers=4,
            d_state=16,
            expand_factor=2,
            d_conv=4,
            dt_rank='auto',
        )
        self.text_backbone = MambaSequenceModel(mamba_config)

        self.text_pool = nn.AdaptiveAvgPool1d(1)
        self.text_proj = nn.Linear(embedding_dim, feat_dim)

        # ── Fusion ────────────────────────────────────────────────
        # AGF: c = [f_I; f_T; f_I⊙f_T; |f_I-f_T|] → MLP gate → weighted sum
        self.fusion = AdaptiveGatedFusion(feat_dim)

        # ── Classifier head ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, text):
        """
        Args:
            image: [B, 3, H, W] RGB images.
            text:  [B, L] token indices.
        Returns:
            logits: [B, num_classes].
        """
        # Image features
        x = self.image_backbone.forward_embeddings(image)
        img_feat = self.image_backbone.forward_tokens(x)
        img_feat = self.image_pool(img_feat).flatten(1)
        img_feat = self.image_proj(img_feat)

        # Text features
        text_emb = self.embedding(text)
        text_feat = self.text_backbone(text_emb)
        text_feat = text_feat.transpose(1, 2)
        text_feat = self.text_pool(text_feat).squeeze(-1)
        text_feat = self.text_proj(text_feat)

        # AGF Fusion + Classification
        fused = self.fusion(img_feat, text_feat)
        output = self.classifier(fused)
        return output
