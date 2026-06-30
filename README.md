# Multimodal Rumor Detection — Model Architecture

---

## Overview

Multimodal rumor detection framework that fuses image and text modalities
for detecting rumors on social media. Evaluated on PHEME and Weibo datasets.

The architecture integrates two novel components into a lightweight
CNN-Transformer hybrid backbone:

| Component | Description |
|-----------|-------------|
| **MUDD** | Multiway Dynamic Dense Connection — cross-layer feature routing with learnable dynamic weights |
| **Adaptive Gated Fusion (AGF)** | Sample-adaptive modality fusion balancing agreement and discrepancy |

Additional technique: **Differential Attention (DA)** for noise-reducing
visual feature refinement.

---

## File Structure

```
├── README.md
├── README_CN.md
├── config.py                         # Architecture hyperparameters
├── requirements.txt
└── models/
    ├── __init__.py
    ├── helpers.py                    # ConvModule and utility layers
    ├── differential_attention.py     # Differential Attention
    ├── mudd.py                       # MUDD connection
    ├── adaptive_gated_fusion.py      # AGF fusion module
    ├── mamba_encoder.py             # Mamba SSM text encoder
    ├── transxnet_backbone.py        # Enhanced TransXNet backbone
    └── rumor_detector.py            # Full MultimodalRumorDetector
```

---

## Key Modules

### MUDD — `models/mudd.py`

Multiway Dynamic Dense Connection. Splits channels into N parallel paths and
dynamically routes information from all previous layers through a learned
weight matrix. Uses depthwise-separable convolutions for efficiency.

### Differential Attention — `models/differential_attention.py`

Splits query heads into two groups (Q₁, Q₂), normalizes Q₂ via GroupNorm,
and computes the difference between their attention scores:
`Attn = softmax(Q₁K₁ᵀ) − λ·softmax(Q₂K₂ᵀ)`. Applied in deeper stages (3–4).

### Adaptive Gated Fusion — `models/adaptive_gated_fusion.py`

Constructs an interaction vector `c = [f_I; f_T; f_I⊙f_T; |f_I−f_T|]`
capturing cross-modal agreement and discrepancy. A lightweight MLP generates
sample-specific gates: `f_fused = g⊙f_I + (1−g)⊙f_T`.

---

## Dependencies

- PyTorch ≥ 2.0
- timm ≥ 0.6.12

```bash
pip install -r requirements.txt
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{XXX,
  title     = {XXX},
  author    = {XXX},
  journal   = {XXX},
  year      = {2026},
}
```

The TransXNet backbone is based on:

```bibtex
@article{lou2023transxnet,
  title     = {TransXNet: Learning Both Global and Local Dynamics with
               a Dual Dynamic Token Mixer for Visual Recognition},
  author    = {Meng Lou and Shu Zhang and Hong-Yu Zhou and Sibei Yang
               and Chuan Wu and Yizhou Yu},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems},
  year      = {2025},
}
```

The Mamba text encoder is based on:

```bibtex
@article{gu2023mamba,
  title     = {Mamba: Linear-Time Sequence Modeling with Selective
               State Spaces},
  author    = {Albert Gu and Tri Dao},
  journal   = {arXiv preprint arXiv:2312.00752},
  year      = {2023},
}
```

---

## License

This project is released for academic purposes. Full license terms
will be provided upon publication.
