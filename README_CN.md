# 多模态谣言检测 — 模型架构

---

## 概述

基于图像与文本多模态融合的社交媒体谣言检测框架，在 PHEME 和微博数据集上进行了验证。

该架构在轻量级 CNN-Transformer 混合骨干网络中集成以下创新：

| 模块 | 描述 |
|------|------|
| **MUDD** | 多路动态密集连接 — 跨层特征路由，可学习动态权重 |
| **自适应门控融合 (AGF)** | 样本级自适应模态融合，平衡跨模态一致性与差异性 |

辅助技术：**差分注意力 (Differential Attention)** 用于去噪视觉特征精炼。

---

## 文件结构

```
├── README.md
├── README_CN.md
├── config.py                         # 架构超参数
├── requirements.txt
└── models/
    ├── __init__.py
    ├── helpers.py                    # ConvModule 等工具层
    ├── differential_attention.py     # 差分注意力
    ├── mudd.py                       # MUDD 连接
    ├── adaptive_gated_fusion.py      # AGF 融合模块
    ├── mamba_encoder.py             # Mamba 文本编码器
    ├── transxnet_backbone.py        # 改进的 TransXNet 骨干网络
    └── rumor_detector.py            # 完整多模态谣言检测模型
```

---

## 核心模块

### MUDD — `models/mudd.py`

多路动态密集连接。将通道维度分割为 N 条并行路径，通过可学习的权重矩阵动态路由所有前层的特征信息。采用深度可分离卷积保持轻量化。

### 差分注意力 — `models/differential_attention.py`

将查询头分为两组 (Q₁, Q₂)，对 Q₂ 施加 GroupNorm 归一化，计算两组注意力分数的差值：`Attn = softmax(Q₁K₁ᵀ) − λ·softmax(Q₂K₂ᵀ)`。在较深阶段（Stage 3–4）应用，用于抑制冗余注意力模式。

### 自适应门控融合 — `models/adaptive_gated_fusion.py`

构建交互向量 `c = [f_I; f_T; f_I⊙f_T; |f_I−f_T|]`，同时捕捉跨模态一致性 (⊙) 和差异性 (|·|)。轻量 MLP 生成样本级门控权重：`f_fused = g⊙f_I + (1−g)⊙f_T`。

---

## 依赖

- PyTorch ≥ 2.0
- timm ≥ 0.6.12

```bash
pip install -r requirements.txt
```

---

## 引用

如果您认为本工作对您的研究有帮助，请引用：

```bibtex
@article{XXX,
  title     = {XXX},
  author    = {XXX},
  journal   = {XXX},
  year      = {2026},
}
```

TransXNet 骨干网络基于：

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

Mamba 文本编码器基于：

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

## 许可证

本项目用于学术用途，完整许可条款将在论文发表后提供。
