"""
Architecture configuration for the Multimodal Rumor Detector.
"""


class ModelArchConfig:
    """Model architecture hyperparameters."""

    # ── Input ──
    input_size = 256
    max_seq_length = 128
    vocab_size = 10000
    num_classes = 2

    # ── Image branch: TransXNet backbone ──
    backbone_arch = {
        'layers':      [3, 3, 9, 3],
        'embed_dims':  [48, 96, 224, 448],
        'kernel_size': [7, 7, 7, 7],
        'num_groups':  [2, 2, 2, 2],
        'sr_ratio':    [8, 4, 2, 1],
        'num_heads':   [1, 2, 4, 8],
        'mlp_ratios':  [4, 4, 4, 4],
        'layer_scale_init_value': 1e-5,
    }

    use_mudd = True
    use_differential_attn = True
    da_stages = [2, 3]

    # ── Text branch: Mamba ──
    embedding_dim = 128
    mamba_n_layers = 4
    mamba_d_state = 16
    mamba_expand_factor = 2

    # ── Fusion & Classifier ──
    image_feat_dim = 128
    dropout = 0.5
