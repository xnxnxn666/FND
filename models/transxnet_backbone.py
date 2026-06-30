"""
Enhanced TransXNet Backbone with MUDD and Differential Attention.

Modified TransXNet backbone (Lou et al., TNNLS 2025) integrating:

  - MUDD (Multiway Dynamic Dense Connection): Cross-layer dynamic routing
  - Differential Attention (DA): Noise-reducing dual-group attention

Architecture Overview:
  Input (3 x H x W)
    ↓ PatchEmbed (7x7 conv, stride 4)
  Stage 1: [Block x N]  dim=48
    ↓ PatchEmbed (downsample)
  Stage 2: [Block x N]  dim=96
    ↓ PatchEmbed (downsample)
  Stage 3: [Block x N]  dim=224, MUDD + DA
    ↓ PatchEmbed (downsample)
  Stage 4: [Block x N]  dim=448, MUDD + DA
    ↓ Classifier (or feature output)

Each Block contains a D-Mixer (HybridTokenMixer) combining:
  - DynamicConv2d (local branch, IDConv)
  - Attention or DifferentialAttention (global branch)

Reference:
  Original TransXNet: Lou et al., TNNLS 2025.
  https://github.com/LMMMEng/TransXNet
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

try:
    from timm.layers import DropPath, to_2tuple
except ImportError:
    from timm.models.layers import DropPath, to_2tuple

from .differential_attention import DifferentialAttention
from .mudd import MUDDConnection
from .helpers import ConvModule, build_activation_layer, build_norm_layer


# ══════════════════════════════════════════════════════════════════════
# Patch Embedding
# ══════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Image to Patch Embedding via convolutional projection."""

    def __init__(self, patch_size=7, stride=4, padding=3,
                 in_chans=3, embed_dim=96, norm_layer=None, act_cfg=None):
        super().__init__()
        self.proj = ConvModule(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=stride, padding=padding,
            norm_cfg=norm_layer, act_cfg=act_cfg,
        )

    def forward(self, x):
        return self.proj(x)


# ══════════════════════════════════════════════════════════════════════
# OSRA: Overlapped Spatial Reduction Attention
# ══════════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """Overlapped Spatial Reduction Attention (OSRA)."""

    def __init__(self, dim, num_heads=1, qk_scale=None, attn_drop=0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)

        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim,
                          kernel_size=sr_ratio + 3, stride=sr_ratio,
                          padding=(sr_ratio + 3) // 2, groups=dim,
                          bias=False,
                          norm_cfg=dict(type='BN2d'), act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                          kernel_size=1, groups=dim, bias=False,
                          norm_cfg=dict(type='BN2d'), act_cfg=None),
            )
        else:
            self.sr = nn.Identity()

        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)

        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(
                    relative_pos_enc, size=attn.shape[2:],
                    mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)


# ══════════════════════════════════════════════════════════════════════
# IDConv: Input-Dependent Depthwise Convolution
# ══════════════════════════════════════════════════════════════════════

class DynamicConv2d(nn.Module):
    """Input-Dependent Dynamic Convolution (IDConv)."""

    def __init__(self, dim, kernel_size=3, reduction_ratio=4,
                 num_groups=1, bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias

        self.weight = nn.Parameter(
            torch.empty(num_groups, dim, kernel_size, kernel_size),
            requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim, dim // reduction_ratio, kernel_size=1,
                      norm_cfg=dict(type='BN2d'), act_cfg=dict(type='GELU')),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1),
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(
            B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(
                scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight, padding=self.K // 2,
                     groups=B * C, bias=bias)
        return x.reshape(B, C, H, W)


# ══════════════════════════════════════════════════════════════════════
# D-Mixer: Dual Dynamic Token Mixer
# ══════════════════════════════════════════════════════════════════════

class HybridTokenMixer(nn.Module):
    """
    Dual Dynamic Token Mixer (D-Mixer).

    Splits channels into halves:
      - Local:  IDConv (DynamicConv2d) for local dynamics
      - Global: Attention or DifferentialAttention for global context
    """

    def __init__(self, dim, kernel_size=3, num_groups=2, num_heads=1,
                 sr_ratio=1, reduction_ratio=8, use_differential_attn=False):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divisible by 2."

        self.local_unit = DynamicConv2d(
            dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)

        if use_differential_attn:
            self.global_unit = DifferentialAttention(
                dim=dim // 2, num_heads=num_heads)
        else:
            self.global_unit = Attention(
                dim=dim // 2, num_heads=num_heads, sr_ratio=sr_ratio)

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)

        if isinstance(self.global_unit, DifferentialAttention):
            x2 = self.global_unit(x2)
        else:
            x2 = self.global_unit(x2, relative_pos_enc)

        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x
        return x


# ══════════════════════════════════════════════════════════════════════
# Supporting Layers
# ══════════════════════════════════════════════════════════════════════

class MultiScaleDWConv(nn.Module):
    """Multi-scale depthwise convolution for MS-FFN."""

    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                           kernel_size=scale[i], padding=scale[i] // 2,
                           groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = [self.proj[i](feat) for i, feat in enumerate(x)]
        return torch.cat(out, dim=1)


class Mlp(nn.Module):
    """Multi-Scale Feed-Forward Network (MS-FFN)."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_cfg=dict(type='GELU'), drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    """LayerScale for training stability (CaiT-style)."""

    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])


# ══════════════════════════════════════════════════════════════════════
# Core Building Block
# ══════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """
    TransXNet Block with MUDD connection and Differential Attention.

      1. Position embedding (7x7 depthwise conv)
      2. MUDD cross-layer connection (stages 2-4)
      3. D-Mixer: local (IDConv) + global (Attention or DA) token mixing
      4. MS-FFN: multi-scale feature refinement
      5. LayerScale + DropPath
    """

    def __init__(self, dim=64, kernel_size=3, sr_ratio=1, num_groups=2,
                 num_heads=1, mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'), drop=0, drop_path=0,
                 layer_scale_init_value=1e-5, grad_checkpoint=False,
                 use_mudd=True, use_differential_attn=False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.token_mixer = HybridTokenMixer(
            dim, kernel_size=kernel_size, num_groups=num_groups,
            num_heads=num_heads, sr_ratio=sr_ratio,
            use_differential_attn=use_differential_attn)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                      act_cfg=act_cfg, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_mudd = use_mudd
        if use_mudd:
            if dim % 4 != 0:
                dim = (dim // 4) * 4
            self.mudd_connection = MUDDConnection(dim, num_paths=4)

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def _forward_impl(self, x, relative_pos_enc=None, previous_features=None):
        x = x + self.pos_embed(x)

        if self.use_mudd and previous_features is not None:
            valid_prev = [f for f in previous_features if f is not None]
            if valid_prev:
                x = self.mudd_connection(x, valid_prev)

        x = x + self.drop_path(
            self.layer_scale_1(self.token_mixer(self.norm1(x), relative_pos_enc)))
        x = x + self.drop_path(
            self.layer_scale_2(self.mlp(self.norm2(x))))
        return x

    def forward(self, x, relative_pos_enc=None, previous_features=None):
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(
                self._forward_impl, x, relative_pos_enc, previous_features)
        else:
            x = self._forward_impl(x, relative_pos_enc, previous_features)
        return x


# ══════════════════════════════════════════════════════════════════════
# Stage Builder
# ══════════════════════════════════════════════════════════════════════

def basic_blocks(dim, index, layers, kernel_size=3, num_groups=2,
                 num_heads=1, sr_ratio=1, mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'), drop_rate=0, drop_path_rate=0,
                 layer_scale_init_value=1e-5, grad_checkpoint=False,
                 use_mudd=True, use_differential_attn=False):
    """Build a list of Block modules for one stage."""
    blocks = nn.ModuleList()
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)

        block_use_mudd = use_mudd and (index >= 2)
        block_use_da = use_differential_attn and (index >= 2)

        blocks.append(Block(
            dim, kernel_size=kernel_size, num_groups=num_groups,
            num_heads=num_heads, sr_ratio=sr_ratio, mlp_ratio=mlp_ratio,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
            drop=drop_rate, drop_path=block_dpr,
            layer_scale_init_value=layer_scale_init_value,
            grad_checkpoint=grad_checkpoint,
            use_mudd=block_use_mudd,
            use_differential_attn=block_use_da,
        ))
    return blocks


# ══════════════════════════════════════════════════════════════════════
# TransXNet Backbone
# ══════════════════════════════════════════════════════════════════════

class TransXNet(nn.Module):
    """
    Enhanced TransXNet backbone with MUDD and Differential Attention.

    Key parameters:
      - arch: Architecture preset ('t'/'s'/'b') or custom dict.
      - use_mudd: Enable Multiway Dynamic Dense Connection.
      - use_differential_attn: Enable Differential Attention in deep stages.
      - da_stages: Stage indices where DA is applied (e.g., [3]).
    """

    arch_settings = {
        **dict.fromkeys(['t', 'tiny', 'T'], {
            'layers':      [3, 3, 9, 3],
            'embed_dims':  [48, 96, 224, 448],
            'kernel_size': [7, 7, 7, 7],
            'num_groups':  [2, 2, 2, 2],
            'sr_ratio':    [8, 4, 2, 1],
            'num_heads':   [1, 2, 4, 8],
            'mlp_ratios':  [4, 4, 4, 4],
            'layer_scale_init_value': 1e-5,
        }),
        **dict.fromkeys(['s', 'small', 'S'], {
            'layers':      [4, 4, 12, 4],
            'embed_dims':  [64, 128, 320, 512],
            'kernel_size': [7, 7, 7, 7],
            'num_groups':  [2, 2, 3, 4],
            'sr_ratio':    [8, 4, 2, 1],
            'num_heads':   [1, 2, 5, 8],
            'mlp_ratios':  [6, 6, 4, 4],
            'layer_scale_init_value': 1e-5,
        }),
        **dict.fromkeys(['b', 'base', 'B'], {
            'layers':      [4, 4, 21, 4],
            'embed_dims':  [76, 152, 336, 672],
            'kernel_size': [7, 7, 7, 7],
            'num_groups':  [2, 2, 4, 4],
            'sr_ratio':    [8, 4, 2, 1],
            'num_heads':   [2, 4, 8, 16],
            'mlp_ratios':  [8, 8, 4, 4],
            'layer_scale_init_value': 1e-5,
        }),
    }

    def __init__(self,
                 image_size=224,
                 arch='tiny',
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_chans=3,
                 in_patch_size=7, in_stride=4, in_pad=3,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0, drop_path_rate=0,
                 grad_checkpoint=False,
                 num_classes=1000,
                 fork_feat=False,
                 use_mudd=True,
                 use_differential_attn=True,
                 da_stages=None,    # e.g., [3] for stage 4 only
                 **kwargs):
        super().__init__()

        self.fork_feat = fork_feat
        if not fork_feat:
            self.num_classes = num_classes
        self.grad_checkpoint = grad_checkpoint
        self.use_mudd = use_mudd
        self.use_differential_attn = use_differential_attn

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f"Unknown arch '{arch}'. Choose from {set(self.arch_settings)}."
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        kernel_size = arch['kernel_size']
        num_groups = arch['num_groups']
        sr_ratio = arch['sr_ratio']
        num_heads = arch['num_heads']
        mlp_ratios = arch.get('mlp_ratios', [4, 4, 4, 4])
        layer_scale_init_value = arch.get('layer_scale_init_value', 1e-5)

        checkpoint_stage = [0] * 4

        # Stem
        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=in_chans, embed_dim=embed_dims[0])

        # Relative position encodings
        self.relative_pos_enc = []
        img_size = to_2tuple(image_size)
        img_size = [math.ceil(img_size[0] / in_stride),
                    math.ceil(img_size[1] / in_stride)]
        for i in range(4):
            num_patches = img_size[0] * img_size[1]
            sr_patches = (math.ceil(img_size[0] / sr_ratio[i]) *
                         math.ceil(img_size[1] / sr_ratio[i]))
            self.relative_pos_enc.append(
                nn.Parameter(torch.zeros(1, num_heads[i], num_patches, sr_patches),
                            requires_grad=True))
            img_size = [math.ceil(img_size[0] / 2),
                       math.ceil(img_size[1] / 2)]
        self.relative_pos_enc = nn.ParameterList(self.relative_pos_enc)

        # Build stages
        network = []
        for i in range(len(layers)):
            stage_use_mudd = use_mudd and (i >= 2)
            if da_stages is not None:
                stage_use_da = use_differential_attn and (i in da_stages)
            else:
                stage_use_da = use_differential_attn and (i >= 2)

            stage = basic_blocks(
                embed_dims[i], i, layers,
                kernel_size=kernel_size[i], num_groups=num_groups[i],
                num_heads=num_heads[i], sr_ratio=sr_ratio[i],
                mlp_ratio=mlp_ratios[i],
                norm_cfg=norm_cfg, act_cfg=act_cfg,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=checkpoint_stage[i],
                use_mudd=stage_use_mudd,
                use_differential_attn=stage_use_da,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                network.append(PatchEmbed(
                    patch_size=down_patch_size, stride=down_stride,
                    padding=down_pad,
                    in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]))
        self.network = nn.ModuleList(network)

        # Classification head
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                layer = build_norm_layer(norm_cfg, embed_dims[(i_layer + 1) // 2])[1]
                self.add_module(f'norm{i_layer}', layer)
        else:
            self.classifier = nn.Sequential(
                build_norm_layer(norm_cfg, embed_dims[-1])[1],
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1),
            ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_model_weights)

    def _init_model_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, MUDDConnection):
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode='fan_out')
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
        elif isinstance(m, DifferentialAttention):
            nn.init.trunc_normal_(m.qkv.weight, std=0.02)
            nn.init.constant_(m.lambda_param, 0.8)
            if hasattr(m, 'proj'):
                nn.init.trunc_normal_(m.proj.weight, std=0.02)
                if m.proj.bias is not None:
                    nn.init.zeros_(m.proj.bias)

    def forward_embeddings(self, x):
        return self.patch_embed(x)

    def forward_tokens(self, x):
        pos_idx = 0
        previous_features = [] if self.use_mudd else None

        for idx in range(len(self.network)):
            if idx in [0, 2, 4, 6]:
                if self.use_mudd:
                    previous_features = []

                for blk_idx, blk in enumerate(self.network[idx]):
                    blk_prev = None
                    if self.use_mudd and previous_features is not None:
                        blk_prev = previous_features.copy()

                    x = blk(x, self.relative_pos_enc[pos_idx], blk_prev)

                    if self.use_mudd:
                        if previous_features is None:
                            previous_features = []
                        if len(previous_features) >= 4:
                            previous_features.pop(0)
                        previous_features.append(x.clone())

                pos_idx += 1
            else:
                x = self.network[idx](x)

        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if not self.fork_feat:
            x = self.classifier(x).flatten(1)
        return x

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier[-1].out_channels = num_classes
        else:
            self.classifier = nn.Identity()


# ══════════════════════════════════════════════════════════════════════
# Factory Functions
# ══════════════════════════════════════════════════════════════════════

def create_transxnet_t(img_size=224, **kwargs):
    """Create TransXNet-Tiny with MUDD and Differential Attention."""
    defaults = dict(use_mudd=True, use_differential_attn=True)
    defaults.update(kwargs)
    return TransXNet(arch='t', image_size=img_size, **defaults)


def create_transxnet_s(img_size=224, **kwargs):
    """Create TransXNet-Small with MUDD and Differential Attention."""
    defaults = dict(use_mudd=True, use_differential_attn=True)
    defaults.update(kwargs)
    return TransXNet(arch='s', image_size=img_size, **defaults)


def create_transxnet_b(img_size=224, **kwargs):
    """Create TransXNet-Base with MUDD and Differential Attention."""
    defaults = dict(use_mudd=True, use_differential_attn=True)
    defaults.update(kwargs)
    return TransXNet(arch='b', image_size=img_size, **defaults)
