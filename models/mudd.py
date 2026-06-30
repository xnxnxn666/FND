"""
Multiway Dynamic Dense Connection (MUDD)

A lightweight cross-layer feature aggregation mechanism that dynamically
routes information from all previous layers to the current layer through
multiple parallel pathways.

Key ideas:
1. **Multi-path decomposition**: Channel dimension is split into N paths,
   each processed independently with depthwise-separable convolutions.
2. **Dynamic routing**: A lightweight weight generator (avgpool + 2-layer
   conv) produces a N x N softmax matrix, adaptively weighting how each
   previous layer's path contributes to each current path.
3. **Dense connectivity**: Every previous layer can contribute to every
   current path, weighted by the learned dynamic matrix.
"""

import os
import torch
from torch import nn
from torch.nn import functional as F


class MUDDConnection(nn.Module):
    """
    MUltiway Dynamic Dense Connection.

    Args:
        dim: Input channel dimension.
        num_paths: Number of parallel pathways (default 4).
        reduction_ratio: Channel reduction ratio for the weight generator.
    """

    def __init__(self, dim, num_paths=4, reduction_ratio=8):
        super().__init__()
        self.dim = dim
        self.num_paths = num_paths
        assert dim % num_paths == 0, \
            f"dim {dim} must be divisible by num_paths {num_paths}"
        self.path_dims = dim // num_paths
        self.reduction_ratio = reduction_ratio

        # Dynamic weight generator — lightweight design
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, num_paths * num_paths, 1),
        )

        # Path-specific transforms using depthwise-separable convolutions
        self.path_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.path_dims, self.path_dims, 3,
                          padding=1, groups=self.path_dims),
                nn.GELU(),
                nn.Conv2d(self.path_dims, self.path_dims, 1),
            ) for _ in range(num_paths)
        ])

        # Spatial adapters for handling resolution mismatches
        self.spatial_adapter = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.path_dims, self.path_dims, 1),
                nn.GELU(),
            ) for _ in range(num_paths)
        ])

        # Channel adapter for merging multi-path outputs
        self.channel_adapter = nn.Conv2d(dim, dim, 1)

        if os.environ.get("TRANSXNET_DEBUG", "").lower() in ("1", "true", "yes"):
            print(f"MUDD Connection: dim={dim}, paths={num_paths}, "
                  f"path_dims={self.path_dims}")

    def _adapt_spatial_size(self, x, target_size):
        """Adjust feature map spatial size to match target."""
        if x.size(-2) != target_size[-2] or x.size(-1) != target_size[-1]:
            x = F.interpolate(x, size=target_size,
                             mode='bilinear', align_corners=False)
        return x

    def forward(self, current_x, previous_features):
        """
        Args:
            current_x: Current layer features [B, C, H, W].
            previous_features: List of features from all previous layers
                               [[B, C, H, W], ...].
        Returns:
            Aggregated features [B, C, H, W].
        """
        if not previous_features:
            return current_x

        B, C, H, W = current_x.shape
        target_spatial_size = (H, W)

        # Generate dynamic connection weights [B, num_paths, num_paths]
        weight_logits = self.weight_generator(current_x)
        weight_matrix = torch.softmax(
            weight_logits.view(B, self.num_paths, self.num_paths), dim=-1)

        # Split current features into paths
        current_paths = torch.split(current_x, self.path_dims, dim=1)

        # Process previous layer features, ensuring size compatibility
        previous_paths_list = []
        for prev_feat in previous_features:
            prev_feat_adapted = self._adapt_spatial_size(
                prev_feat, target_spatial_size)
            prev_paths = torch.split(prev_feat_adapted, self.path_dims, dim=1)
            if len(prev_paths) < self.num_paths:
                prev_paths = list(prev_paths)
                while len(prev_paths) < self.num_paths:
                    prev_paths.append(torch.zeros_like(prev_paths[0]))
            previous_paths_list.append(prev_paths)

        # Multi-path dynamic aggregation
        output_paths = []
        for path_idx in range(self.num_paths):
            if path_idx >= len(current_paths):
                continue

            path_output = self.path_transforms[path_idx](
                current_paths[path_idx])

            # Aggregate corresponding paths from all previous layers
            for layer_idx, prev_paths in enumerate(previous_paths_list):
                if path_idx < len(prev_paths) and prev_paths[path_idx] is not None:
                    weight = weight_matrix[
                        :, path_idx, layer_idx % self.num_paths
                    ].view(B, 1, 1, 1)
                    weighted_prev = prev_paths[path_idx] * weight

                    if path_output.shape[-2:] != weighted_prev.shape[-2:]:
                        weighted_prev = F.interpolate(
                            weighted_prev, size=path_output.shape[-2:],
                            mode='bilinear', align_corners=False)

                    path_output = path_output + weighted_prev

            output_paths.append(path_output)

        # Merge paths
        if output_paths:
            output = torch.cat(output_paths, dim=1)
            if output.size(1) != current_x.size(1):
                output = self.channel_adapter(output)
        else:
            output = current_x

        return output
