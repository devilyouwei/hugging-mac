"""D3 vision export preserving upstream patch grids and interpolation."""

from typing import Any

import torch
from torch.nn import functional as F

from .preprocess import position_inputs
from .types import ExportGraph


class VisionGraph(torch.nn.Module):
    def __init__(self, visual: Any) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, pixels: Any, indices: Any, weights: Any, positions: Any) -> Any:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

        v = self.visual
        hidden = F.linear(
            pixels.to(v.patch_embed.proj.weight.dtype),
            v.patch_embed.proj.weight.flatten(1),
            v.patch_embed.proj.bias,
        )
        hidden = hidden + (v.pos_embed(indices) * weights[:, :, None]).sum(1).to(hidden.dtype)
        cos, sin = v.rotary_pos_emb(hidden, positions)
        for block in v.blocks:
            x = block.norm1(hidden)
            attn = block.attn
            q, k, values = (
                attn.qkv(x).reshape(x.shape[0], 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            y = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                values.transpose(0, 1).unsqueeze(0),
                scale=attn.scaling,
            )
            hidden = hidden + attn.proj(y.squeeze(0).transpose(0, 1).reshape(x.shape[0], -1))
            hidden = hidden + block.mlp(block.norm2(hidden))
        return v.merger(hidden).float()


def prepare_vision(visual: Any) -> ExportGraph:
    grid = torch.tensor([[1, 16, 20]])
    patches = int(grid.prod())
    width = 3 * visual.config.temporal_patch_size * visual.config.patch_size**2
    args = (torch.zeros(patches, width), *position_inputs(visual, grid))
    patches_dim = visual.config.spatial_merge_size**2 * torch.export.Dim("vision_cells", min=1)
    return ExportGraph(
        VisionGraph(visual).eval(),
        args,
        ("pixels", "indices", "weights", "positions"),
        "features",
        tuple({0: patches_dim} for _ in args),
    )
