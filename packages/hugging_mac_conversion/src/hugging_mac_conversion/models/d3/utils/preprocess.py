"""Upstream grid-dependent positional inputs; no media resizing overrides."""

from typing import Any


def position_inputs(visual: Any, grid: Any) -> tuple[Any, Any, Any]:
    import torch
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    indices, weights = get_vision_interpolation_indices_and_weights(
        grid,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size,
        kwargs={},
    )
    positions = get_vision_position_ids(grid, visual.spatial_merge_size, kwargs={})
    return indices.to(torch.int32), weights.float(), positions.to(torch.int32)
