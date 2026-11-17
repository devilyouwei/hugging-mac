"""Upstream grid-dependent positional inputs; no media resizing overrides."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hugging_mac_sdk.schemas.decision import DecisionRequest
    from hugging_mac_sdk.schemas.detection import ImageInput


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


def prepare_request(request: "DecisionRequest") -> dict[str, Any]:
    """Map SDK media values to the upstream host processor without framework tensors."""
    from io import BytesIO
    from pathlib import Path

    from PIL import Image

    def decode(image: "ImageInput") -> Any:
        source = image.path if image.path is not None else BytesIO(image.data or b"")
        with Image.open(source) as value:
            return value.convert("RGB")

    return {
        "state": request.state,
        "questions": {
            key: question.model_dump(mode="json", exclude_none=True)
            for key, question in request.questions.items()
        },
        "images": [
            str(image.path) if image.path is not None else decode(image) for image in request.images
        ]
        or None,
        "videos": [
            str(video) if isinstance(video, Path) else [decode(frame) for frame in video.frames]
            for video in request.videos
        ]
        or None,
    }
