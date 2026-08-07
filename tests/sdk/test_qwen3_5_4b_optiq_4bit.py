from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.models.qwen3_5_mlx import QWEN3_5_MLX_MANIFEST
from hugging_mac_sdk.models.qwen3_5_mlx.config import (
    QWEN3_5_MLX_REQUIRED_FILES,
    Qwen35MlxInstanceConfig,
)
from hugging_mac_sdk.models.qwen3_5_mlx.mlx import _sanitize_vision_weights
from hugging_mac_sdk.models.qwen3_5_mlx.resources import Qwen35MlxResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_4b_optiq_variant_remains_pinned_with_mixed_precision_metadata() -> None:
    variant = QWEN3_5_MLX_MANIFEST.get_variant("4b-optiq-4bit")

    assert variant.resources[0].repo_id == "mlx-community/Qwen3.5-4B-OptiQ-4bit"
    assert variant.resources[0].revision == "6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9"
    assert variant.metadata["quantization_method"] == "optiq"
    assert variant.metadata["mixed_precision_8bit_layers"] == 75
    assert variant.metadata["mixed_precision_4bit_layers"] == 173


def test_optiq_vision_sidecar_is_sanitized_and_required(tmp_path: Path) -> None:
    class FakeModel:
        @staticmethod
        def sanitize(weights: dict[str, object]) -> dict[str, object]:
            return {
                key.replace("model.visual", "vision_tower", 1): value
                for key, value in weights.items()
            }

    assert _sanitize_vision_weights(
        FakeModel(),
        {
            "model.visual.blocks.0.weight": "vision",
            "model.language_model.layers.0.weight": "language",
        },
    ) == {"vision_tower.blocks.0.weight": "vision"}

    model = tmp_path / "model"
    model.mkdir()
    for filename in QWEN3_5_MLX_REQUIRED_FILES["4b-optiq-4bit"]:
        path = model / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model-data")
    resolver = Qwen35MlxResourceResolver(
        HuggingFaceSource(
            repo_id="mlx-community/Qwen3.5-4B-OptiQ-4bit",
            revision="6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9",
        ),
        Qwen35MlxInstanceConfig(source_path=model, variant="4b-optiq-4bit"),
    )

    assert resolver.status().artifacts[0].available
    (model / "optiq" / "optiq_vision.safetensors").unlink()
    assert not resolver.status().artifacts[0].available
