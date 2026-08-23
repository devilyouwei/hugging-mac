from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.models.qwen3_5 import QWEN3_5_DEFINITION, QWEN3_5_MANIFEST
from hugging_mac_sdk.models.qwen3_5.config import Qwen35MlxInstanceConfig
from hugging_mac_sdk.models.qwen3_5.mlx import _sanitize_vision_weights
from hugging_mac_sdk.models.qwen3_5.resources import Qwen35MlxResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_4b_optiq_artifact_owns_source_and_variant_metadata() -> None:
    variant = QWEN3_5_MANIFEST.get_variant("4b")

    source = QWEN3_5_DEFINITION.get_artifact("mlx", "mlx-optiq-4bit", "4b").source
    assert isinstance(source, HuggingFaceSource)
    assert source.repo_id == "mlx-community/Qwen3.5-4B-OptiQ-4bit"
    assert source.revision == "main"
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
    artifact = QWEN3_5_DEFINITION.get_artifact("mlx", "mlx-optiq-4bit", "4b")
    tokenizer = QWEN3_5_DEFINITION.shared_artifacts[0]
    for filename in artifact.required_files:
        path = model / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model-data")
    resolver = Qwen35MlxResourceResolver(
        HuggingFaceSource(
            repo_id="mlx-community/Qwen3.5-4B-OptiQ-4bit",
            revision="main",
        ),
        Qwen35MlxInstanceConfig(source_path=model, variant="4b"),
        manifest=QWEN3_5_MANIFEST,
        artifact=artifact,
        tokenizer_artifact=tokenizer,
    )

    assert resolver.status().artifacts[0].available
    (model / "optiq" / "optiq_vision.safetensors").unlink()
    assert not resolver.status().artifacts[0].available
