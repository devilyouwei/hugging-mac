from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.models.qwen3_5 import QWEN3_5_DEFINITION, QWEN3_5_MANIFEST
from hugging_mac_sdk.models.qwen3_5.config import Qwen35MlxInstanceConfig
from hugging_mac_sdk.models.qwen3_5.resources import Qwen35MlxResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_9b_artifact_owns_default_optiq_source() -> None:
    variant = QWEN3_5_MANIFEST.get_variant("9b")

    assert QWEN3_5_DEFINITION.supported_runtimes == ("mlx",)
    assert QWEN3_5_MANIFEST.default_variant == "9b"
    source = QWEN3_5_DEFINITION.get_artifact("mlx", "mlx-optiq-4bit", "9b").source
    assert isinstance(source, HuggingFaceSource)
    assert source.repo_id == "mlx-community/Qwen3.5-9B-OptiQ-4bit"
    assert source.revision == "main"
    assert variant.metadata["quantization_method"] == "optiq"
    assert variant.metadata["mixed_precision_8bit_layers"] == 132
    assert variant.metadata["mixed_precision_4bit_layers"] == 116


def test_9b_snapshot_is_available_only_when_its_required_files_exist(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    artifact = QWEN3_5_DEFINITION.get_artifact("mlx", "mlx-optiq-4bit", "9b")
    tokenizer = QWEN3_5_DEFINITION.shared_artifacts[0]
    for filename in artifact.required_files:
        path = model / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model-data")

    resolver = Qwen35MlxResourceResolver(
        HuggingFaceSource(
            repo_id="mlx-community/Qwen3.5-9B-OptiQ-4bit",
            revision="main",
        ),
        Qwen35MlxInstanceConfig(source_path=model, variant="9b"),
        manifest=QWEN3_5_MANIFEST,
        artifact=artifact,
        tokenizer_artifact=tokenizer,
    )

    assert resolver.status().artifacts[0].available
    (model / "optiq" / "optiq_vision.safetensors").unlink()
    assert not resolver.status().artifacts[0].available
