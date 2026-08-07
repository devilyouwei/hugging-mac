from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.models.qwen3_5_mlx import QWEN3_5_MLX_DEFINITION, QWEN3_5_MLX_MANIFEST
from hugging_mac_sdk.models.qwen3_5_mlx.config import (
    QWEN3_5_MLX_REQUIRED_FILES,
    QWEN3_5_MLX_VARIANTS,
    Qwen35MlxInstanceConfig,
)
from hugging_mac_sdk.models.qwen3_5_mlx.resources import Qwen35MlxResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_2b_optiq_variant_is_pinned_with_its_mixed_precision_metadata() -> None:
    variant = QWEN3_5_MLX_MANIFEST.get_variant("2b-optiq-4bit")

    assert QWEN3_5_MLX_DEFINITION.supported_runtimes == ("mlx",)
    assert QWEN3_5_MLX_VARIANTS == ("9b-4bit", "4b-optiq-4bit", "2b-optiq-4bit")
    assert variant.resources[0].repo_id == "mlx-community/Qwen3.5-2B-OptiQ-4bit"
    assert variant.resources[0].revision == "adc8669eb431e3168aeb4e320bd7b757914350e2"
    assert variant.metadata == {
        "parameter_count_billions": 2,
        "quantization_bits": 4,
        "quantization_group_size": 64,
        "disk_size_bytes": 1_400_000_000,
        "multimodal": True,
        "quantization_method": "optiq",
        "mixed_precision_8bit_layers": 56,
        "mixed_precision_4bit_layers": 130,
    }


def test_2b_optiq_snapshot_requires_the_visual_sidecar(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for filename in QWEN3_5_MLX_REQUIRED_FILES["2b-optiq-4bit"]:
        path = model / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model-data")

    resolver = Qwen35MlxResourceResolver(
        HuggingFaceSource(
            repo_id="mlx-community/Qwen3.5-2B-OptiQ-4bit",
            revision="adc8669eb431e3168aeb4e320bd7b757914350e2",
        ),
        Qwen35MlxInstanceConfig(source_path=model, variant="2b-optiq-4bit"),
    )

    assert resolver.status().artifacts[0].available
    (model / "optiq" / "optiq_vision.safetensors").unlink()
    assert not resolver.status().artifacts[0].available
