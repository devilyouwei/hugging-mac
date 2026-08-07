from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.models.qwen3_5_mlx import QWEN3_5_MLX_DEFINITION, QWEN3_5_MLX_MANIFEST
from hugging_mac_sdk.models.qwen3_5_mlx.config import (
    QWEN3_5_MLX_REQUIRED_FILES,
    Qwen35MlxInstanceConfig,
)
from hugging_mac_sdk.models.qwen3_5_mlx.resources import Qwen35MlxResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_9b_variant_remains_the_default_standard_mlx_checkpoint() -> None:
    variant = QWEN3_5_MLX_MANIFEST.get_variant("9b-4bit")

    assert QWEN3_5_MLX_DEFINITION.supported_runtimes == ("mlx",)
    assert QWEN3_5_MLX_MANIFEST.default_variant == "9b-4bit"
    assert variant.resources[0].repo_id == "mlx-community/Qwen3.5-9B-MLX-4bit"
    assert variant.resources[0].revision == "938d8919941c6e7efd3c7150eff7fe9d12afa631"


def test_9b_snapshot_is_available_only_when_its_required_files_exist(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for filename in QWEN3_5_MLX_REQUIRED_FILES["9b-4bit"]:
        (model / filename).write_bytes(b"model-data")

    resolver = Qwen35MlxResourceResolver(
        HuggingFaceSource(
            repo_id="mlx-community/Qwen3.5-9B-MLX-4bit",
            revision="938d8919941c6e7efd3c7150eff7fe9d12afa631",
        ),
        Qwen35MlxInstanceConfig(source_path=model, variant="9b-4bit"),
    )

    assert resolver.status().artifacts[0].available
    (model / "tokenizer.json").unlink()
    assert not resolver.status().artifacts[0].available
