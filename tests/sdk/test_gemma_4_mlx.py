from __future__ import annotations

from pathlib import Path

import pytest
from hugging_mac_sdk.capabilities import Chat
from hugging_mac_sdk.models.gemma_4 import GEMMA_4_DEFINITION, GEMMA_4_MANIFEST
from hugging_mac_sdk.models.gemma_4.config import Gemma4MlxInstanceConfig
from hugging_mac_sdk.models.gemma_4.resources import Gemma4MlxResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_variants_are_canonical_and_sources_belong_to_artifacts() -> None:
    e4b = GEMMA_4_MANIFEST.get_variant("e4b")
    e2b = GEMMA_4_MANIFEST.get_variant("e2b")

    assert GEMMA_4_MANIFEST.model_id == "google/gemma-4"
    assert GEMMA_4_MANIFEST.default_variant == "e4b"
    assert GEMMA_4_MANIFEST.capabilities == frozenset({"chat"})
    assert GEMMA_4_DEFINITION.supported_runtimes == ("mlx",)
    assert tuple(item.name for item in GEMMA_4_MANIFEST.variants) == ("e4b", "e2b")
    e4b_source = GEMMA_4_DEFINITION.get_artifact("mlx", "mlx-optiq-4bit", "e4b").source
    e2b_source = GEMMA_4_DEFINITION.get_artifact("mlx", "mlx-optiq-4bit", "e2b").source
    assert isinstance(e4b_source, HuggingFaceSource)
    assert isinstance(e2b_source, HuggingFaceSource)
    assert e4b_source.repo_id == "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit"
    assert e4b_source.revision == "main"
    assert e2b_source.repo_id == "mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit"
    assert e2b_source.revision == "main"
    assert e4b.metadata["quantization_aware_trained"] is True
    assert e2b.metadata["quantization_aware_trained"] is True


@pytest.mark.parametrize("variant", ("e4b", "e2b"))
def test_snapshot_requires_every_model_file_and_vision_sidecar(
    tmp_path: Path, variant: str
) -> None:
    model = tmp_path / variant
    model.mkdir()
    artifact = GEMMA_4_DEFINITION.get_artifact("mlx", "mlx-optiq-4bit", variant)
    tokenizer = GEMMA_4_DEFINITION.shared_artifacts[0]
    for filename in artifact.required_files:
        path = model / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model-data")
    resolver = Gemma4MlxResourceResolver(
        HuggingFaceSource(repo_id=f"test/{variant}", revision="a" * 40),
        Gemma4MlxInstanceConfig(source_path=model, variant=variant),
        manifest=GEMMA_4_MANIFEST,
        artifact=artifact,
        tokenizer_artifact=tokenizer,
    )

    assert resolver.status().artifacts[0].available
    (model / "optiq" / "optiq_vision.safetensors").unlink()
    assert not resolver.status().artifacts[0].available


def test_runtime_instance_exposes_the_shared_chat_capability() -> None:
    instance = GEMMA_4_DEFINITION.create({"variant": "e2b"}, runtime="mlx")

    assert instance.supports(Chat)
    assert instance.require(Chat) is instance
