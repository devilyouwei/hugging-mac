from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.kokoro import KOKORO_82M_MANIFEST, register_kokoro
from hugging_mac_sdk.models.kokoro.config import Kokoro82mInstanceConfig
from hugging_mac_sdk.models.kokoro.instance import Kokoro82mInstance
from hugging_mac_sdk.models.kokoro.resources import Kokoro82mResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

PYTORCH_SOURCE = HuggingFaceSource(
    repo_id="hexgrad/Kokoro-82M",
    revision="f3ff3571791e39611d31c381e3a41a3af07b4987",
)
COREML_SOURCE = HuggingFaceSource(
    repo_id="aufklarer/Kokoro-82M-CoreML",
    revision="f8ff771e4cab0bb3368e8af3a090a7e847485401",
)


def test_kokoro_manifest_registers_downloadable_coreml_runtime() -> None:
    assert KOKORO_82M_MANIFEST.model_id == "hexgrad/kokoro"
    assert KOKORO_82M_MANIFEST.default_variant == "v1.0"
    assert KOKORO_82M_MANIFEST.default_runtime == "coreml"
    assert {runtime.name for runtime in KOKORO_82M_MANIFEST.runtimes} == {
        "coreml",
        "pytorch-mps",
    }
    models = ModelRegistry()
    definition = register_kokoro(models)
    assert definition.converter_ids == ()
    for runtime in ("coreml", "pytorch-mps"):
        instance = definition.create(runtime=runtime, variant="v1.0")
        assert isinstance(instance, Kokoro82mInstance)
        assert instance.supports(SpeechSynthesis)  # type: ignore[type-abstract]


async def test_kokoro_resolves_downloaded_coreml_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "coreml"
    for relative in (
        "kokoro_5s.mlmodelc/model.mil",
        "kokoro_5s.mlmodelc/weights/weight.bin",
        "G2PEncoder.mlmodelc/model.mil",
        "G2PDecoder.mlmodelc/model.mil",
        "vocab_index.json",
        "voices/af_heart.json",
    ):
        target = artifact / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test")
    config = Kokoro82mInstanceConfig(artifact_path=artifact)
    resolver = Kokoro82mResourceResolver(PYTORCH_SOURCE, COREML_SOURCE, config)
    resolved = await resolver.resolve_coreml()
    assert resolved.path == artifact
    assert resolved.source == COREML_SOURCE
    status = resolver.status()
    coreml = next(item for item in status.artifacts if item.artifact_id == "coreml")
    assert coreml.provisioning == "download"
    assert status.conversion_targets == ()
