from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from hugging_mac_sdk.capabilities import Chat
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.models.qwen3_5_4b_optiq_4bit import (
    QWEN3_5_4B_OPTIQ_4BIT_DEFINITION,
    QWEN3_5_4B_OPTIQ_4BIT_MANIFEST,
)
from hugging_mac_sdk.models.qwen3_5_4b_optiq_4bit.config import (
    QWEN3_5_4B_OPTIQ_4BIT_REVISION,
    Qwen35OptiQMlxInstanceConfig,
)
from hugging_mac_sdk.models.qwen3_5_4b_optiq_4bit.instance import Qwen35OptiQMlxInstance
from hugging_mac_sdk.models.qwen3_5_4b_optiq_4bit.mlx import _sanitize_vision_weights
from hugging_mac_sdk.models.qwen3_5_4b_optiq_4bit.resources import Qwen35OptiQMlxResourceResolver
from hugging_mac_sdk.models.qwen3_5_4b_optiq_4bit.utils.types import GenerationOutput
from hugging_mac_sdk.schemas.chat import ChatMessage, ChatRequest


class FakeEngine:
    runtime_name = "mlx"
    device = "gpu"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.loaded = False
        self.closed = False

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact
        self.loaded = True

    async def infer(self, request: ChatRequest) -> GenerationOutput:
        assert request.messages[-1].content == "你好"
        return GenerationOutput("你好!", prompt_tokens=3, generated_tokens=2)

    async def stream(self, request: ChatRequest) -> AsyncIterator[GenerationOutput]:
        assert request.messages[-1].content == "你好"
        yield GenerationOutput("你", prompt_tokens=3, generated_tokens=1)
        yield GenerationOutput("好!", prompt_tokens=3, generated_tokens=2)

    async def close(self) -> None:
        self.closed = True


def test_manifest_is_pinned_and_declares_mlx_chat() -> None:
    assert QWEN3_5_4B_OPTIQ_4BIT_MANIFEST.revision == QWEN3_5_4B_OPTIQ_4BIT_REVISION
    assert QWEN3_5_4B_OPTIQ_4BIT_MANIFEST.capabilities == {
        "chat",
        "vision-language-generation",
    }
    assert QWEN3_5_4B_OPTIQ_4BIT_MANIFEST.runtimes[0].required_modules == (
        "mlx",
        "mlx_vlm",
    )
    assert QWEN3_5_4B_OPTIQ_4BIT_DEFINITION.supported_runtimes == ("mlx",)


def test_optiq_vision_sidecar_is_sanitized_and_scoped_to_visual_tower() -> None:
    class FakeModel:
        @staticmethod
        def sanitize(weights: dict[str, object]) -> dict[str, object]:
            return {
                key.replace("model.visual", "vision_tower", 1): value
                for key, value in weights.items()
            }

    weights = _sanitize_vision_weights(
        FakeModel(),
        {
            "model.visual.blocks.0.weight": "vision",
            "model.language_model.layers.0.weight": "language",
        },
    )

    assert weights == {"vision_tower.blocks.0.weight": "vision"}


async def test_instance_lifecycle_and_chat(tmp_path: Path) -> None:
    engine = FakeEngine(tmp_path)
    instance = Qwen35OptiQMlxInstance(Qwen35OptiQMlxInstanceConfig(source_path=tmp_path), engine)

    assert instance.supports(Chat)
    await instance.load()
    assert instance.state is ModelState.READY
    response = await instance.chat(
        ChatRequest(messages=(ChatMessage(role="user", content="你好"),))
    )
    assert response.message.content == "你好!"
    assert response.generated_tokens == 2

    events = [
        event
        async for event in instance.stream_chat(
            ChatRequest(messages=(ChatMessage(role="user", content="你好"),))
        )
    ]
    assert "".join(event.delta for event in events) == "你好!"
    assert events[-1].finish_reason == "stop"
    await instance.unload()
    assert engine.closed


def test_instance_options_reject_unknown_values() -> None:
    with pytest.raises(ValueError):
        Qwen35OptiQMlxInstanceConfig.model_validate({"unknown": True})


def test_downloaded_snapshot_is_reported_available(tmp_path: Path) -> None:
    from hugging_mac_sdk.models.qwen3_5_4b_optiq_4bit.config import (
        QWEN3_5_4B_OPTIQ_4BIT_REQUIRED_FILES,
    )
    from hugging_mac_sdk.schemas.resources import HuggingFaceSource

    model = tmp_path / "model"
    model.mkdir()
    for filename in QWEN3_5_4B_OPTIQ_4BIT_REQUIRED_FILES:
        (model / filename).parent.mkdir(parents=True, exist_ok=True)
        (model / filename).write_bytes(b"model-data")
    resolver = Qwen35OptiQMlxResourceResolver(
        HuggingFaceSource(repo_id="mlx-community/Qwen3.5-4B-OptiQ-4bit"),
        Qwen35OptiQMlxInstanceConfig(source_path=model),
    )

    status = resolver.status()

    assert status.artifacts[0].artifact_id == "model"
    assert status.artifacts[0].available
    assert status.runtimes[0].available
    assert status.total_size_bytes > 0
