from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar

import pytest
from hugging_mac_sdk.capabilities import Chat
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.models.qwen3_5_9b_mlx_4bit import (
    QWEN3_5_9B_MLX_4BIT_DEFINITION,
    QWEN3_5_9B_MLX_4BIT_MANIFEST,
)
from hugging_mac_sdk.models.qwen3_5_9b_mlx_4bit.config import (
    QWEN3_5_9B_MLX_4BIT_REVISION,
    Qwen35MlxInstanceConfig,
)
from hugging_mac_sdk.models.qwen3_5_9b_mlx_4bit.instance import Qwen35MlxInstance
from hugging_mac_sdk.models.qwen3_5_9b_mlx_4bit.mlx import _IncrementalTokenDecoder
from hugging_mac_sdk.models.qwen3_5_9b_mlx_4bit.resources import Qwen35MlxResourceResolver
from hugging_mac_sdk.models.qwen3_5_9b_mlx_4bit.utils.types import GenerationOutput
from hugging_mac_sdk.schemas.chat import ChatMessage, ChatRequest


class ByteTokenizer:
    all_special_ids = (0,)
    pieces: ClassVar[dict[int, bytes]] = {
        1: b"\xe4",
        2: b"\xbd",
        3: b"\xa0",
        4: b"\xe5",
        5: b"\xa5",
        6: b"\xbd",
        7: b" ",
        8: b"O",
        9: b"K",
    }

    def decode(self, tokens: list[int], **_: object) -> str:
        return b"".join(self.pieces[token] for token in tokens).decode("utf-8", errors="replace")


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
    assert QWEN3_5_9B_MLX_4BIT_MANIFEST.revision == QWEN3_5_9B_MLX_4BIT_REVISION
    assert QWEN3_5_9B_MLX_4BIT_MANIFEST.capabilities == {
        "chat",
        "vision-language-generation",
    }
    assert QWEN3_5_9B_MLX_4BIT_DEFINITION.supported_runtimes == ("mlx",)


def test_incremental_decoder_flushes_chinese_without_spaces() -> None:
    decoder = _IncrementalTokenDecoder(ByteTokenizer())

    chunks = [decoder.add(token) for token in (1, 2, 3, 4, 5, 6)]

    assert chunks == ["", "", "你", "", "", "好"]
    assert "".join(chunks) == "你好"


def test_incremental_decoder_keeps_english_streaming_and_skips_special_tokens() -> None:
    decoder = _IncrementalTokenDecoder(ByteTokenizer())

    chunks = [decoder.add(token) for token in (0, 7, 8, 9)]

    assert chunks == ["", " ", "O", "K"]


async def test_instance_lifecycle_and_chat(tmp_path: Path) -> None:
    engine = FakeEngine(tmp_path)
    instance = Qwen35MlxInstance(Qwen35MlxInstanceConfig(source_path=tmp_path), engine)

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
        Qwen35MlxInstanceConfig.model_validate({"unknown": True})


def test_downloaded_snapshot_is_reported_available(tmp_path: Path) -> None:
    from hugging_mac_sdk.models.qwen3_5_9b_mlx_4bit.config import (
        QWEN3_5_9B_MLX_4BIT_REQUIRED_FILES,
    )
    from hugging_mac_sdk.schemas.resources import HuggingFaceSource

    model = tmp_path / "model"
    model.mkdir()
    for filename in QWEN3_5_9B_MLX_4BIT_REQUIRED_FILES:
        (model / filename).write_bytes(b"model-data")
    resolver = Qwen35MlxResourceResolver(
        HuggingFaceSource(repo_id="mlx-community/Qwen3.5-9B-MLX-4bit"),
        Qwen35MlxInstanceConfig(source_path=model),
    )

    status = resolver.status()

    assert status.artifacts[0].artifact_id == "model"
    assert status.artifacts[0].available
    assert status.runtimes[0].available
    assert status.total_size_bytes > 0
