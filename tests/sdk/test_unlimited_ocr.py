from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from hugging_mac_sdk.capabilities import DocumentParsing
from hugging_mac_sdk.models.unlimited_ocr import (
    UNLIMITED_OCR_DEFINITION,
    UNLIMITED_OCR_MANIFEST,
)
from hugging_mac_sdk.models.unlimited_ocr.config import UnlimitedOcrMlxInstanceConfig
from hugging_mac_sdk.models.unlimited_ocr.instance import UnlimitedOcrInstance
from hugging_mac_sdk.models.unlimited_ocr.mlx import MlxUnlimitedOcrEngine
from hugging_mac_sdk.schemas.document_parsing import DocumentImage, DocumentParsingRequest
from hugging_mac_sdk.schemas.resources import HuggingFaceSource
from pydantic import ValidationError


def test_manifest_declares_mlx_community_checkpoint() -> None:
    artifact = UNLIMITED_OCR_DEFINITION.get_artifact("mlx", "mlx-4bit", "4bit")

    assert UNLIMITED_OCR_MANIFEST.model_id == "baidu/unlimited-ocr"
    assert UNLIMITED_OCR_DEFINITION.supported_runtimes == ("mlx",)
    assert UNLIMITED_OCR_MANIFEST.capabilities == frozenset({"document-parsing"})
    assert isinstance(artifact.source, HuggingFaceSource)
    assert artifact.source.repo_id == "mlx-community/Unlimited-OCR-4bit"
    assert Path("model.safetensors") in artifact.required_files
    assert Path("modeling_unlimitedocr.py") not in artifact.required_files


async def test_instance_exposes_document_parsing(tmp_path: Path) -> None:
    class Engine:
        runtime_name = "mlx"
        device = "gpu"

        async def resolve(self) -> Path:
            return tmp_path

        async def load(self, artifact: Path) -> None:
            assert artifact == tmp_path

        async def infer(self, request: DocumentParsingRequest) -> str:
            assert request.images[0].path == tmp_path / "page.png"
            return "# Parsed document"

        async def stream(self, request: DocumentParsingRequest):
            assert request.images[0].path == tmp_path / "page.png"
            for delta in ("# Parsed", " document"):
                yield delta

        async def close(self) -> None:
            pass

    instance = UnlimitedOcrInstance(
        UnlimitedOcrMlxInstanceConfig(artifact_path=tmp_path),
        Engine(),  # type: ignore[arg-type]
        UNLIMITED_OCR_MANIFEST,
    )
    await instance.load()

    parser = instance.require(DocumentParsing)
    response = await parser.parse_document(
        DocumentParsingRequest(images=(DocumentImage(path=tmp_path / "page.png"),))
    )

    assert response.text == "# Parsed document"
    assert response.page_count == 1
    events = [
        event
        async for event in parser.stream_document(
            DocumentParsingRequest(images=(DocumentImage(path=tmp_path / "page.png"),))
        )
    ]
    assert "".join(event.delta for event in events) == "# Parsed document"
    assert events[-1].finish_reason == "stop"
    await instance.unload()


async def test_mlx_engine_selects_internal_image_and_repetition_settings(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []
    mlx_vlm = ModuleType("mlx_vlm")
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")

    def apply_chat_template(
        processor: object, config: object, prompt: str, *, num_images: int
    ) -> str:
        assert processor is engine._processor  # type: ignore[attr-defined]
        assert config == {"model_type": "unlimited-ocr"}
        return f"<image>{prompt}:{num_images}"

    def generate(
        model: object, processor: object, prompt: str, images: list[str], **kwargs: object
    ) -> object:
        calls.append({"prompt": prompt, "images": images, **kwargs})
        return type("Result", (), {"text": "parsed"})()

    def stream_generate(
        model: object, processor: object, prompt: str, images: list[str], **kwargs: object
    ):
        calls.append({"prompt": prompt, "images": images, **kwargs})
        yield type("Result", (), {"token": 1})()
        yield type("Result", (), {"token": 2})()

    class Processor:
        tokenizer = None

        def __init__(self) -> None:
            self.tokenizer = self
            self.all_special_ids: set[int] = set()

        def decode(self, tokens: list[int], **_kwargs: object) -> str:
            return "".join({1: "par", 2: "sed"}[token] for token in tokens)

    mlx_vlm.generate = generate  # type: ignore[attr-defined]
    mlx_vlm.stream_generate = stream_generate  # type: ignore[attr-defined]
    prompt_utils.apply_chat_template = apply_chat_template  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    engine = object.__new__(MlxUnlimitedOcrEngine)
    engine._config = UnlimitedOcrMlxInstanceConfig(artifact_path=tmp_path)  # type: ignore[attr-defined]
    engine._model = type("Model", (), {"config": {"model_type": "unlimited-ocr"}})()  # type: ignore[attr-defined]
    engine._processor = Processor()  # type: ignore[attr-defined]
    monkeypatch.setattr(
        MlxUnlimitedOcrEngine,
        "_logits_processors",
        staticmethod(lambda *, window: [window]),
    )

    single = DocumentParsingRequest(images=(DocumentImage(path=tmp_path / "one.png"),))
    assert await engine.infer(single) == "parsed"
    assert calls[-1]["cropping"] is True
    assert calls[-1]["image_size"] == 640
    assert calls[-1]["base_size"] == 1024
    assert calls[-1]["logits_processors"] == [128]

    multi = DocumentParsingRequest(
        images=(DocumentImage(path=tmp_path / "one.png"), DocumentImage(path=tmp_path / "two.png")),
    )
    assert await engine.infer(multi) == "parsed"
    assert calls[-1]["cropping"] is False
    assert calls[-1]["image_size"] == 1024
    assert calls[-1]["logits_processors"] == [1024]
    assert calls[-1]["images"] == [str(tmp_path / "one.png"), str(tmp_path / "two.png")]

    assert "".join([delta async for delta in engine.stream(single)]) == "parsed"
    assert calls[-1]["cropping"] is True
    assert calls[-1]["verbose"] is False


def test_document_parsing_request_rejects_internal_runtime_options(tmp_path: Path) -> None:
    image = (DocumentImage(path=tmp_path / "page.png"),)

    for option in (
        {"image_mode": "base"},
        {"no_repeat_ngram_size": 0},
        {"ngram_window": 4096},
    ):
        with pytest.raises(ValidationError):
            DocumentParsingRequest(images=image, **option)  # type: ignore[arg-type]
