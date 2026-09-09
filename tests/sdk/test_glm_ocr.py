from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from hugging_mac_sdk.capabilities import DocumentParsing
from hugging_mac_sdk.models.glm_ocr import GLM_OCR_DEFINITION, GLM_OCR_MANIFEST
from hugging_mac_sdk.models.glm_ocr.config import GlmOcrMlxInstanceConfig
from hugging_mac_sdk.models.glm_ocr.instance import GlmOcrInstance
from hugging_mac_sdk.models.glm_ocr.mlx import MlxGlmOcrEngine
from hugging_mac_sdk.schemas.document_parsing import DocumentImage, DocumentParsingRequest
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_manifest_declares_mlx_community_checkpoint() -> None:
    artifact = GLM_OCR_DEFINITION.get_artifact("mlx", "mlx-8bit", "8bit")

    assert GLM_OCR_MANIFEST.model_id == "zai-org/glm-ocr"
    assert GLM_OCR_DEFINITION.supported_runtimes == ("mlx",)
    assert GLM_OCR_MANIFEST.capabilities == frozenset({"document-parsing"})
    assert isinstance(artifact.source, HuggingFaceSource)
    assert artifact.source.repo_id == "mlx-community/GLM-OCR-8bit"
    assert Path("model.safetensors") in artifact.required_files
    assert Path("preprocessor_config.json") in artifact.required_files
    assert Path("tokenizer.json") in artifact.required_files


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

    instance = GlmOcrInstance(
        GlmOcrMlxInstanceConfig(artifact_path=tmp_path),
        Engine(),  # type: ignore[arg-type]
        GLM_OCR_MANIFEST,
    )
    await instance.load()
    parser = instance.require(DocumentParsing)
    request = DocumentParsingRequest(images=(DocumentImage(path=tmp_path / "page.png"),))

    response = await parser.parse_document(request)
    events = [event async for event in parser.stream_document(request)]

    assert response.text == "# Parsed document"
    assert "".join(event.delta for event in events) == "# Parsed document"
    assert events[-1].finish_reason == "stop"
    await instance.unload()


async def test_mlx_engine_uses_native_prompt_and_processes_pages_sequentially(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []
    mlx_vlm = ModuleType("mlx_vlm")
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")

    def apply_chat_template(
        processor: object, config: object, prompt: str, *, num_images: int
    ) -> str:
        del processor, config
        assert prompt == "Text Recognition:"
        assert num_images == 1
        return "formatted recognition prompt"

    def generate(
        model: object, processor: object, prompt: str, images: list[str], **kwargs
    ) -> object:
        del model, processor
        calls.append({"prompt": prompt, "images": images, **kwargs})
        return type("Result", (), {"text": f"text-{Path(images[0]).stem}"})()

    def stream_generate(model: object, processor: object, prompt: str, images: list[str], **kwargs):
        del model, processor
        calls.append({"prompt": prompt, "images": images, **kwargs})
        yield type("Result", (), {"token": 1})()
        yield type("Result", (), {"token": 2})()

    class Processor:
        tokenizer = None

        def __init__(self) -> None:
            self.tokenizer = self
            self.all_special_ids: set[int] = set()

        def decode(self, tokens: list[int], **_kwargs: object) -> str:
            return "".join({1: "pa", 2: "ge"}[token] for token in tokens)

    mlx_vlm.generate = generate  # type: ignore[attr-defined]
    mlx_vlm.stream_generate = stream_generate  # type: ignore[attr-defined]
    prompt_utils.apply_chat_template = apply_chat_template  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    engine = object.__new__(MlxGlmOcrEngine)
    engine._model = type("Model", (), {"config": {"model_type": "glm_ocr"}})()  # type: ignore[attr-defined]
    engine._processor = Processor()  # type: ignore[attr-defined]
    request = DocumentParsingRequest(
        images=(
            DocumentImage(path=tmp_path / "one.png"),
            DocumentImage(path=tmp_path / "two.png"),
        ),
        prompt="An external prompt the model should not receive",
    )

    assert await engine.infer(request) == (
        "<!-- Page 1 -->\n\ntext-one\n\n<!-- Page 2 -->\n\ntext-two"
    )
    calls.clear()
    streamed = "".join([delta async for delta in engine.stream(request)])
    assert streamed == "<!-- Page 1 -->\n\npage\n\n<!-- Page 2 -->\n\npage"
    assert len(calls) == 2
    assert all(call["prompt"] == "formatted recognition prompt" for call in calls)


def test_mlx_engine_routes_only_supported_native_tasks() -> None:
    assert MlxGlmOcrEngine._task_prompt("document parsing.") == "Text Recognition:"
    assert MlxGlmOcrEngine._task_prompt("Table Recognition:") == "Table Recognition:"
    assert MlxGlmOcrEngine._task_prompt("Formula Recognition:") == "Formula Recognition:"
    assert MlxGlmOcrEngine._task_prompt("Convert this table recognition region") == (
        "Table Recognition:"
    )
