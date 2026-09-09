from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from hugging_mac_sdk.capabilities import DocumentParsing
from hugging_mac_sdk.models.got_ocr2 import GOT_OCR2_DEFINITION, GOT_OCR2_MANIFEST
from hugging_mac_sdk.models.got_ocr2.config import GotOcr2MlxInstanceConfig
from hugging_mac_sdk.models.got_ocr2.definition import _ARTIFACT, _SOURCE
from hugging_mac_sdk.models.got_ocr2.instance import GotOcr2Instance
from hugging_mac_sdk.models.got_ocr2.mlx import MlxGotOcr2Engine
from hugging_mac_sdk.models.got_ocr2.resources import GotOcr2ResourceResolver
from hugging_mac_sdk.schemas.document_parsing import DocumentImage, DocumentParsingRequest
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_manifest_declares_mlx_community_checkpoint() -> None:
    artifact = GOT_OCR2_DEFINITION.get_artifact("mlx", "mlx-8bit", "8bit")

    assert GOT_OCR2_MANIFEST.model_id == "stepfun-ai/got-ocr2.0"
    assert GOT_OCR2_DEFINITION.supported_runtimes == ("mlx",)
    assert GOT_OCR2_MANIFEST.capabilities == frozenset({"document-parsing"})
    assert isinstance(artifact.source, HuggingFaceSource)
    assert artifact.source.repo_id == "mlx-community/GOT-OCR2_0-8bit"
    assert Path("model.safetensors") in artifact.required_files
    assert Path("qwen.tiktoken") in artifact.required_files
    assert Path("tokenization_qwen.py") in artifact.required_files
    assert "tokenization_qwen.py" in artifact.source.allow_patterns


async def test_incomplete_existing_checkpoint_is_repaired(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    calls: list[bool] = []

    class Downloader:
        async def download(
            self, source, destination: Path, *, overwrite: bool, token: str | None
        ) -> None:
            del source, token
            calls.append(overwrite)
            for filename in _ARTIFACT.required_files:
                target = destination / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()

    config = GotOcr2MlxInstanceConfig(artifact_path=checkpoint)
    resolver = GotOcr2ResourceResolver(
        _SOURCE,
        config,
        manifest=GOT_OCR2_MANIFEST,
        artifact=_ARTIFACT,
        downloader=Downloader(),  # type: ignore[arg-type]
    )
    await resolver.download()

    assert calls == [True]
    assert (checkpoint / "tokenization_qwen.py").is_file()


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

    instance = GotOcr2Instance(
        GotOcr2MlxInstanceConfig(artifact_path=tmp_path),
        Engine(),  # type: ignore[arg-type]
        GOT_OCR2_MANIFEST,
    )
    await instance.load()
    parser = instance.require(DocumentParsing)
    request = DocumentParsingRequest(images=(DocumentImage(path=tmp_path / "page.png"),))

    response = await parser.parse_document(request)
    events = [event async for event in parser.stream_document(request)]

    assert response.text == "# Parsed document"
    assert response.page_count == 1
    assert "".join(event.delta for event in events) == "# Parsed document"
    assert events[-1].finish_reason == "stop"
    await instance.unload()


async def test_mlx_engine_uses_native_prompt_and_processes_pages_sequentially(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []
    mlx_vlm = ModuleType("mlx_vlm")
    generate_module = ModuleType("mlx_vlm.generate")
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")

    def apply_chat_template(
        processor: object, config: object, prompt: str, *, num_images: int
    ) -> str:
        assert prompt == "OCR with format: "
        assert num_images == 1
        return "formatted OCR prompt"

    def generate(
        model: object, processor: object, prompt: str, images: list[str], **kwargs: object
    ) -> object:
        calls.append({"prompt": prompt, "images": images, **kwargs})
        return type("Result", (), {"text": f"text-{Path(images[0]).stem}"})()

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
            return "".join({1: "pa", 2: "ge"}[token] for token in tokens)

    generate_module.generate = generate  # type: ignore[attr-defined]
    generate_module.stream_generate = stream_generate  # type: ignore[attr-defined]
    prompt_utils.apply_chat_template = apply_chat_template  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    engine = object.__new__(MlxGotOcr2Engine)
    engine._model = type("Model", (), {"config": {"model_type": "got"}})()  # type: ignore[attr-defined]
    engine._processor = Processor()  # type: ignore[attr-defined]
    request = DocumentParsingRequest(
        images=(
            DocumentImage(path=tmp_path / "one.png"),
            DocumentImage(path=tmp_path / "two.png"),
        ),
        prompt="A prompt GOT cannot safely follow",
    )

    assert await engine.infer(request) == (
        "<!-- Page 1 -->\n\ntext-one\n\n<!-- Page 2 -->\n\ntext-two"
    )
    assert [call["images"] for call in calls] == [
        [str(tmp_path / "one.png")],
        [str(tmp_path / "two.png")],
    ]
    assert all(call["prompt"] == "formatted OCR prompt" for call in calls)

    calls.clear()
    streamed = "".join([delta async for delta in engine.stream(request)])
    assert streamed == "<!-- Page 1 -->\n\npage\n\n<!-- Page 2 -->\n\npage"
    assert len(calls) == 2
    assert all(call["verbose"] is False for call in calls)
