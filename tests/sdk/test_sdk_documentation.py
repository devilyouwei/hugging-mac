from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from urllib.parse import unquote

import yaml

REPOSITORY_ROOT = Path(__file__).parents[2]
MODEL_ROOT = (
    REPOSITORY_ROOT
    / "packages"
    / "hugging_mac_sdk"
    / "src"
    / "hugging_mac_sdk"
    / "models"
)
DOC_ROOT = REPOSITORY_ROOT / "docs"
MODEL_DOC_ROOT = DOC_ROOT / "sdk" / "models"
MODEL_DOC_NAMES = {
    "audio8_asr": "audio8-asr.md",
    "audio8_tts": "audio8-tts.md",
    "deepfilternet3": "deepfilternet3.md",
    "gemma_4": "gemma-4.md",
    "got_ocr2": "got-ocr2.md",
    "glm_ocr": "glm-ocr.md",
    "kokoro": "kokoro.md",
    "mediapipe_hand_detection": "mediapipe-hand-detection.md",
    "moss_tts_nano": "moss-tts-nano.md",
    "nemotron_3_5_asr": "nemotron-3.5-asr.md",
    "pp_doclayout_v3": "pp-doclayout-v3.md",
    "qwen3_5": "qwen3.5.md",
    "qwen3_asr": "qwen3-asr.md",
    "qwen3_tts": "qwen3-tts.md",
    "retinaface": "retinaface.md",
    "sensevoice": "sensevoice.md",
    "silero": "silero-vad.md",
    "unlimited_ocr": "unlimited-ocr.md",
    "yolov8": "yolov8.md",
    "yolov8_pose": "yolov8-pose.md",
    "yolov8_seg": "yolov8-seg.md",
}
MODEL_DOCUMENT_HEADINGS = (
    "## Purpose",
    "## Support",
    "## Prepare resources",
    "## Example",
    "## Notes and limits",
    "## License",
)
PUBLIC_FORBIDDEN_TERMS = (
    "revision",
    "sha-256",
    "expected_sha",
    "runtime_factories",
    "modeldefinition",
    "modelartifact",
    "src/hugging_mac_sdk",
    "config.py",
    "definition.py",
    "converter.py",
    "resources.py",
)
PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _manifest(package_name: str) -> dict[str, object]:
    package = yaml.safe_load(
        (MODEL_ROOT / package_name / "model.yaml").read_text(encoding="utf-8")
    )
    return package["manifest"]


def _public_markdown_files() -> tuple[Path, ...]:
    return tuple(sorted(DOC_ROOT.rglob("*.md")))


def test_docs_is_the_only_public_documentation_tree() -> None:
    assert DOC_ROOT.is_dir()
    assert not (REPOSITORY_ROOT / "doc").exists()

    mkdocs = (REPOSITORY_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(r"^docs_dir: docs$", mkdocs, re.MULTILINE)
    assert re.search(r"^  language: en$", mkdocs, re.MULTILINE)

    app_docs = tuple(
        path
        for path in (REPOSITORY_ROOT / "apps" / "web").rglob("*.md")
        if "node_modules" not in path.parts and "dist" not in path.parts
    )
    sdk_docs = tuple(
        (REPOSITORY_ROOT / "packages" / "hugging_mac_sdk").rglob("*.md")
    )
    scan_roots = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "CONTRIBUTING.md",
        REPOSITORY_ROOT / "ARCHITECTURE.md",
        REPOSITORY_ROOT / "mkdocs.yml",
        *DOC_ROOT.rglob("*.md"),
        *app_docs,
        *sdk_docs,
    )
    stale = [path for path in scan_roots if "doc/" in path.read_text(encoding="utf-8")]
    assert not stale, f"References to the removed doc/ tree remain: {stale}"


def test_public_docs_do_not_contain_implementation_details() -> None:
    for path in _public_markdown_files():
        text = path.read_text(encoding="utf-8").lower()
        found = [term for term in PUBLIC_FORBIDDEN_TERMS if term in text]
        assert not found, f"{path}: implementation-only terms in public docs: {found}"


def test_every_model_package_has_current_public_sdk_documentation() -> None:
    package_names = {path.parent.name for path in MODEL_ROOT.glob("*/model.yaml")}
    assert package_names == set(MODEL_DOC_NAMES)

    index = (MODEL_DOC_ROOT / "README.md").read_text(encoding="utf-8")
    for package_name, filename in MODEL_DOC_NAMES.items():
        document = MODEL_DOC_ROOT / filename
        assert document.is_file(), f"Missing SDK document for {package_name}"
        assert f"]({filename})" in index

        manifest = _manifest(package_name)
        text = document.read_text(encoding="utf-8")
        assert all(heading in text for heading in MODEL_DOCUMENT_HEADINGS)
        assert f"`{manifest['model_id']}`" in text
        assert f"`{manifest['license']}`" in text

        for variant in manifest["variants"]:  # type: ignore[index]
            assert f"`{variant['name']}`" in text
        for runtime in manifest["runtimes"]:  # type: ignore[index]
            assert f"`{runtime['name']}`" in text
        for capability in manifest["capabilities"]:  # type: ignore[index]
            assert f"`{capability}`" in text

        assert f"`{manifest['default_variant']}` (default)" in text
        assert f"`{manifest['default_runtime']}` (default)" in text


def test_model_examples_are_complete_and_syntactically_valid() -> None:
    for filename in MODEL_DOC_NAMES.values():
        path = MODEL_DOC_ROOT / filename
        examples = PYTHON_FENCE.findall(path.read_text(encoding="utf-8"))
        assert len(examples) == 1, f"{path}: expected one complete Python example"
        code = examples[0]
        compile(code, str(path), "exec")
        assert "asyncio.run(main())" in code


def test_public_python_imports_resolve() -> None:
    for path in _public_markdown_files():
        for code in PYTHON_FENCE.findall(path.read_text(encoding="utf-8")):
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        importlib.import_module(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        if alias.name != "*":
                            assert hasattr(module, alias.name), (
                                f"{path}: {node.module}.{alias.name} is not importable"
                            )


def test_documentation_relative_links_resolve() -> None:
    documents = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "CONTRIBUTING.md",
        REPOSITORY_ROOT / "ARCHITECTURE.md",
        REPOSITORY_ROOT / "packages" / "hugging_mac_sdk" / "README.md",
        REPOSITORY_ROOT / "apps" / "web" / "README.md",
        REPOSITORY_ROOT / "apps" / "web" / "backend" / "README.md",
        REPOSITORY_ROOT / "apps" / "web" / "frontend" / "README.md",
        *_public_markdown_files(),
    )
    for document in documents:
        for raw_target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", maxsplit=1)[0])
            assert (document.parent / relative).resolve().exists(), (
                f"{document}: broken relative link {raw_target}"
            )
