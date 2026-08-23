from __future__ import annotations

import importlib
import re
from pathlib import Path

from fastapi.routing import APIRoute
from hugging_mac_web.index import create_index_router
from hugging_mac_web.media import create_media_router
from hugging_mac_web.models import create_models_router
from hugging_mac_web.system import create_system_router
from starlette.routing import WebSocketRoute

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPOSITORY_ROOT / "apps" / "web" / "backend"
SOURCE_ROOT = BACKEND_ROOT / "src" / "hugging_mac_web"
TECHNICAL_READMES = (BACKEND_ROOT / "README.md", *sorted(SOURCE_ROOT.rglob("README.md")))
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _blueprint_directories() -> tuple[Path, ...]:
    return tuple(sorted(path.parent for path in SOURCE_ROOT.glob("*/blueprint.py")))


def _blueprint(directory: Path) -> object:
    module = importlib.import_module(f"hugging_mac_web.{directory.name}.blueprint")
    return module.create_blueprint()


def _assert_route_documented(text: str, method: str, path: str) -> None:
    expected = f"| {method} | `{path}` |"
    assert expected in text, f"Route is not documented as a protocol table row: {expected}"


def test_every_backend_logic_module_has_a_technical_readme() -> None:
    missing = [directory / "README.md" for directory in _blueprint_directories()]
    missing = [path for path in missing if not path.is_file()]

    assert not missing
    assert (SOURCE_ROOT / "README.md").is_file()
    assert (SOURCE_ROOT / "shared" / "README.md").is_file()
    assert (SOURCE_ROOT / "shared" / "cache" / "README.md").is_file()
    assert (SOURCE_ROOT / "shared" / "storage" / "README.md").is_file()
    assert (SOURCE_ROOT / "shared" / "utils" / "README.md").is_file()
    object_detection_names = {
        path.name for path in (SOURCE_ROOT / "object_detection").iterdir()
    }
    assert "README.md" in object_detection_names
    assert "readme.md" not in object_detection_names


def test_backend_technical_readmes_are_english_and_well_formed() -> None:
    for readme in TECHNICAL_READMES:
        text = readme.read_text(encoding="utf-8")

        assert not CJK_CHARACTER.search(text), (
            f"Backend technical documentation must be English: {readme}"
        )

        fence_open = False
        for line in text.splitlines():
            if not line.startswith("```"):
                continue
            fence_open = not fence_open
        assert not fence_open, f"Unclosed Markdown fence in {readme}"

        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>").split("#", maxsplit=1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (readme.parent / target).resolve()
            assert resolved.exists(), f"Broken relative link in {readme}: {raw_target}"


def test_app_readmes_follow_the_technical_documentation_contract() -> None:
    for directory in _blueprint_directories():
        readme = directory / "README.md"
        text = readme.read_text(encoding="utf-8")
        blueprint = _blueprint(directory)
        manifest = blueprint.manifest

        assert "## Summary" in text
        assert "## Module structure" in text
        assert "## Interfaces" in text or "## Interface boundary" in text
        assert "## Tests and review points" in text
        assert manifest.api_prefix in text

        for requirement in manifest.required_models:
            assert requirement.model_id in text
            for capability in requirement.capabilities:
                assert capability in text


def test_app_readmes_cover_every_http_and_websocket_route() -> None:
    for directory in _blueprint_directories():
        blueprint = _blueprint(directory)
        text = (directory / "README.md").read_text(encoding="utf-8")

        for route in blueprint.create_router().routes:
            if isinstance(route, APIRoute):
                for method in sorted(route.methods):
                    _assert_route_documented(text, method, route.path)
            elif isinstance(route, WebSocketRoute):
                _assert_route_documented(text, "WS", route.path)


def test_platform_readme_covers_every_core_router_route() -> None:
    text = (SOURCE_ROOT / "README.md").read_text(encoding="utf-8")
    routers = (
        create_index_router(),
        create_models_router(),
        create_media_router(),
        create_system_router(),
    )

    for router in routers:
        for route in router.routes:
            assert isinstance(route, APIRoute)
            for method in sorted(route.methods):
                _assert_route_documented(text, method, route.path)


def test_live_transcription_readme_preserves_pipeline_design_sections() -> None:
    text = (SOURCE_ROOT / "live_transcription" / "README.md").read_text(encoding="utf-8")
    required_sections = (
        "## Summary",
        "## Models and capabilities",
        "## Module structure",
        "## Interface boundary",
        "## Audio data structures and memory",
        "## VAD and segmentation state machine",
        "## Regular ASR pipeline",
        "## Streaming ASR pipeline",
        "## Result protocol",
        "## Lifecycle and failures",
        "## Observability",
        "## Tests and review points",
    )

    for section in required_sections:
        assert section in text

    for queue_name in ("`raw`", "`utterances`", "`asr`", "`results`", "`stream_queue`"):
        assert queue_name in text
    assert "8-byte big-endian unsigned utterance ID" in text
    assert "16 kHz" in text
