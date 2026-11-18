"""Enforce model ownership and independent SDK/conversion implementations."""

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / "packages/hugging_mac_sdk/src/hugging_mac_sdk"
CONVERSION = ROOT / "packages/hugging_mac_conversion/src/hugging_mac_conversion"


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_sdk_and_conversion_have_no_cross_package_dependencies():
    for source, forbidden in ((SDK, "hugging_mac_conversion"), (CONVERSION, "hugging_mac_sdk")):
        for path in source.rglob("*.py"):
            assert not any(name.startswith(forbidden) for name in imports(path)), path
    import tomllib

    project = tomllib.loads((CONVERSION.parents[1] / "pyproject.toml").read_text())
    assert not any(dep.startswith("hugging-mac-sdk") for dep in project["project"]["dependencies"])


def test_model_private_imports_stay_within_owning_package():
    for source in (SDK, CONVERSION, ROOT / "apps"):
        for path in source.rglob("*.py"):
            for name in imports(path):
                parts = name.split(".")
                if (
                    len(parts) >= 4
                    and parts[0] in {"hugging_mac_sdk", "hugging_mac_conversion"}
                    and parts[1] == "models"
                    and parts[3] == "utils"
                ):
                    owner = (
                        (SDK if parts[0] == "hugging_mac_sdk" else CONVERSION) / "models" / parts[2]
                    )
                    assert path.is_relative_to(owner), (path, name)


def test_new_model_roots_follow_standard_layout():
    allowed = {
        "__init__.py",
        "readme.md",
        "model.yaml",
        "config.py",
        "definition.py",
        "instance.py",
        "resources.py",
        "torch.py",
        "mlx.py",
        "coreml.py",
        "coreai.py",
        "onnx.py",
        "converter.py",
    }
    for model in (p.parent.name for p in (SDK / "models").glob("*/model.yaml")):
        assert {
            path.name for path in (SDK / "models" / model).iterdir() if path.is_file()
        } <= allowed


def test_sdk_has_no_attribution_files_or_runtime_requirements():
    assert not [
        p
        for p in SDK.rglob("*")
        if p.is_file() and p.name.upper().startswith(("LICENSE", "NOTICE"))
    ]
    for path in (SDK / "models").glob("*/model.yaml"):
        for artifact in yaml.safe_load(path.read_text())["artifacts"]:
            assert not any(
                Path(name).name.upper().startswith(("LICENSE", "NOTICE"))
                for name in artifact.get("required_files", [])
            ), path


def test_artifact_source_order():
    order = {
        name: rank
        for rank, names in enumerate(
            (
                ("kind",),
                ("repo_id", "url"),
                ("filename", "allow_patterns"),
                ("format", "strip_prefix", "strip_components"),
            )
        )
        for name in names
    }
    for path in (SDK / "models").glob("*/model.yaml"):
        for artifact in yaml.safe_load(path.read_text())["artifacts"]:
            keys = [order[key] for key in artifact.get("source", {}) if key in order]
            assert keys == sorted(keys), (path, artifact["artifact_id"])
