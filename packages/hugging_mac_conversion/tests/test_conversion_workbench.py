from __future__ import annotations

import ast
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hugging_mac_conversion.registry import get_model
from hugging_mac_conversion.shared import pipeline
from hugging_mac_conversion.shared.artifacts import create_run, default_output_root, inventory
from hugging_mac_conversion.shared.config import ModelConfig
from hugging_mac_conversion.shared.contracts import PreparedGraph
from hugging_mac_conversion.shared.validation import validate_capability_coverage
from pydantic import ValidationError

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "packages/hugging_mac_conversion/src/hugging_mac_conversion"


def release_report(definition, run):
    variant = definition.config.default_variant
    return {
        "schema_version": 1,
        "model": definition.config.name,
        "variant": variant,
        "config": definition.snapshot(variant),
        "status": "validated",
        "bundle_verified": True,
        "files": inventory(run),
        "runtime": "coreml",
        "artifact_scope": "full",
        "capability_coverage": {
            n: {"status": "passed", "evidence": "test case"}
            for n in definition.config.required_capabilities
        },
    }


def test_catalog_is_offline_and_framework_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from hugging_mac_conversion.registry import get_model; "
                "assert get_model('vela').config.variant('3-fast').source == 'vllm-sr/vela3-fast'; "
                "heavy = {'torch', 'coremltools', 'coreai_torch', 'transformers'}; "
                "assert not heavy & sys.modules.keys()"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize("name", ["../vela", "/tmp/vela", "unknown"])
def test_registry_rejects_unknown_models(name):
    with pytest.raises(ValueError, match="Unknown model"):
        get_model(name)


def test_variants_are_local_and_build_config_ignores_unselected_variants():
    original = get_model("vela")
    data = original.config.model_dump()
    data["variants"]["another"] = dict(data["variants"]["3-fast"], source="example/another")
    another = replace(original, config=ModelConfig.model_validate(data))
    assert another.config.variant("another").source == "example/another"
    assert another.snapshot("3-fast") == original.snapshot("3-fast")
    assert "another" not in get_model("d3").config.variants
    with pytest.raises(ValueError, match="Unknown variant"):
        original.config.variant("another")


@pytest.mark.parametrize(
    "change",
    [
        {"default_variant": "missing"},
        {"name": "../escape"},
        {"unexpected": True},
    ],
)
def test_config_is_validated(change):
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(get_model("vela").config.model_dump() | change)


def test_shared_layer_and_models_have_no_reverse_or_cross_model_imports():
    for path in SOURCE.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "shared" in path.parts:
                    assert "models" not in module and "registry" not in module
                if "models" in path.parts:
                    other = "d3" if "vela" in path.parts else "vela"
                    assert other not in module.split(".")


def test_artifacts_are_project_local_from_any_subdirectory(tmp_path):
    assert default_output_root(SOURCE / "models/vela") == ROOT / "tmp/conversions"
    run = create_run(tmp_path, "vela", "3-fast", "coreml")
    assert run.parent == tmp_path / "vela/3-fast/coreml"
    assert create_run(tmp_path, "vela", "3-fast", "coreml") != run
    with pytest.raises(ValueError, match="Invalid artifact"):
        create_run(tmp_path, "../outside", "edge", "coreai")
    subprocess.run(
        ["git", "check-ignore", "--quiet", str(ROOT / "tmp/conversions/model.bin")], check=True
    )


def test_inventory_detects_changed_assets_and_rejects_symlinks(tmp_path):
    weight = tmp_path / "weight.bin"
    weight.write_bytes(b"old")
    before = inventory(tmp_path)
    weight.write_bytes(b"new")
    assert inventory(tmp_path) != before
    (tmp_path / "link").symlink_to(weight)
    with pytest.raises(ValueError, match="symlinks"):
        inventory(tmp_path)


@pytest.mark.parametrize("runtime", ["coreml", "coreai"])
def test_pipeline_uses_injected_model_and_runtime(tmp_path, monkeypatch, runtime):
    tensor = SimpleNamespace(numpy=lambda: np.array([[1.0]]))
    value = SimpleNamespace(float=lambda: tensor)
    graph = PreparedGraph(lambda *args: value, (tensor,), ("x",), "y")
    definition = replace(get_model("d3"), prepare=lambda config, root: graph)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setattr(pipeline.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(pipeline.hub, "resolve_revision", lambda config, revision: "a" * 40)
    monkeypatch.setattr(
        pipeline.hub, "download_probe", lambda config, revision, destination: tmp_path
    )
    monkeypatch.setattr(pipeline.hub, "package_assets", lambda *args: None)

    def export(prepared, run):
        assert prepared is graph
        artifact = run / "model.bin"
        artifact.write_bytes(b"converted")
        return artifact

    async def coreai_predict(*args):
        return {"y": tensor.numpy()}

    backend = pipeline.coreml if runtime == "coreml" else pipeline.coreai
    monkeypatch.setattr(backend, "export", export)
    monkeypatch.setattr(
        backend,
        "predict",
        (lambda *args: {"y": tensor.numpy()}) if runtime == "coreml" else coreai_predict,
    )
    run = pipeline.build_probe(definition, "edge", runtime, tmp_path)
    report = json.loads((run / "build.json").read_text())
    assert report["status"] == "diagnostic-validated"
    assert report["source_revision"] == "a" * 40
    assert report["config"] == definition.snapshot("edge")
    assert report["files"] == inventory(run)
    with pytest.raises(ValueError, match="Diagnostic exports"):
        pipeline.publish(definition, run, report, dry_run=False)


def test_build_failure_is_recorded_without_publishing(tmp_path, monkeypatch):
    def fail(name):
        raise RuntimeError("missing conversion dependency")

    monkeypatch.setattr(pipeline.importlib.metadata, "version", fail)
    with pytest.raises(RuntimeError, match="missing conversion"):
        pipeline.build_probe(get_model("vela"), "3-fast", "coreml", tmp_path)
    report = json.loads(next(tmp_path.rglob("build.json")).read_text())
    assert report["status"] == "failed"
    assert report["stage"] == "check-environment"


@pytest.mark.parametrize(
    "status,integrated", [("building", True), ("failed", True), ("validated", False)]
)
def test_unverified_models_cannot_be_published(tmp_path, status, integrated):
    definition = get_model("d3")
    report = release_report(definition, tmp_path) | {
        "status": status,
        "bundle_verified": integrated,
    }
    with pytest.raises(ValueError, match="Native capability"):
        pipeline.publish(definition, tmp_path, report, dry_run=False)


def test_publication_rejects_mutated_release(tmp_path):
    definition = get_model("vela")
    (tmp_path / "model.bin").write_bytes(b"before")
    report = release_report(definition, tmp_path)
    (tmp_path / "model.bin").write_bytes(b"after")
    with pytest.raises(ValueError, match="changed since validation"):
        pipeline.publish(definition, tmp_path, report)


def test_dry_run_keeps_target_from_selected_variant(tmp_path):
    definition = get_model("d3")
    report = release_report(definition, tmp_path)
    report["runtime"] = "coreai"
    result = pipeline.publish(definition, tmp_path, report)
    assert "hugging-mac/d3-edge-CoreAI/coreai" in result


def test_untested_features_cannot_be_silently_dropped():
    definition = get_model("d3")
    with pytest.raises(ValueError, match="Unverified upstream capability"):
        validate_capability_coverage(definition, {"capability_coverage": {}})
    coverage = {name: {"status": "unsupported"} for name in definition.config.required_capabilities}
    with pytest.raises(ValueError, match="evidence"):
        validate_capability_coverage(definition, {"capability_coverage": coverage})


def test_legacy_reports_remain_unpublishable(tmp_path):
    with pytest.raises(ValueError, match="Legacy build report"):
        pipeline.publish(get_model("vela"), tmp_path, {"recipe": {"name": "vela3-fast"}})
