from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hugging_mac_conversion.registry import get_model
from hugging_mac_conversion.shared import hub
from hugging_mac_conversion.shared.artifacts import (
    default_evaluation_root,
    default_source_root,
    inventory,
    source_directory,
)
from hugging_mac_conversion.shared.evaluation import engine
from hugging_mac_conversion.shared.evaluation.contracts import (
    EvaluationCase,
    EvaluationConfig,
    EvaluationDefinition,
    EvaluationSample,
)
from hugging_mac_conversion.shared.evaluation.metrics import compare
from pydantic import ValidationError

REVISION = "a" * 40
ROOT = Path(__file__).parents[3]


def test_offline_source_checks_identity_and_contents(tmp_path):
    config = get_model("d3").config.variant()
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"original")
    hub.record_source(config, REVISION, tmp_path, "probe")
    assert hub.local_source(config, REVISION, tmp_path) == tmp_path
    with pytest.raises(ValueError, match="revision"):
        hub.local_source(config, "b" * 40, tmp_path)
    weights.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        hub.local_source(config, REVISION, tmp_path)


def test_project_original_and_evaluation_roots():
    assert default_source_root(ROOT / "packages") == ROOT / "tmp/originals"
    assert default_evaluation_root(ROOT / "packages") == ROOT / "tmp/evaluations"
    with pytest.raises(ValueError, match="immutable"):
        source_directory(ROOT, "vela", "3-fast", "../main")


@pytest.mark.parametrize("probe", [False, True])
def test_download_materializes_project_files_and_records_selection(tmp_path, monkeypatch, probe):
    config = get_model("vela").config.variant()
    destination = tmp_path / "original"
    calls = []

    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        (kwargs["local_dir"] / "model.bin").write_bytes(b"weights")
        return str(kwargs["local_dir"])

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert hub.download_source(config, REVISION, destination, probe=probe) == destination
    assert calls[0][1]["local_dir"] == destination
    assert calls[0][1]["ignore_patterns"] == (list(config.probe.ignore_patterns) if probe else None)
    report = json.loads((destination / ".source.json").read_text())
    assert report["selections"] == (["probe"] if probe else ["full"])
    assert report["files"] == hub.source_inventory(destination)
    assert not (destination / "model.bin").is_symlink()


def test_failed_download_never_marks_source_complete(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("download interrupted")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)
    with pytest.raises(RuntimeError):
        hub.download_source(get_model("d3").config.variant(), REVISION, tmp_path)
    assert not (tmp_path / ".source.json").exists()


def test_source_inventory_ignores_loader_cache_but_rejects_external_links(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    before = hub.source_inventory(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "model.pyc").write_bytes(b"generated")
    assert hub.source_inventory(tmp_path) == before
    (tmp_path / "link").symlink_to(tmp_path / "config.json")
    with pytest.raises(ValueError, match="real files"):
        hub.source_inventory(tmp_path)


@pytest.mark.parametrize("model", ["vela", "d3"])
def test_model_owned_suites_have_distinct_cases_and_lazy_adapters(model):
    definition = get_model(model)
    config, cases = definition.evaluation.load(definition.config.default_variant)
    assert config.scope == "diagnostic"
    assert len(cases) >= 4
    assert {c.capability for c in cases} <= set(definition.config.required_capabilities)
    assert definition.evaluation.prepare.__module__.endswith(f"models.{model}.evaluation.runner")


def test_case_paths_and_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValidationError):
        EvaluationConfig(cases=("../cases.json",))
    case = {"id": "same", "capability": "choice", "payload": {}}
    (tmp_path / "cases.json").write_text(json.dumps([case, case]))
    path = tmp_path / "config.yaml"
    path.write_text("cases: [cases.json]\n")
    definition = EvaluationDefinition(path, lambda *args: [])
    with pytest.raises(ValueError, match="unique IDs"):
        definition.load("edge")


def test_metrics_compare_probabilities_and_choices_without_relaxing_errors():
    case = EvaluationCase(id="choice", capability="choice", payload={}, expected="a")
    sample = EvaluationSample(case, {}, np.array([0.8, 0.2]), ("a", "b"), "scores")
    config = EvaluationConfig(cases=("unused.json",), atol=0.001, rtol=0.001)
    good = compare(sample, np.log([[0.8, 0.2]]), config)
    assert good["passed"] and good["converted_correct"]
    bad = compare(sample, np.log([[0.2, 0.8]]), config)
    assert not bad["passed"] and not bad["decision_agreement"]
    with pytest.raises(ValueError, match="Non-finite"):
        compare(sample, [[np.nan, 0]], config)


def setup_evaluation(tmp_path, monkeypatch, *, skip=False):
    definition = get_model("vela")
    variant = definition.config.default_variant
    source_root = tmp_path / "originals"
    source = source_directory(source_root, "vela", variant, REVISION)
    source.mkdir(parents=True)
    (source / "model.bin").write_bytes(b"original")
    hub.record_source(definition.config.variant(), REVISION, source, "cached-partial")
    config_dir = tmp_path / "suite"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("cases: [cases.json]\natol: 0.001\nrtol: 0.001\n")
    (config_dir / "cases.json").write_text(
        json.dumps(
            [
                {"id": "choice", "capability": "choice", "payload": {}, "expected": "a"},
            ]
        )
    )

    def prepare(config, root, cases):
        assert root == source
        return [
            EvaluationSample(
                cases[0],
                {"x": np.array([1])},
                np.array([0.8, 0.2]),
                ("a", "b"),
                "scores",
                error="task not yet covered" if skip else None,
            )
        ]

    definition = replace(
        definition, evaluation=EvaluationDefinition(config_dir / "config.yaml", prepare)
    )
    bundle = tmp_path / "conversions" / "run"
    artifact = bundle / "model.mlpackage"
    artifact.mkdir(parents=True)
    (artifact / "weights.bin").write_bytes(b"converted")
    report = {
        "schema_version": 1,
        "model": "vela",
        "variant": variant,
        "config": definition.snapshot(variant),
        "runtime": "coreml",
        "source_revision": REVISION,
        "status": "diagnostic-validated",
        "artifact": "model.mlpackage",
        "files": inventory(bundle),
    }
    (bundle / "build.json").write_text(json.dumps(report))
    monkeypatch.setattr(engine.importlib.metadata, "version", lambda name: "test")
    return definition, source_root, source, bundle


@pytest.mark.parametrize(
    "candidate,skip,status",
    [
        ([0.8, 0.2], False, "diagnostic-passed"),
        ([0.2, 0.8], False, "failed"),
        ([0.8, 0.2], True, "incomplete"),
    ],
)
def test_evaluation_reports_results_separately_and_preserves_artifacts(
    tmp_path, monkeypatch, candidate, skip, status
):
    definition, source_root, source, bundle = setup_evaluation(tmp_path, monkeypatch, skip=skip)
    original_report = (bundle / "build.json").read_bytes()
    original_inventory = inventory(bundle)
    closed = []

    @asynccontextmanager
    async def backend(runtime, artifact, coreai_device="auto"):
        async def predict(inputs):
            return {"scores": np.log([candidate])}

        try:
            yield predict
        finally:
            closed.append(True)

    monkeypatch.setattr(engine, "open_backend", backend)
    output = tmp_path / "evaluations"
    run = engine.evaluate(definition, "3-fast", [bundle], source_root, output)
    report = json.loads((run / "report.json").read_text())
    assert report["status"] == status
    readable = (run / "report.md").read_text()
    assert f"**{status}**" in readable
    assert "span | untested" in readable
    assert str(bundle.resolve()) in readable
    assert run.parent == output / "vela/3-fast"
    assert report["source_path"] == str(source)
    assert report["capability_coverage"]["span"] == "untested"
    assert closed
    assert inventory(bundle) == original_inventory
    assert (bundle / "build.json").read_bytes() == original_report


def test_evaluation_rejects_tampered_original_before_inference(tmp_path, monkeypatch):
    definition, source_root, source, bundle = setup_evaluation(tmp_path, monkeypatch)
    (source / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Original model files changed"):
        engine.evaluate(definition, "3-fast", [bundle], source_root, tmp_path / "evaluations")


def test_evaluation_rejects_mismatched_revisions(tmp_path, monkeypatch):
    definition, source_root, _source, bundle = setup_evaluation(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="one source commit"):
        engine.evaluate(
            definition, "3-fast", [bundle], source_root, tmp_path / "evaluations", "b" * 40
        )
