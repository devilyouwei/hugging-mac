"""Offline reference/candidate evaluation with provenance and isolated reports."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

from ..artifacts import create_run, inventory, source_directory
from ..contracts import ModelDefinition
from ..hub import source_inventory
from .backends import open_backend
from .contracts import EvaluationConfig, EvaluationSample
from .metrics import compare
from .reporting import markdown


def inspect_bundle(definition: ModelDefinition, variant: str, run: Path) -> dict[str, Any]:
    report = json.loads((run / "build.json").read_text())
    config = definition.config.variant(variant)
    if report.get("schema_version") == 1:
        if report.get("config") != definition.snapshot(variant):
            raise ValueError("Build configuration does not match the selected model/variant")
    elif report.get("recipe", {}).get("source") != config.source:
        raise ValueError("Legacy build source does not match the selected model/variant")
    else:
        # Legacy diagnostic bundles can be evaluated, but never promoted/published.
        recipe = report["recipe"]
        if recipe.get("sequence_length") != config.probe.sequence_length:
            raise ValueError("Legacy build input length does not match evaluation configuration")
        if (
            recipe.get("inputs") != list(config.probe.inputs)
            or recipe.get("output") != config.probe.output
        ):
            raise ValueError(
                "Legacy build tensor interface does not match evaluation configuration"
            )
    runtime = report.get("runtime")
    if runtime not in {"coreml", "coreai"}:
        raise ValueError("Evaluation requires a Core ML or Core AI artifact")
    name = report.get("artifact", "model.mlpackage" if runtime == "coreml" else "model.aimodel")
    artifact = (run / name).resolve()
    if not artifact.is_relative_to(run.resolve()) or not artifact.is_dir():
        raise ValueError("Missing or unsafe evaluation artifact")
    files = inventory(run)
    if report.get("files") is not None and report["files"] != files:
        raise ValueError("Converted bundle changed since its build")
    return {
        "run": str(run.resolve()),
        "artifact": str(artifact),
        "runtime": runtime,
        "source_revision": report["source_revision"],
        "build_status": report["status"],
        "files": files,
        "build_report_sha256": hashlib.sha256((run / "build.json").read_bytes()).hexdigest(),
    }


async def evaluate_candidate(
    bundle: dict[str, Any],
    samples: list[EvaluationSample],
    config: EvaluationConfig,
    coreai_device: str = "auto",
) -> dict[str, Any]:
    result = dict(bundle, cases=[])
    try:
        started = perf_counter()
        async with open_backend(
            bundle["runtime"], Path(bundle["artifact"]), coreai_device
        ) as predict:
            result["load_ms"] = (perf_counter() - started) * 1000
            for sample in samples:
                row: dict[str, Any] = {
                    "id": sample.case.id,
                    "capability": sample.case.capability,
                    "reference_ms": sample.reference_ms,
                }
                if sample.error:
                    row.update(status="not_evaluated", reason=sample.error)
                else:
                    try:
                        started = perf_counter()
                        outputs = await predict(sample.inputs)
                        row["converted_ms"] = (perf_counter() - started) * 1000
                        row.update(compare(sample, outputs[sample.output_name], config))
                        row["status"] = "passed" if row["passed"] else "failed"
                    except Exception as error:
                        row.update(
                            status="error", error_type=type(error).__name__, reason=str(error)
                        )
                result["cases"].append(row)
        statuses = {row["status"] for row in result["cases"]}
        result["status"] = (
            "failed"
            if statuses & {"failed", "error"}
            else "incomplete"
            if "not_evaluated" in statuses
            else "diagnostic-passed"
        )
    except Exception as error:
        result.update(status="error", error_type=type(error).__name__, reason=str(error))
    return result


def _save(run: Path, report: dict[str, Any]) -> None:
    staged = run / ".report.json.tmp"
    staged.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    staged.replace(run / "report.json")
    staged_markdown = run / ".report.md.tmp"
    staged_markdown.write_text(markdown(report))
    staged_markdown.replace(run / "report.md")


def evaluate(
    definition: ModelDefinition,
    variant: str,
    runs: list[Path],
    source_root: Path,
    output: Path,
    revision: str | None = None,
    coreai_device: str = "auto",
) -> Path:
    if definition.evaluation is None:
        raise ValueError("This model has no evaluation adapter")
    config = definition.config.variant(variant)
    evaluation_config, cases = definition.evaluation.load(variant)
    bundles = [inspect_bundle(definition, variant, run) for run in runs]
    revisions = {bundle["source_revision"] for bundle in bundles}
    if revision:
        revisions.add(revision)
    if len(revisions) != 1:
        raise ValueError(
            "Compare artifacts from one source commit, or supply --revision for reference-only"
        )
    revision = revisions.pop()
    source = source_directory(source_root, definition.config.name, variant, revision)
    metadata_path = source / ".source.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata["repo_id"], metadata["revision"]) != (config.source, revision):
        raise ValueError("Original model provenance does not match the converted artifact")
    if metadata["files"] != source_inventory(source):
        raise ValueError("Original model files changed since download/migration")
    run = create_run(output, definition.config.name, variant, None)
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": definition.config.name,
        "variant": variant,
        "scope": "diagnostic",
        "status": "running",
        "source_path": str(source.resolve()),
        "source_revision": revision,
        "source_selections": metadata["selections"],
        "source_manifest_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "evaluation_config": evaluation_config.model_dump(mode="json"),
        "cases_sha256": hashlib.sha256(
            json.dumps([c.model_dump(mode="json") for c in cases], sort_keys=True).encode()
        ).hexdigest(),
        "capability_coverage": {
            name: "untested" for name in definition.config.required_capabilities
        },
        "candidates": [],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "coreml_device": "cpu",
            "coreai_device": coreai_device,
            "USE_OS_COREAI": os.environ.get("USE_OS_COREAI"),
        },
    }
    _save(run, report)
    try:
        backend_packages = {
            package
            for bundle in bundles
            for package in (
                ("coremltools",)
                if bundle["runtime"] == "coreml"
                else ("coreai-torch", "coreai-core")
            )
        }
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", *config.probe.packages, *sorted(backend_packages))
        }
        samples = definition.evaluation.prepare(config, source, cases)
        if [s.case.id for s in samples] != [c.id for c in cases]:
            raise ValueError("Model evaluation adapter omitted or reordered cases")
        report["reference_cases"] = []
        for sample in samples:
            row: dict[str, Any] = {
                "id": sample.case.id,
                "capability": sample.case.capability,
                "inference_ms": sample.reference_ms,
            }
            if sample.error:
                row.update(status="not_evaluated", reason=sample.error)
            else:
                import numpy as np

                probabilities = np.asarray(sample.reference, dtype=np.float64)
                if (
                    probabilities.shape != (len(sample.labels),)
                    or not np.isfinite(probabilities).all()
                ):
                    raise ValueError("Invalid original-model probabilities")
                decision = sample.labels[int(probabilities.argmax())]
                row.update(
                    status="evaluated",
                    probabilities=probabilities.tolist(),
                    decision=decision,
                    expected=sample.case.expected,
                    correct=decision == sample.case.expected
                    if sample.case.expected is not None
                    else None,
                )
                report["capability_coverage"][sample.case.capability] = "exercised-in-diagnostic"
            report["reference_cases"].append(row)
        gc.collect()  # Model adapters return arrays, so reference weights can be released now.
        _save(run, report)
        for bundle in bundles:
            report["active_candidate"] = bundle["run"]
            _save(run, report)
            report["candidates"].append(
                asyncio.run(evaluate_candidate(bundle, samples, evaluation_config, coreai_device))
            )
            report.pop("active_candidate", None)
            _save(run, report)
        statuses = {item["status"] for item in report["candidates"]}
        report["status"] = (
            "failed"
            if statuses & {"failed", "error"}
            else "incomplete"
            if "incomplete" in statuses or any(s.error for s in samples)
            else "diagnostic-passed"
            if bundles
            else "reference-only"
        )
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        _save(run, report)
        print(f"Evaluation report: {run / 'report.json'}", flush=True)
    return run
