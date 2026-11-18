"""Numeric checks and complete Native/task acceptance before publication."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .artifacts import inventory, write_report
from .contracts import ModelDefinition


def check_tensors(actual: Any, expected: Any) -> dict[str, Any]:
    import numpy as np

    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.02)
    if not np.isfinite(actual).all():
        raise ValueError("Non-finite converted output")
    return {
        "samples": 1,
        "max_absolute_error": float(np.max(np.abs(actual - expected))),
        "atol": 0.02,
        "rtol": 0.02,
        "scope": "tensor smoke; task parity still required",
    }


def validate_capability_coverage(definition: ModelDefinition, report: dict[str, Any]) -> None:
    coverage = report.get("capability_coverage", {})
    for name in definition.config.required_capabilities:
        check = coverage.get(name, {})
        if check.get("status") not in {"passed", "unsupported"}:
            raise ValueError(f"Unverified upstream capability: {name}")
        if not check.get("evidence"):
            raise ValueError(f"Missing parity test or conversion failure evidence: {name}")


def validate_bundle(definition: ModelDefinition, run: Path, report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("Legacy build report; rebuild with the current model configuration")
    if report.get("config") != definition.snapshot(report["variant"]):
        raise ValueError("Model configuration changed; rebuild before verification or publication")
    if report.get("artifact_scope") != "full":
        raise ValueError("Diagnostic exports are not complete SDK models")
    if report.get("status") != "validated":
        raise ValueError(
            "Publication requires validated artifacts and Native capability task parity"
        )
    if report.get("runtime") not in {"coreml", "coreai"}:
        raise ValueError("Invalid release runtime")
    if report.get("files") != inventory(run):
        raise ValueError("Bundle changed since validation")


def verify_bundle(definition: ModelDefinition, run: Path, report: dict[str, Any]) -> None:
    validate_bundle(definition, run, report)
    module_name, function_name = definition.config.bundle_validator.split(":")
    result = getattr(importlib.import_module(module_name), function_name)(run)
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise ValueError("Bundle validation task parity did not pass")
    samples = result.get("samples")
    if type(samples) is not int or samples < 2:
        raise ValueError("Bundle validation must validate at least two task examples")
    if report.get("files") != inventory(run):
        raise ValueError("Bundle validation modified the bundle")
    report["capability_coverage"] = result.get("capability_coverage", {})
    validate_capability_coverage(definition, report)
    report.update(bundle_verified=True, bundle_validation=result)
    write_report(run, report)
