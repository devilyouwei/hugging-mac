"""Model-independent orchestration. Model definitions are injected by the CLI."""

from __future__ import annotations

import asyncio
import gc
import importlib.metadata
from pathlib import Path
from typing import Any

from . import coreai, coreml, hub
from .artifacts import create_run, default_source_root, inventory, source_directory, write_report
from .contracts import ModelDefinition, Runtime
from .validation import check_tensors, validate_bundle, validate_capability_coverage, verify_sdk


def build_probe(
    definition: ModelDefinition,
    variant: str,
    runtime: Runtime,
    output: Path,
    revision: str = "main",
    source_root: Path | None = None,
    offline: bool = False,
    coreai_device: str = "auto",
) -> Path:
    if runtime not in {"coreml", "coreai"}:
        raise ValueError(f"Unsupported conversion runtime: {runtime}")
    config = definition.config.variant(variant)
    run = create_run(output, definition.config.name, variant, runtime)
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": definition.config.name,
        "variant": variant,
        "config": definition.snapshot(variant),
        "runtime": runtime,
        "requested_revision": revision,
        "status": "building",
        "sdk_integrated": False,
        "artifact_scope": "diagnostic",
        "coreai_device": coreai_device if runtime == "coreai" else None,
        "coreai_optimize": False if runtime == "coreai" else None,
        "capability_coverage": {
            name: {"status": "untested"} for name in definition.config.required_capabilities
        },
    }

    def stage(name: str) -> None:
        report["stage"] = name
        write_report(run, report)
        print(f"[{definition.config.name}/{variant}/{runtime}] {name}", flush=True)

    try:
        stage("check-environment")
        packages = ["torch", "huggingface-hub", *config.probe.packages]
        packages += ["coremltools"] if runtime == "coreml" else ["coreai-torch", "coreai-core"]
        report["versions"] = {p: importlib.metadata.version(p) for p in packages}
        torch = importlib.import_module("torch")
        stage("resolve-source")
        resolved = revision if offline else hub.resolve_revision(config, revision)
        report["source_revision"] = resolved
        stage("download")
        destination = source_directory(
            source_root or default_source_root(), definition.config.name, variant, resolved
        )
        source = (
            hub.local_source(config, resolved, destination)
            if offline
            else hub.download_probe(config, resolved, destination)
        )
        report["source_path"] = str(source.resolve())
        stage("prepare-graph")
        graph = definition.prepare(config, source)
        with torch.no_grad():
            expected = graph.module(*graph.args).float().numpy()
        stage("convert")
        artifact = (coreml if runtime == "coreml" else coreai).export(graph, run)
        inputs = {
            name: arg.numpy() for name, arg in zip(graph.input_names, graph.args, strict=True)
        }
        output_name = graph.output_name
        del graph  # Release source weights before compiling the converted runtime.
        gc.collect()  # Locally defined graph classes can retain weights in reference cycles.
        stage("validate-tensors")
        outputs = (
            coreml.predict(artifact, inputs)
            if runtime == "coreml"
            else asyncio.run(coreai.predict(artifact, inputs, coreai_device))
        )
        report["validation"] = check_tensors(outputs[output_name], expected)
        hub.package_assets(config, source, run)
        (run / "README.md").write_text(
            f"# {definition.config.name} / {variant} ({runtime})\n\n"
            f"Source: https://huggingface.co/{config.source}/tree/{resolved}\n\n"
            f"Diagnostic only: {config.probe.description}.\n\n"
            "This is not a complete model release. See build.json for coverage and validation.\n"
        )
        report.update(status="diagnostic-validated", artifact=artifact.name, files=inventory(run))
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_report(run, report)
        print(f"Build report: {run / 'build.json'}", flush=True)
    return run


def publish(
    definition: ModelDefinition,
    run: Path,
    report: dict[str, Any],
    *,
    dry_run: bool = True,
) -> str:
    validate_bundle(definition, run, report)
    if not report.get("sdk_integrated"):
        raise ValueError("Publication requires SDK capability task parity")
    validate_capability_coverage(definition, report)
    config = definition.config.variant(report["variant"])
    if dry_run:
        return f"Would upload {run} to {config.target}/{report['runtime']}"
    verify_sdk(definition, run, report)
    return hub.upload(config, run, report["runtime"], report["source_revision"])
