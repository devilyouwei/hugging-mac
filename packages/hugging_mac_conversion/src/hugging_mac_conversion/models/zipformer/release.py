"""Build, verify, and optionally publish the Zipformer streaming Core ML bundle."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ...shared import artifacts
from .converter import export, prepare_sources, sha256
from .definition import DEFINITION, VARIANT
from .evaluation import validate

ASSETS = Path(__file__).with_name("publication")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def build(output_root: Path, source: Path, reference: Path) -> Path:
    run = artifacts.create_run(output_root, DEFINITION.name, DEFINITION.default_variant, "coreml")
    print(f"Build directory: {run}", flush=True)
    report: dict[str, Any] = {"status": "building"}
    artifacts.write_report(run, report)
    try:
        graphs = export(source, reference, run)
        for name in ("README.md", "LICENSE", "NOTICE", "spotter.py", "keywords.txt"):
            shutil.copyfile(ASSETS / name, run / name)
        shutil.copyfile(reference / "tokens.txt", run / "tokens.txt")
        manifest = {
            "source": VARIANT.source,
            "source_revision": VARIANT.revision,
            "source_files_sha256": VARIANT.source_files,
            "upstream_checkpoint_revision": VARIANT.upstream_checkpoint_revision,
            "upstream_checkpoint_sha256": VARIANT.upstream_checkpoint_sha256,
            "reference_url": VARIANT.archive,
            "reference_sha256": VARIANT.archive_sha256,
            "reference_graphs_sha256": {
                part: sha256(reference / f"{part}-{VARIANT.graph_suffix}") for part in graphs
            },
            "precision": "float32",
            "minimum_deployment_target": "iOS17/macOS14",
            "features_per_chunk": VARIANT.features_per_chunk,
            "features_with_lookahead": VARIANT.features_with_lookahead,
            "graphs": graphs,
            "tool_versions": {
                name: version(name)
                for name in (
                    "torch",
                    "coremltools",
                    "transformers",
                    "numpy",
                    "onnxruntime",
                    "kaldi-native-fbank",
                )
            },
            "validation_platform": platform.platform(),
        }
        write_json(run / "manifest.json", manifest)
        write_json(run / "evaluation.json", validate(run, reference, manifest))
        report = {
            "status": "validated",
            "target": VARIANT.target,
            "files": artifacts.inventory(run),
        }
    except Exception as error:
        report = {"status": "failed", "error": str(error)}
        raise
    finally:
        artifacts.write_report(run, report)
    return run


def verify(run: Path, reference: Path) -> dict[str, Any]:
    report = json.loads((run / "build.json").read_text())
    if report["status"] != "validated" or report["target"] != VARIANT.target:
        raise ValueError("Expected a validated Zipformer release for this target")
    if artifacts.inventory(run) != report["files"]:
        raise ValueError("Release files changed after validation")
    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest["source_revision"], manifest["reference_sha256"]) != (
        VARIANT.revision,
        VARIANT.archive_sha256,
    ):
        raise ValueError("Release provenance differs from the recipe")
    return validate(run, reference, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--offline", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("build")
    convert.add_argument("--output", type=Path)
    commands.add_parser("verify").add_argument("run", type=Path)
    publish = commands.add_parser("push")
    publish.add_argument("run", type=Path)
    publish.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    source, reference = prepare_sources(
        options.source_root or artifacts.default_source_root(), offline=options.offline
    )
    if options.command == "build":
        print(build(options.output or artifacts.default_output_root(), source, reference))
        return

    result = verify(options.run, reference)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if options.command == "push":
        print(f"Target: {VARIANT.target}")
        if options.execute:
            from huggingface_hub import HfApi

            api = HfApi()
            api.whoami()
            api.create_repo(VARIANT.target, repo_type="model", exist_ok=True)
            print(
                api.upload_folder(
                    repo_id=VARIANT.target,
                    repo_type="model",
                    folder_path=options.run,
                    ignore_patterns=["__pycache__/**", ".cache/**"],
                    commit_message=f"Publish Zipformer Core ML from {VARIANT.revision[:12]}",
                )
            )


if __name__ == "__main__":
    main()
