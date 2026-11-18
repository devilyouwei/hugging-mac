"""Build the complete Vela variant, including every upstream task adapter."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

from hugging_mac_conversion.models.vela.utils.full_host import adapter_files

from ...registry import get_model
from ...shared.artifacts import create_run, default_output_root, inventory, write_report
from ...shared.hub import local_source
from .utils.components import architecture_digest, export_component, graph_kind


def read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 10_000_000:
            raise ValueError("Unexpected safetensors header size")
        return dict(json.loads(stream.read(size)))


def bind_states(graph: dict[str, Any], filename: str, headers: dict[str, Any]) -> dict[str, Any]:
    states = {}
    for key, spec in graph["states"].items():
        tensor = spec["name"]
        weight_file = filename
        if tensor not in headers[weight_file]:
            weight_file = "model.safetensors"
            tensor = tensor.replace(".base.", ".")
        original = headers[weight_file][tensor]
        if original["shape"] != spec["shape"] or original["dtype"] != "F32":
            raise ValueError(f"Graph state differs from source tensor: {filename}/{tensor}")
        states[key] = {"file": weight_file, "tensor": tensor, "shape": spec["shape"]}
    used = {row["tensor"] for row in states.values() if row["file"] == filename}
    if used != set(headers[filename]) - {"__metadata__"}:
        raise ValueError(f"Unaccounted checkpoint parameters: {filename}")
    return states


def assemble(source: Path, output: Path, variant: str) -> Path:
    original, destination = source.resolve(), output.resolve()
    if (
        original == destination
        or original.is_relative_to(destination)
        or destination.is_relative_to(original)
    ):
        raise ValueError("Original snapshot and converted bundle must not overlap")
    definition = get_model("vela")
    config = definition.config.variant(variant)
    provenance = json.loads((source / ".source.json").read_text())
    local_source(config, provenance["revision"], source)
    files = {"decision": "model.safetensors", **adapter_files(source)}
    if set(files.values()) != {p.name for p in source.glob("*.safetensors")}:
        raise ValueError("Every upstream checkpoint must be represented by a component")
    headers = {name: read_header(source / name) for name in files.values()}
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": "vela",
        "variant": variant,
        "config": definition.snapshot(variant),
        "runtime": "coreai",
        "artifact_scope": "full",
        "status": "building",
        "source_revision": provenance["revision"],
        "source_path": str(source.resolve()),
        "bundle_verified": False,
    }
    write_report(output, report)
    try:
        components = {}
        for name, filename in files.items():
            header = None if name == "decision" else headers[filename]
            kind = graph_kind(header)
            graph_dir = output / "graphs" / kind
            if not (graph_dir / "graph.json").exists():
                print("Exporting", kind, flush=True)
                export_component(source, header, graph_dir)
                gc.collect()
            graph = json.loads((graph_dir / "graph.json").read_text())
            if graph.get("weight_binding") != "inputs":
                raise ValueError("Rebuild mutable-state graph as immutable weight inputs")
            if graph.get("architecture_sha256") != architecture_digest(source):
                raise ValueError("Resume graph architecture differs from the original snapshot")
            with (graph_dir / "model.aimodel/main.mlirb").open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != graph["graph_sha256"]:
                    raise ValueError("A previously exported graph changed")
            states = bind_states(graph, filename, headers)
            components[name] = {"graph": str(graph_dir.relative_to(output)), "states": states}
        weights = output / "weights"
        weights.mkdir(exist_ok=True)
        for filename in headers:
            destination = weights / filename
            if not destination.exists():
                try:
                    os.link(source / filename, destination)
                except OSError:
                    shutil.copyfile(source / filename, destination)
        host_files = (
            "render31.py",
            "ub_d3_common.py",
            "vela3_encoder.py",
            "vela3_schema.py",
            "configuration_ultrabert.py",
            "modeling_ultrabert.py",
            "LICENSE",
        )
        for path in [*source.glob("*.json"), *(source / name for name in host_files)]:
            shutil.copyfile(path, output / path.name)
        shutil.copyfile(Path(__file__).with_name("NOTICE"), output / "NOTICE")
        for filename in ("README.md", "README.zh-CN.md"):
            shutil.copyfile(
                Path(__file__).with_name("publication") / "full" / filename, output / filename
            )
        manifest = {
            "format": "vela-full-coreai/1",
            "source": config.source,
            "source_revision": provenance["revision"],
            "variant": variant,
            "components": components,
            "precision": "float32",
            "lora_merged": False,
            "neural_runtime": "coreai",
            "weight_storage": "exact original tensors as cached immutable Core AI inputs",
            "weight_binding": "inputs",
            "sequence_length": {"decision": 32768, "schema_window": 8192, "evidence": 32768},
            "dynamic_axes": ["batch", "sequence", "questions", "options", "units", "labels"],
            "source_files": provenance["files"],
            "capability_coverage": {
                name: {"status": "untested"} for name in definition.config.required_capabilities
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report.update(
            status="converted-unvalidated",
            capability_coverage=manifest["capability_coverage"],
            files=inventory(output),
        )
        write_report(output, report)
        return output
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_report(output, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variant", default="3_1-fast")
    parser.add_argument("--resume", type=Path, help="Resume this variant's unfinished graph build")
    args = parser.parse_args()
    output = args.resume or create_run(default_output_root(), "vela", args.variant, "coreai")
    if (output / "build.json").exists():
        previous = json.loads((output / "build.json").read_text())
        if (
            previous["variant"] != args.variant
            or Path(previous["source_path"]) != args.source.resolve()
        ):
            raise ValueError("Resume source and variant must match")
    print(assemble(args.source, output, args.variant))


if __name__ == "__main__":
    main()
