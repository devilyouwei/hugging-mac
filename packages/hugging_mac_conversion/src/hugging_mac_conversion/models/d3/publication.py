"""Assemble a tested D3 multimodal bundle without original neural weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from hugging_mac_sdk.models.d3.utils.adapter import create_runtime
from PIL import Image

from ...shared.config import load_config
from ...shared.hub import source_inventory
from .evaluation.native import fingerprint, suite
from .evaluation.sdk_validation import assert_response_close


def copy_asset(source: Path, target: Path) -> None:
    target.mkdir()
    for path in source.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError("Expected plain Core AI artifact files")
        os.link(path, target / path.name)


def assemble(
    source: Path,
    decision_build: Path,
    vision: Path,
    evaluation: Path,
    text_evaluation: Path,
    sdk_wheel: Path,
    output: Path,
) -> None:
    source_report = json.loads((source / ".source.json").read_text())
    if source_inventory(source) != source_report["files"]:
        raise ValueError("Original source snapshot changed")
    build = json.loads((decision_build / "build.json").read_text())
    native = json.loads(evaluation.read_text())
    text = json.loads(text_evaluation.read_text())
    if (
        build["status"] != "diagnostic-validated"
        or native["status"] != "passed"
        or text["status"] != "diagnostic-passed"
    ):
        raise ValueError("All conversion and evaluation gates must pass")
    if Path(native["decision"]).resolve() != (decision_build / "model.aimodel").resolve():
        raise ValueError("Native evaluation used a different decision artifact")
    if Path(native["vision"]).resolve() != vision.resolve():
        raise ValueError("Native evaluation used a different vision artifact")
    if (
        native["source_digest"]
        != hashlib.sha256((source / ".source.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Native source snapshot differs")
    for expected, actual in zip(native["reference"], native["converted"], strict=True):
        assert_response_close(actual["response"]["answers"], expected["response"]["answers"])
        assert_response_close(actual["response"]["usage"], expected["response"]["usage"])
    runtime = create_runtime(
        source,
        None,
        None,
        model_name=load_config(Path(__file__).with_name("model.yaml")).variant().source,
    )
    cases = suite(runtime)
    if fingerprint(cases) != native["cases_digest"]:
        raise ValueError("Native fixtures changed")
    output.mkdir(parents=True, exist_ok=False)
    copy_asset(decision_build / "model.aimodel", output / "decision.aimodel")
    copy_asset(vision, output / "vision.aimodel")
    names = [
        "LICENSE",
        "config.json",
        "decision_config.json",
        "d3_runtime.py",
        "d3_format.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
    ]
    for name in names:
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(Path(__file__).with_name("NOTICE"), output / "NOTICE")
    shutil.copyfile(sdk_wheel, output / sdk_wheel.name)
    fixtures = output / "fixtures"
    fixtures.mkdir()

    def portable(value: Any) -> Any:
        if isinstance(value, Image.Image):
            key = fingerprint(value)
            path = fixtures / (key + ".png")
            if not path.exists():
                value.save(path)
            return path.relative_to(output).as_posix()
        if isinstance(value, np.ndarray):
            key = fingerprint(value)
            path = fixtures / (key + ".npy")
            if not path.exists():
                np.save(path, value, allow_pickle=False)
            return {"frames_npy": path.relative_to(output).as_posix()}
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        return value

    validation_cases = []
    for case, reference in zip(cases, native["reference"], strict=True):
        validation_cases.append(
            {
                "id": case["id"],
                "payload": portable(case["payload"]),
                "expected": reference["response"],
            }
        )
    originals = json.loads(
        Path(__file__).with_name("evaluation").joinpath("cases/decisions.json").read_text()
    )
    reference_by_id = {row["id"]: row for row in text["reference_cases"]}
    for case in originals:
        payload = {
            "state": case["payload"]["state"],
            "questions": {case["id"]: case["payload"]["question"]},
        }
        prepared = runtime.prepare(**payload)
        expected = runtime.respond(
            prepared,
            {case["id"]: reference_by_id[case["id"]]["probabilities"]},
            len(prepared.sequences[case["id"]]),
        )
        validation_cases.append({"id": case["id"], "payload": payload, "expected": expected})
    invalid = {"state": "test", "questions": {"bad": {"type": "unknown"}}}
    validation_cases.append(
        {
            "id": "invalid-question-contract",
            "payload": invalid,
            "expected": runtime.system_one(**invalid),
        }
    )
    (output / "validation_cases.json").write_text(
        json.dumps(validation_cases, ensure_ascii=False, indent=2) + "\n"
    )
    capabilities = {
        name: {"status": "passed", "evidence": "evaluation.json and validation_cases.json"}
        for name in build["capability_coverage"]
    }
    manifest = {
        "source": source_report["repo_id"],
        "source_revision": source_report["revision"],
        "format": "Apple Core AI .aimodel",
        "function": "main",
        "output": "logits",
        "precision": "float32",
        "weight_changes": "Exact promotion of original BF16 values; no training or quantization",
        "dynamic_sequence_length": True,
        "dynamic_vision_patch_count": True,
        "max_length": runtime.max_length,
        "export_sequence_bound": build["config"]["settings"]["probe"]["dynamic_sequence_length"],
        "verified_sequence_lengths": [1024, 1536],
        "inputs": [
            "input_ids",
            "attention_mask",
            "position_ids",
            "visual_embeddings",
            "visual_mask",
        ],
        "vision_inputs": ["pixels", "indices", "weights", "positions"],
        "vision_output": "features",
        "modalities": ["text", "images", "videos"],
        "decision_types": ["choice", "noul", "score"],
        "maximum_answer_codes": 255,
        "media_processing": "Unmodified pinned upstream processor and runtime",
        "tested_device": "cpu",
        "experimental": True,
        "capability_coverage": capabilities,
        "source_files_sha256": {name: source_report["files"][name] for name in names},
        "source_weights_sha256": {
            n: v for n, v in source_report["files"].items() if n.endswith(".safetensors")
        },
        "export_versions": build["versions"],
        "tensor_validation": build["validation"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    public = {
        "text_cases": text["candidates"][0]["cases"],
        "multimodal_checks": native["checks"],
        "reference": native["reference"],
        "converted": native["converted"],
        "scope": "Smoke parity, not a quality benchmark or maximum-context stress test",
    }
    (output / "evaluation.json").write_text(json.dumps(public, indent=2) + "\n")
    (output / ".python-version").write_text("3.12\n")
    (output / "pyproject.toml").write_text(
        '[project]\nname = "d3-edge-coreai-example"\nversion = "0.1.0"\n'
        'requires-python = ">=3.12,<3.14"\n'
        'dependencies = ["hugging-mac-sdk[d3]"]\n\n[tool.uv]\npackage = false\n\n'
        "[tool.uv.workspace]\nmembers = []\n\n[tool.uv.sources]\n"
        f'hugging-mac-sdk = {{path = "{sdk_wheel.name}"}}\n'
    )
    print(output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "decision-build",
        "vision",
        "evaluation",
        "text-evaluation",
        "sdk-wheel",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assemble(
        args.source,
        args.decision_build,
        args.vision,
        args.evaluation,
        args.text_evaluation,
        args.sdk_wheel,
        args.output,
    )


if __name__ == "__main__":
    main()
