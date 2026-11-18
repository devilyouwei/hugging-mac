"""Stage an explicitly requested experimental release; never changes full-release gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from ....shared.artifacts import inventory
from ....shared.evaluation.contracts import EvaluationCase
from ....shared.hub import source_inventory
from ....shared.imports import source_modules
from ..converter import inputs_for_record


def stage(build: Path, evaluation: Path, source: Path, output: Path) -> None:
    report = json.loads((evaluation / "report.json").read_text())
    build_report = json.loads((build / "build.json").read_text())
    source_report = json.loads((source / ".source.json").read_text())
    if report["model"] != "vela" or report["variant"] != "3-fast":
        raise ValueError("Vela3-fast evaluation required")
    if report["status"] != "diagnostic-passed" or report["scope"] != "diagnostic":
        raise ValueError("A passed diagnostic evaluation is required")
    candidate = next(c for c in report["candidates"] if Path(c["run"]).resolve() == build.resolve())
    if candidate["runtime"] != "coreai" or candidate["status"] != "diagnostic-passed":
        raise ValueError("A passed Core AI candidate is required")
    if inventory(build) != candidate["files"]:
        raise ValueError("Artifact changed after evaluation")
    revision = report["source_revision"]
    if build_report["source_revision"] != revision or source_report["revision"] != revision:
        raise ValueError("Source revisions differ")
    if source_report["repo_id"] != "vllm-sr/vela3-fast":
        raise ValueError("Unexpected upstream repository")
    if source_inventory(source) != source_report["files"]:
        raise ValueError("Original snapshot changed")
    if report.get("environment", {}).get("coreai_device") != "cpu":
        raise ValueError("This publication requires a passed CPU evaluation")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copytree(build / "model.aimodel", output / "model.aimodel")
    for name in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        shutil.copyfile(source / name, output / name)
    template = Path(__file__).parent
    for name in (
        "vela_coreai.py",
        "example.py",
        "smoke.py",
        "verify_coreai.swift",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "README.zh-CN.md",
    ):
        shutil.copyfile(template / name, output / name)
    (output / ".python-version").write_text("3.12\n")
    notice = (template.parent / "NOTICE").read_text()
    (output / "NOTICE").write_text(
        notice + "\nThe standalone host adapter also adapts rendering from vela3_encoder.py "
        "and ub_d3_common.py at the same upstream commit. Changes: tokenizers/NumPy input "
        "packing, explicit bounds and unsupported-task rejection, Core AI inference, "
        "and Python/Swift examples. No model training or fine-tuning.\n"
    )
    config = json.loads((source / "config.json").read_text())
    decision = {key: config[key] for key in ("bos_token_id", "eos_token_id", "pad_token_id")} | {
        "marker_ids": config["decision"]["marker_ids"]
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    probe = build_report["config"]["settings"]["probe"]
    dynamic = probe.get("dynamic_sequence_length")
    capacity = dynamic or probe["sequence_length"]
    sequence_shape = "sequence" if dynamic else capacity
    manifest = {
        "repository": "hugging-mac/vela3-fast-CoreAI",
        "source": "vllm-sr/vela3-fast",
        "source_revision": revision,
        "experimental": True,
        "complete_upstream_model": False,
        "bundle_verified": False,
        "artifact_scope": "decision-only",
        "format": "Core AI .aimodel",
        "function": "main",
        "output": "logits",
        "sequence_length": capacity,
        "dynamic_sequence_length": bool(dynamic),
        "batch_size": 1,
        "option_count": "dynamic" if dynamic else 2,
        "tested_device": "cpu",
        "inputs": {
            "input_ids": {"dtype": "int32", "shape": [1, sequence_shape]},
            "attention_mask": {"dtype": "int32", "shape": [1, sequence_shape]},
            "option_positions": {"dtype": "int32", "shape": [1, "options" if dynamic else 2]},
            "pool_mask": {"dtype": "float32", "shape": [1, sequence_shape]},
        },
        "source_weights_sha256": source_report["files"]["model.safetensors"],
        "artifact_files_sha256": candidate["files"],
        "export_versions": build_report["versions"],
        "evaluation_versions": report["versions"],
        "upstream_capability_coverage": report["capability_coverage"],
        "initial_build_validation": build_report["validation"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Generate tokenizer fixtures from upstream code, without constructing neural weights.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=True)
    proxy = SimpleNamespace(
        tok=tokenizer,
        mk=decision["marker_ids"],
        max_len=capacity,
        _ids=lambda value: tokenizer(value, add_special_tokens=False)["input_ids"] if value else [],
    )
    cases = json.loads((template.parent / "evaluation/cases/decisions.json").read_text())
    typed_cases = [EvaluationCase.model_validate(case).model_dump(mode="json") for case in cases]
    cases_hash = hashlib.sha256(json.dumps(typed_cases, sort_keys=True).encode()).hexdigest()
    if cases_hash != report["cases_sha256"]:
        raise ValueError("Case contents changed after evaluation")
    reference = {row["id"]: row for row in report["reference_cases"]}
    if [row["id"] for row in cases] != list(reference):
        raise ValueError("Evaluation cases changed; rerun evaluation before staging")
    with source_modules(source):
        encoder = importlib.import_module("vela3_encoder").DecisionEncoder
        for case in cases:
            record = encoder.encode(proxy, case["payload"]["state"], case["payload"]["question"])
            args = inputs_for_record(proxy, record, len(record["ids"]) if dynamic else capacity)
            case["inputs"] = {
                name: value.numpy().ravel().tolist()
                for name, value in zip(manifest["inputs"], args, strict=True)
            }
            case["reference_probabilities"] = reference[case["id"]]["probabilities"]
            case["reference_decision"] = reference[case["id"]]["decision"]
    (output / "validation_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n"
    )
    public_report = {
        key: report[key]
        for key in (
            "model",
            "variant",
            "scope",
            "status",
            "source_revision",
            "versions",
            "cases_sha256",
            "evaluation_config",
            "capability_coverage",
            "reference_cases",
        )
    }
    public_report["candidate"] = {key: candidate[key] for key in ("runtime", "status", "cases")}
    (output / "evaluation.json").write_text(json.dumps(public_report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experimental", action="store_true", required=True)
    args = parser.parse_args()
    stage(args.build, args.evaluation, args.source, args.output)


if __name__ == "__main__":
    main()
