"""Original FP32 oracle versus all Core AI components and complete public responses."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch
from hugging_mac_sdk.models.vela.full import VelaFullCoreAI, decision_host
from hugging_mac_sdk.models.vela.utils.full_host import adapter_files, make_host
from hugging_mac_sdk.models.vela.utils.full_upstream import load_full_upstream
from hugging_mac_sdk.models.vela.utils.validation import compare

from ....registry import get_model
from ....shared.artifacts import inventory, write_report
from ....shared.hub import local_source
from ....shared.validation import check_tensors, validate_bundle, validate_capability_coverage
from ..components import sample_inputs
from .full_cases import cases


class OriginalComponents:
    def __init__(self, source: Path) -> None:
        self.upstream = load_full_upstream(source)
        self.base = self.upstream.DecisionEncoder.load(source, device="cpu")
        self.base.span = False
        self.source = source
        self.files = adapter_files(source)
        self.name: str | None = None
        self.net: Any = None

    def run(self, name: str, *arrays: Any) -> tuple[Any, Any]:
        if name != self.name:
            self.net = None
            gc.collect()
            self.net = (
                self.base
                if name == "decision"
                else self.upstream.schema.build_adapter(
                    self.base, self.source / self.files[name], "cpu"
                )
            )
            self.name = name
        args = [torch.as_tensor(value, dtype=torch.long) for value in arrays]
        with torch.no_grad():
            hidden = self.net.encoder(input_ids=args[0], attention_mask=args[1]).last_hidden_state
            result = self.upstream.schema.SchemaNet.heads(self.net, hidden, *args[2:])
        return tuple(value.float().numpy() for value in result)


def evaluate(source: Path, bundle: Path, output: Path, reference: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bundle / "manifest.json").read_text())
    definition = get_model("vela")
    build_before = json.loads((bundle / "build.json").read_text())
    if build_before["config"] != definition.snapshot(manifest["variant"]):
        raise ValueError("Conversion configuration changed")
    local_source(
        definition.config.variant(manifest["variant"]), manifest["source_revision"], source
    )
    if inventory(bundle) != build_before["files"]:
        raise ValueError("Converted bundle changed before evaluation")
    source_digest = hashlib.sha256((source / ".source.json").read_bytes()).hexdigest()
    expected_cases = cases(source, decision_host(source, cast(Any, None)))
    cases_digest = hashlib.sha256(
        json.dumps(expected_cases, ensure_ascii=False).encode()
    ).hexdigest()
    if reference is None:
        original = OriginalComponents(source)
        fixtures = cases(source, original.base)
        host = make_host(
            source, original.base, lambda name: lambda *args: original.run(name, *args)
        )
        rows = []
        for case in fixtures:
            start = perf_counter()
            result = host.system_one(case["state"], case["questions"])
            rows.append({"id": case["id"], "response": result})
            print("ORIGINAL", case["id"], round(perf_counter() - start, 2), flush=True)
        args = tuple(x.numpy() for x in sample_inputs())
        tensors = {}
        for name in manifest["components"]:
            for index, value in enumerate(original.run(name, *args)):
                tensors[f"{name}__{index}"] = value
        np.savez(output / "reference_tensors.npz", **tensors)
        record = {
            "source_digest": source_digest,
            "cases_digest": cases_digest,
            "cases": fixtures,
            "responses": rows,
        }
        (output / "reference.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        del host, original
        gc.collect()
    else:
        record = json.loads((reference / "reference.json").read_text())
        if record["source_digest"] != source_digest or record["cases_digest"] != cases_digest:
            raise ValueError("Original source reference mismatch")
        fixtures, rows = record["cases"], record["responses"]
        (output / "reference.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        import shutil

        shutil.copyfile(reference / "reference_tensors.npz", output / "reference_tensors.npz")
    checks: list[dict[str, Any]] = []
    tensor_checks: list[dict[str, Any]] = []
    with VelaFullCoreAI(bundle) as native:
        expected_tensors = np.load(output / "reference_tensors.npz")
        args = tuple(x.numpy() for x in sample_inputs())
        for name in manifest["components"]:
            check = {"component": name, "passed": False}
            try:
                actual = native._backend.run(name, *args)
                check["outputs"] = [
                    check_tensors(value, expected_tensors[f"{name}__{i}"])
                    for i, value in enumerate(actual)
                ]
                check["passed"] = True
            except Exception as error:
                check["error"] = f"{type(error).__name__}: {error}"
            tensor_checks.append(check)
            (output / "progress.json").write_text(
                json.dumps(
                    {"tensor_checks": tensor_checks, "checks": checks}, ensure_ascii=False, indent=2
                )
            )
            print("NATIVE TENSOR", name, check["passed"], flush=True)
        native._backend.used.clear()
        for case, expected in zip(fixtures, rows, strict=True):
            check = {"id": case["id"], "passed": False}
            try:
                response = native.system_one(case["state"], case["questions"])
                check["response"] = response
                check["max_absolute_error"] = compare(response, expected["response"])
                check["passed"] = True
            except Exception as error:
                check["error"] = f"{type(error).__name__}: {error}"
            checks.append(check)
            (output / "progress.json").write_text(
                json.dumps(
                    {"tensor_checks": tensor_checks, "checks": checks}, ensure_ascii=False, indent=2
                )
            )
            print("NATIVE", case["id"], check["passed"], flush=True)
        exercised = native.used_components
    passed = all(x["passed"] for x in checks + tensor_checks) and exercised == set(
        manifest["components"]
    )
    report = {
        "passed": passed,
        "samples": len(checks),
        "checks": checks,
        "tensor_checks": tensor_checks,
        "source_revision": manifest["source_revision"],
        "source_digest": source_digest,
        "atol": 0.02,
        "rtol": 0.02,
        "device": "cpu",
        "components_exercised": sorted(exercised),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    build = json.loads((bundle / "build.json").read_text())
    build["status"] = "task-validated" if passed else "failed-evaluation"
    build["evaluation"] = str(output.resolve())
    write_report(bundle, build)
    if not passed:
        raise SystemExit(1)
    portable = [
        {**case, "expected": row["response"]} for case, row in zip(fixtures, rows, strict=True)
    ]
    (bundle / "validation_cases.json").write_text(
        json.dumps(portable, ensure_ascii=False, indent=2)
    )
    (bundle / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    coverage = {
        name: {
            "status": "passed",
            "evidence": "evaluation.json: complete responses and all component routes",
        }
        for name in manifest["capability_coverage"]
    }
    manifest["capability_coverage"] = coverage
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sdk_validation = {
        "passed": True,
        "samples": len(checks),
        "capability_coverage": coverage,
        "components_exercised": sorted(exercised),
    }
    (bundle / "sdk_validation.json").write_text(json.dumps(sdk_validation, indent=2))
    build.update(
        status="validated",
        sdk_integrated=True,
        sdk_validation=sdk_validation,
        capability_coverage=coverage,
        files=inventory(bundle),
    )
    definition = get_model("vela")
    validate_capability_coverage(definition, build)
    validate_bundle(definition, bundle, build)
    write_report(bundle, build)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    evaluate(args.source, args.bundle, args.output, args.reference)


if __name__ == "__main__":
    main()
