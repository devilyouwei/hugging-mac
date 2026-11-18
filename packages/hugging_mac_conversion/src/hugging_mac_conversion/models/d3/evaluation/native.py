"""Evaluate original D3 and native Core AI with identical unmodified media processing."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from hugging_mac_conversion.models.d3.utils.adapter import create_runtime

from ....shared.config import load_config
from ....shared.imports import source_modules
from .multimodal_cases import cases as media_cases


def fingerprint(value: Any) -> str:
    """Hash actual media pixels and all request fields for safe oracle reuse."""
    from PIL import Image

    if isinstance(value, (np.ndarray, Image.Image)):
        array = np.asarray(value)
        data = str((array.shape, array.dtype)).encode() + array.tobytes()
    elif isinstance(value, dict):
        data = json.dumps({k: fingerprint(v) for k, v in value.items()}, sort_keys=True).encode()
    elif isinstance(value, (list, tuple)):
        data = json.dumps([fingerprint(v) for v in value]).encode()
    else:
        data = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def suite(runtime: Any) -> list[dict[str, Any]]:
    rows = media_cases()
    question = {
        "type": "choice",
        "instructions": "What is the topic?",
        "criteria": {"coding": "Software", "cooking": "Food", "sports": "Athletics"},
    }
    for width in [1024, 1536]:
        count = 0
        state = ""
        for _ in range(8):
            count = len(runtime.prepare(state, {"topic": question}).sequences["topic"])
            if count == width:
                break
            state = " code" * (state.count(" code") + width - count)
        if count != width:
            raise ValueError("Unable to generate exact-length fixture")
        rows.append(
            {"id": f"text-{width}", "payload": {"state": state, "questions": {"topic": question}}}
        )
    rows.append(
        {
            "id": "structured-multiple-questions",
            "payload": {
                "state": {"request": "Write Python code", "language": "Python"},
                "questions": {
                    "topic": question,
                    "software": {"type": "noul", "instructions": "Is this about software?"},
                },
            },
        }
    )
    return rows


def run_cases(runtime: Any, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        start = perf_counter()
        try:
            prepared = runtime.prepare(**case["payload"])
            if prepared.errors:
                raise ValueError(str(prepared.errors))
            probabilities, usage = runtime.run(prepared)
            response = runtime.respond(prepared, probabilities, usage)
            row = {
                "id": case["id"],
                "probabilities": probabilities,
                "response": response,
                "token_lengths": prepared.lengths,
                "elapsed_seconds": perf_counter() - start,
            }
            rows.append(row)
            print(case["id"], "completed", row["elapsed_seconds"], flush=True)
        except Exception as error:
            rows.append({"id": case["id"], "error": f"{type(error).__name__}: {error}"})
            print(case["id"], rows[-1]["error"], flush=True)
    return rows


async def evaluate(
    source: Path, decision: Path, vision: Path, output: Path, reference_report: Path | None = None
) -> None:
    from hugging_mac_conversion.shared.coreai_runtime import CoreAIProvider

    from .bundle_validation import assert_response_close

    output.mkdir(parents=True, exist_ok=False)
    with source_modules(source):
        upstream = importlib.import_module("d3_runtime")
    model_name = load_config(Path(__file__).parents[1] / "model.yaml").variant().source
    original = (
        upstream.D3(source, device="cpu", batch_size=1, model_name=model_name)
        if reference_report is None
        else create_runtime(
            source,
            None,
            None,
            model_name=load_config(Path(__file__).parents[1].joinpath("model.yaml"))
            .variant()
            .source,
        )
    )
    cases = suite(original)
    source_digest = hashlib.sha256((source / ".source.json").read_bytes()).hexdigest()
    cases_digest = fingerprint(cases)
    if reference_report is None:
        reference = run_cases(original, cases)
    else:
        previous = json.loads(reference_report.read_text())
        if previous["source_digest"] != source_digest or previous["cases_digest"] != cases_digest:
            raise ValueError("Reference source or cases differ")
        reference = previous["reference"]
    (output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    del original
    gc.collect()
    provider = CoreAIProvider()
    decision_session = await provider.create_session(decision, device="cpu", options={})
    vision_session = await provider.create_session(vision, device="cpu", options={})
    try:
        native = create_runtime(
            source,
            decision_session,
            vision_session,
            model_name=load_config(Path(__file__).parents[1].joinpath("model.yaml"))
            .variant()
            .source,
        )
        actual = await asyncio.to_thread(run_cases, native, cases)
    finally:
        await vision_session.close()
        await decision_session.close()
    checks = []
    for ref, got in zip(reference, actual, strict=True):
        check: dict[str, Any] = {"id": ref["id"], "passed": False}
        try:
            assert "error" not in ref and "error" not in got, (ref.get("error"), got.get("error"))
            assert ref["token_lengths"] == got["token_lengths"]
            assert ref["response"]["usage"] == got["response"]["usage"]
            assert_response_close(got["response"]["answers"], ref["response"]["answers"])
            errors = []
            for key, expected in ref["probabilities"].items():
                values = got["probabilities"][key]
                np.testing.assert_allclose(values, expected, atol=0.02, rtol=0.02)
                assert int(np.argmax(values)) == int(np.argmax(expected))
                errors.append(float(np.max(np.abs(np.array(values) - expected))))
                assert (
                    ref["response"]["answers"][key].keys() == got["response"]["answers"][key].keys()
                )
            check.update(passed=True, max_probability_error=max(errors))
        except (AssertionError, KeyError, ValueError) as error:
            check["error"] = str(error)
        checks.append(check)
    report = {
        "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "source": str(source),
        "decision": str(decision),
        "vision": str(vision),
        "reference": reference,
        "converted": actual,
        "checks": checks,
        "source_digest": source_digest,
        "cases_digest": cases_digest,
        "reference_report": str(reference_report) if reference_report else None,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(checks, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "decision", "vision", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    args = parser.parse_args()
    asyncio.run(
        evaluate(args.source, args.decision, args.vision, args.output, args.reference_report)
    )


if __name__ == "__main__":
    main()
