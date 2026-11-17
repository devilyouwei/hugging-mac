"""Run D3 through the SDK and validate the packaged upstream reference cases."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from hugging_mac_sdk import ModelSdk, ReusePolicy
from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.models.d3 import D3_MANIFEST, register_d3
from hugging_mac_sdk.schemas.decision import DecisionRequest, DecisionVideoFrames
from hugging_mac_sdk.schemas.detection import ImageInput


def load_case_request(root: Path, payload: dict[str, Any]) -> DecisionRequest:
    from io import BytesIO

    import numpy as np
    from PIL import Image

    def encode(frame: Any) -> ImageInput:
        with BytesIO() as buffer:
            Image.fromarray(frame).save(buffer, format="PNG")
            return ImageInput(data=buffer.getvalue())

    return DecisionRequest(
        state=payload["state"],
        questions=payload["questions"],
        images=tuple(ImageInput(path=root / value) for value in payload.get("images", [])),
        videos=tuple(
            DecisionVideoFrames(
                frames=tuple(
                    encode(frame)
                    for frame in np.load(root / value["frames_npy"], allow_pickle=False)
                )
            )
            for value in payload.get("videos", [])
        ),
    )


def assert_response_close(actual: Any, expected: Any) -> None:
    import numpy as np

    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_response_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_response_close(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.02)
    else:
        assert actual == expected, (actual, expected)


async def _validate_bundle(root: Path) -> dict[str, Any]:
    cases = json.loads((root / "validation_cases.json").read_text())
    sdk = ModelSdk()
    register_d3(sdk.registry)
    handle = await sdk.acquire(
        D3_MANIFEST.model_id,
        runtime="coreai",
        options={"artifact_path": root},
        reuse=ReusePolicy.DEDICATED,
    )
    decisions = handle.require(DecisionMaking)
    rows = []
    try:
        for case in cases:
            row: dict[str, Any] = {"id": case["id"], "passed": False}
            try:
                response = await decisions.decide(load_case_request(root, case["payload"]))
                result = response.model_dump(
                    include={"model", "answers", "usage"}, exclude_none=True
                )
                expected_answers = {
                    key: (answer | {"type": "error", "question_type": answer.get("type")})
                    if isinstance(answer, dict) and "error" in answer
                    else answer
                    for key, answer in case["expected"]["answers"].items()
                }
                # Schema validation owns diagnostic wording; error codes and question
                # types remain part of the comparison with the upstream contract.
                for key, expected in expected_answers.items():
                    if expected.get("type") == "error":
                        message = result["answers"][key].get("message")
                        assert isinstance(message, str) and message
                        expected["message"] = message
                assert_response_close(result["answers"], expected_answers)
                assert_response_close(result["usage"], case["expected"]["usage"])
                row["passed"] = True
            except (AssertionError, ValueError, RuntimeError, InferenceError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            print(case["id"], "passed" if row["passed"] else "FAILED", flush=True)
    finally:
        await handle.close()
    manifest = json.loads((root / "manifest.json").read_text())
    passed = all(row["passed"] for row in rows)
    coverage = (
        manifest["capability_coverage"]
        if passed
        else {
            name: {"status": "incomplete", "evidence": "SDK response validation failed"}
            for name in manifest["capability_coverage"]
        }
    )
    return {
        "passed": passed,
        "samples": len(rows),
        "cases": rows,
        "capability_coverage": coverage,
    }


def validate_bundle(root: Path) -> dict[str, Any]:
    return asyncio.run(_validate_bundle(root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, required=True)
    args = parser.parse_args()
    result = validate_bundle(args.artifact_path)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
