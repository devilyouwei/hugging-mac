"""Validate a packaged D3 bundle with conversion-owned native execution."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from ....shared.coreai_runtime import CoreAIProvider
from ..utils.adapter import create_runtime


def load_case_request(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    return dict(
        state=payload["state"],
        questions=payload["questions"],
        images=[str(root / name) for name in payload.get("images", [])] or None,
        videos=[
            list(np.load(root / row["frames_npy"], allow_pickle=False))
            for row in payload.get("videos", [])
        ]
        or None,
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
    manifest = json.loads((root / "manifest.json").read_text())
    provider = CoreAIProvider()
    sessions = []
    rows = []
    try:
        for name in ("decision.aimodel", "vision.aimodel"):
            sessions.append(await provider.create_session(root / name, device="cpu", options={}))
        from ....shared.config import load_config

        model_name = load_config(Path(__file__).parents[1] / "model.yaml").variant().source
        runtime = create_runtime(root, *sessions, model_name=model_name)
        for case in cases:
            row: dict[str, Any] = {"id": case["id"], "passed": False}
            try:
                actual = await asyncio.to_thread(
                    runtime.system_one, **load_case_request(root, case["payload"])
                )
                assert_response_close(actual["answers"], case["expected"]["answers"])
                assert_response_close(actual["usage"], case["expected"]["usage"])
                row["passed"] = True
            except (AssertionError, ValueError, RuntimeError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
    finally:
        for session in reversed(sessions):
            await session.close()
    passed = bool(rows) and all(row["passed"] for row in rows)
    return {
        "passed": passed,
        "samples": len(rows),
        "cases": rows,
        "capability_coverage": manifest["capability_coverage"] if passed else {},
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
