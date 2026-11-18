"""Complete-model acceptance through public SDK contracts, independent of conversion."""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from hugging_mac_sdk import ModelSdk, StructuredPredictionRequest
from hugging_mac_sdk.capabilities import StructuredPrediction
from hugging_mac_sdk.models.vela import register_vela


def assert_response(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_response(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_response(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=0.02, rel=0.02)
    else:
        assert actual == expected


@pytest.mark.smoke
async def test_full_bundle_public_capability() -> None:
    location = os.environ.get("VELA_FULL_COREAI_PATH")
    if not location:
        pytest.skip("Set VELA_FULL_COREAI_PATH to a complete bundle with validation_cases.json")
    root = Path(location).resolve()
    cases = json.loads((root / "validation_cases.json").read_text())
    assert len(cases) >= 2
    sdk = ModelSdk()
    register_vela(sdk.registry)
    async with await sdk.acquire(
        "vllm-sr/vela", variant="3_1-fast-full", runtime="coreai", options={"artifact_path": root}
    ) as handle:
        capability = handle.require(StructuredPrediction)
        for case in cases:
            response = await capability.predict_structure(
                StructuredPredictionRequest(
                    state=case["state"],
                    questions=case["questions"],
                )
            )
            assert_response(response.result, case["expected"])
