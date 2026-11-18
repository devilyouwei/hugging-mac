"""Explicit real-weight parity through the public SDK capability."""

import os
from pathlib import Path

import pytest
from hugging_mac_sdk import ModelSdk, ReusePolicy
from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.models.vela import register_vela
from hugging_mac_sdk.models.vela.definition import VELA_CONFIG
from hugging_mac_sdk.schemas.decision import DecisionRequest


@pytest.mark.smoke
async def test_managed_runtime_parity(tmp_path):
    source = os.environ.get("VELA_SOURCE_PATH")
    bundle = os.environ.get("VELA_COREAI_PATH")
    if not source or not bundle:
        pytest.skip("Set VELA_SOURCE_PATH and VELA_COREAI_PATH to real local artifacts")
    variant = os.environ.get("VELA_VARIANT", "3-fast")
    for artifact in VELA_CONFIG.get_artifacts(variant=variant, include_shared=True):
        original = Path(bundle if artifact.runtime == "coreai" else source)
        root = artifact.resolve(tmp_path)
        for name in artifact.required_files:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).symlink_to((original / name).resolve())
    sdk = ModelSdk()
    register_vela(sdk.registry)
    responses = []
    for runtime in ("pytorch", "coreai"):
        async with await sdk.acquire(
            "vllm-sr/vela",
            variant=variant,
            runtime=runtime,
            device="cpu",
            reuse=ReusePolicy.DEDICATED,
            options={"model_home": tmp_path},
        ) as handle:
            results = []
            for state in (
                "I was charged twice.",
                {"customer": "Please refund the duplicate payment."},
            ):
                response = await handle.require(DecisionMaking).decide(
                    DecisionRequest(
                        state=state,
                        questions={
                            "route": {
                                "type": "choice",
                                "instructions": "Which team?",
                                "criteria": {
                                    "billing": "Payments",
                                    "shipping": "Delivery",
                                    "other": None,
                                },
                            },
                            "refund": {
                                "type": "noul",
                                "instructions": "The customer wants a refund.",
                            },
                            "single": {"type": "choice", "criteria": {"only": None}},
                        },
                    )
                )
                assert response.device == "cpu"
                assert response.runtime == runtime
                assert response.usage.input_tokens > 0
                results.append(response)
            responses.append(results)
    for expected, actual in zip(*responses, strict=True):
        assert actual.usage == expected.usage
        assert actual.answers["route"].choice == expected.answers["route"].choice
        assert actual.answers["route"].probabilities == pytest.approx(
            expected.answers["route"].probabilities, abs=1e-4
        )
        assert actual.answers["refund"].noul == pytest.approx(
            expected.answers["refund"].noul, abs=1e-4
        )
        assert actual.answers["single"] == expected.answers["single"]
