"""Explicit native coverage against a locally supplied original D3 checkpoint."""

from __future__ import annotations

import asyncio
import os
from io import BytesIO
from pathlib import Path

import pytest
from hugging_mac_sdk import ModelSdk, ReusePolicy
from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.models.d3 import D3_DEFINITION, register_d3
from hugging_mac_sdk.schemas.decision import DecisionRequest, DecisionVideoFrames
from hugging_mac_sdk.schemas.detection import ImageInput
from PIL import Image


@pytest.mark.smoke
@pytest.mark.parametrize("managed", [False, True])
def test_original_pytorch_text_image_video_through_handle(tmp_path: Path, managed: bool) -> None:
    asyncio.run(_run_original_pytorch(tmp_path, managed))


async def _run_original_pytorch(tmp_path: Path, managed: bool) -> None:
    value = os.environ.get("D3_SOURCE_PATH")
    if value is None:
        pytest.skip("Set D3_SOURCE_PATH to a complete original checkpoint")
    options = {"source_path": Path(value)}
    if managed:
        for artifact in D3_DEFINITION.artifacts:
            if artifact.shared or artifact.runtime == "pytorch":
                root = artifact.resolve(tmp_path)
                root.mkdir(parents=True)
                for name in artifact.required_files:
                    (root / name).symlink_to(Path(value) / name)
        options = {"model_home": tmp_path}
    with BytesIO() as buffer:
        Image.new("RGB", (256, 256), (255, 0, 0)).save(buffer, format="PNG")
        frame = ImageInput(data=buffer.getvalue())
    sdk = ModelSdk()
    register_d3(sdk.registry)
    async with await sdk.acquire(
        "vllm-sr/d3",
        runtime="pytorch",
        reuse=ReusePolicy.DEDICATED,
        options=options,
    ) as handle:
        decisions = handle.require(DecisionMaking)
        for media in (
            {},
            {"images": (frame,)},
            {"videos": (DecisionVideoFrames(frames=(frame,) * 4),)},
        ):
            response = await decisions.decide(
                DecisionRequest(
                    state="A customer was charged twice.",
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
                        "payment": {"type": "noul", "instructions": "Is this about a payment?"},
                        "severity": {
                            "type": "score",
                            "criteria": [{"level": "low"}, {"level": "high"}],
                        },
                    },
                    **media,
                )
            )
            assert response.runtime == "pytorch"
            assert response.device == "cpu"
            answer = response.answers["route"]
            assert answer.type == "choice", answer
            assert answer.choice in {"billing", "shipping", "other"}
            assert response.answers["payment"].type == "noul"
            assert response.answers["severity"].type == "score"
            assert response.usage.input_tokens > 0
