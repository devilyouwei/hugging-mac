"""Explicit publisher-oracle parity for the downloaded Core AI artifact."""

import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hugging_mac_sdk.models.laya.coreai import CoreAILayaEngine
from hugging_mac_sdk.models.laya.utils.execution import run_blocking


@pytest.mark.smoke
async def test_publisher_reference_rows() -> None:
    location = os.environ.get("LAYA_COREAI_PATH")
    if not location:
        pytest.skip("Set LAYA_COREAI_PATH to a published macos/wfp16-s256 or s512 directory")
    root = Path(location)
    rows = json.loads((root / "reference.json").read_text())["rows"]
    engine = CoreAILayaEngine(
        None, SimpleNamespace(view=lambda: nullcontext(root)), model_name="test"
    )
    try:
        await engine.load(root)
        for row in rows:
            p, act = await run_blocking(
                engine.run_row,
                row["sequence_ids"],
                row["marker_positions"],
                row["question"]["type"],
            )
            np.testing.assert_allclose(p, row["probabilities"], atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(act, row["act_probability"], atol=1e-4, rtol=1e-4)
            ids, markers, _ = engine._packer.pack("", row["question"])
            assert markers == row["marker_positions"]
            assert ids[:-1] == row["sequence_ids"][: len(ids) - 1]
        with pytest.raises(ValueError, match="no truncation"):
            engine._packer.pack(" hello" * 2000, rows[0]["question"])
    finally:
        await engine.close()


@pytest.mark.smoke
async def test_public_sdk_with_downloaded_artifact() -> None:
    from hugging_mac_sdk import ModelSdk
    from hugging_mac_sdk.capabilities import DecisionMaking
    from hugging_mac_sdk.models.laya import register_laya
    from hugging_mac_sdk.schemas.decision import DecisionRequest

    location = os.environ.get("LAYA_COREAI_PATH")
    if not location:
        pytest.skip("Set LAYA_COREAI_PATH to a published variant directory")
    root = Path(location).resolve()
    window = json.loads((root / "metadata.json").read_text())["decision"]["window"]
    sdk = ModelSdk()
    register_laya(sdk.registry)
    async with await sdk.acquire(
        "convaiinnovations/laya",
        variant=f"multilingual-legacy-s{window}",
        runtime="coreai",
        options={"artifact_path": root},
    ) as handle:
        response = await handle.require(DecisionMaking).decide(
            DecisionRequest(
                state={"message": "请帮我写一段 Python 程序", "language": "中文"},
                questions={
                    "topic": {
                        "type": "choice",
                        "instructions": "主题",
                        "criteria": {"code": "编程", "food": "食物"},
                    },
                    "coding": {"type": "noul", "instructions": "是否讨论编程?"},
                    "score": {
                        "type": "score",
                        "instructions": "编程相关程度",
                        "criteria": ["无关", "相关"],
                    },
                },
            )
        )
        assert {answer.type for answer in response.answers.values()} == {"choice", "score", "noul"}
        assert all(answer.action is not None for answer in response.answers.values())
