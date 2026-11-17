"""Matching decisions must not hide changed public confidence or score values."""

import pytest
from hugging_mac_conversion.models.d3.evaluation.sdk_validation import assert_response_close


def test_confidence_regression_is_not_accepted_as_matching_choice() -> None:
    expected = {"answer": {"choice": "yes", "confidence": 0.715477}}
    actual = {"answer": {"choice": "yes", "confidence": 0.756597}}
    with pytest.raises(AssertionError):
        assert_response_close(actual, expected)


def test_response_schema_is_part_of_parity() -> None:
    with pytest.raises(AssertionError):
        assert_response_close(
            {"answer": {"choice": "yes"}}, {"answer": {"choice": "yes", "score": 1.0}}
        )


def test_configured_validator_resolves_in_conversion_package(monkeypatch, tmp_path) -> None:
    import importlib
    from unittest.mock import AsyncMock

    from hugging_mac_conversion.registry import get_model

    target = get_model("d3").config.sdk_validator
    module_name, function_name = target.split(":")
    assert module_name == "hugging_mac_conversion.models.d3.evaluation.sdk_validation"
    module = importlib.import_module(module_name)
    expected = {"passed": True, "samples": 2, "cases": [], "capability_coverage": {}}
    validate = AsyncMock(return_value=expected)
    monkeypatch.setattr(module, "_validate_bundle", validate)

    assert getattr(module, function_name)(tmp_path) == expected
    validate.assert_awaited_once_with(tmp_path)


def test_video_fixtures_become_lossless_sdk_inputs(tmp_path) -> None:
    from io import BytesIO

    import numpy as np
    from hugging_mac_conversion.models.d3.evaluation.sdk_validation import load_case_request
    from hugging_mac_sdk.schemas.decision import DecisionVideoFrames
    from PIL import Image

    frames = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    np.save(tmp_path / "frames.npy", frames)
    request = load_case_request(
        tmp_path,
        {
            "state": "video",
            "questions": {},
            "videos": [{"frames_npy": "frames.npy"}],
        },
    )
    video = request.videos[0]
    assert isinstance(video, DecisionVideoFrames)
    for actual, expected in zip(video.frames, frames, strict=True):
        with Image.open(BytesIO(actual.data)) as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), expected)
