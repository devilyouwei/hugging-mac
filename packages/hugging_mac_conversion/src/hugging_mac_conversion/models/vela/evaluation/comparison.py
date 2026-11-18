"""Compare complete upstream responses, including offsets, scores and usage."""

from typing import Any

import numpy as np


def compare(actual: Any, expected: Any, path: str = "response") -> float:
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: different fields")
        return max(
            (compare(actual[k], value, f"{path}.{k}") for k, value in expected.items()), default=0.0
        )
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: different item/span count")
        return max(
            (
                compare(a, b, f"{path}[{i}]")
                for i, (a, b) in enumerate(zip(actual, expected, strict=True))
            ),
            default=0.0,
        )
    if isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.02, err_msg=path)
        return abs(float(actual) - expected)
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")
    return 0.0
