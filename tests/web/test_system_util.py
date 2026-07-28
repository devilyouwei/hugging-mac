from __future__ import annotations

from hugging_mac_web.shared.utils.system_util import (
    _neural_engine_cores,
    _parse_processor_layout,
)


def test_parse_apple_processor_layout() -> None:
    assert _parse_processor_layout("proc 10:4:6:0") == (4, 6)
    assert _parse_processor_layout("unknown") == (None, None)
    assert _parse_processor_layout(None) == (None, None)


def test_known_apple_neural_engine_core_counts() -> None:
    assert _neural_engine_cores("Apple M5") == 16
    assert _neural_engine_cores("Apple M2 Ultra") == 32
    assert _neural_engine_cores("Apple A18 Pro") == 16
    assert _neural_engine_cores("Intel Core i9") is None
