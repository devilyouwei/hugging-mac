import pytest
from hugging_mac_conversion.models.d3.utils.coreai_export import inputs_for_tokens


def test_d3_packing_keeps_real_positions_and_padding_separate():
    import torch

    ids, mask, positions = inputs_for_tokens([17, 23, 29], 0, 512)
    assert ids.dtype == mask.dtype == positions.dtype == torch.int32
    assert ids.shape == mask.shape == (1, 512)
    assert positions.shape == (4, 1, 512)
    assert ids[0, -3:].tolist() == [17, 23, 29]
    assert mask.sum().item() == 3
    assert not mask[0, :-3].any()
    for axis in range(4):
        assert positions[axis, 0, -3:].tolist() == [0, 1, 2]
        assert not positions[axis, 0, :-3].any()


def test_d3_packing_accepts_exact_capacity_and_rejects_overflow():
    ids, mask, positions = inputs_for_tokens([1] * 512, 0, 512)
    assert ids.shape == (1, 512)
    assert mask.all()
    assert positions[0, 0, -1].item() == 511
    for tokens in ([], [1] * 513):
        with pytest.raises(ValueError, match="no truncation"):
            inputs_for_tokens(tokens, 0, 512)
