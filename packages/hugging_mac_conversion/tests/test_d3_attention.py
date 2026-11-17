"""Regression for the exact upstream recurrence and dynamic Core AI lowering."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hugging_mac_sdk.models.d3.utils.coreai_attention import recurrent_delta
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule


@pytest.mark.parametrize("length", [3, 64, 67])
def test_recurrence_preserves_upstream_math(length: int) -> None:
    torch.manual_seed(length)
    q, k, v = (torch.randn(1, length, 2, 8) for _ in range(3))
    g, beta = -torch.rand(1, length, 2), torch.rand(1, length, 2)
    expected, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(recurrent_delta(q, k, v, g, beta), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.smoke
def test_native_dynamic_recurrence(tmp_path: Path) -> None:
    import asyncio

    from hugging_mac_conversion.shared.coreai import predict
    from hugging_mac_sdk.converters.coreai import convert_pytorch_to_coreai

    class Graph(torch.nn.Module):
        def forward(self, q, k, v, g, beta):
            return recurrent_delta(q, k, v, g, beta)

    def sample(length):
        return (
            *[torch.randn(1, length, 2, 4) for _ in range(3)],
            -torch.rand(1, length, 2),
            torch.rand(1, length, 2),
        )

    names = ("q", "k", "v", "g", "beta")
    sequence = torch.export.Dim("sequence", min=2)
    asset = convert_pytorch_to_coreai(
        Graph().eval(),
        tmp_path / "delta.aimodel",
        example_inputs=sample(8),
        dynamic_shapes=tuple({1: sequence} for _ in names),
        input_names=names,
        output_names=["out"],
        optimize=False,
    )
    for length in (5, 8, 17):
        args = sample(length)
        actual = asyncio.run(
            predict(asset, dict(zip(names, [a.numpy() for a in args], strict=True)), "cpu")
        )["out"]
        expected, _ = torch_chunk_gated_delta_rule(*args, use_qk_l2norm_in_kernel=True)
        np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=2e-5)
