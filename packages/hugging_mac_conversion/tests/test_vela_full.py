"""Full Vela contract and dynamic external-weight execution regressions."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hugging_mac_conversion.models.vela.evaluation.comparison import compare
from hugging_mac_conversion.models.vela.utils.components import (
    INPUTS,
    OUTPUTS,
    ComponentGraph,
    WeightInputsGraph,
    remove_unused_parameters,
    sample_inputs,
)
from hugging_mac_conversion.registry import get_model


def test_full_variant_does_not_filter_adapters() -> None:
    variant = get_model("vela").config.variant("3_1-fast")
    assert variant.source == "vllm-sr/vela3.1-fast"
    assert variant.probe.ignore_patterns == ()
    assert variant.probe.dynamic_sequence_length == 32768


def test_span_offsets_and_scores_are_required() -> None:
    reference = {"spans": [{"start": 2, "end": 7, "score": 0.9}]}
    with pytest.raises(AssertionError):
        compare({"spans": [{"start": 3, "end": 7, "score": 0.9}]}, reference)
    with pytest.raises(AssertionError):
        compare({"spans": [{"start": 2, "end": 7, "score": 0.5}]}, reference)


class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(
            last_hidden_state=self.embedding(input_ids) * attention_mask[..., None]
        )


class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()


def heads(net, hidden, qi, oi, ui, ei):
    query = hidden[qi[:, 0], qi[:, 1]]
    scores = (hidden[oi[:, 0], oi[:, 1]] * query[oi[:, 2]]).sum(-1)
    spans = hidden[ui[:, 0], ui[:, 1]] @ hidden[ei[:, 0], ei[:, 1]].T
    return scores, spans


@pytest.mark.smoke
def test_immutable_weights_and_all_dynamic_axes(tmp_path: Path) -> None:
    from hugging_mac_conversion.shared.coreai_export import convert_pytorch_to_coreai
    from hugging_mac_conversion.shared.coreai_runtime import CoreAIProvider

    graph = ComponentGraph(Net().eval(), heads).eval()
    sample = sample_inputs()
    batch, sequence = torch.export.Dim("batch"), torch.export.Dim("sequence")
    shapes = (
        {0: batch, 1: sequence},
        {0: batch, 1: sequence},
        *({0: torch.export.Dim(name)} for name in ("queries", "options", "units", "labels")),
    )
    import coreai_torch

    weights = tuple(value.detach() for value in graph.parameters())
    exported = torch.export.export(
        WeightInputsGraph(graph), (*weights, *sample), dynamic_shapes=(({}, *shapes),)
    ).run_decompositions(coreai_torch.get_decomp_table())
    remove_unused_parameters(exported)
    assert not exported.state_dict
    asset = convert_pytorch_to_coreai(
        exported,
        tmp_path / "model.aimodel",
        input_names=("weight", *INPUTS),
        output_names=OUTPUTS,
        optimize=False,
    )

    async def run():
        session = await CoreAIProvider().create_session(asset, device="cpu", options={})
        try:
            for delta in (0.0, 0.25):
                graph.net.encoder.embedding.weight.data.add_(delta)
                session.set_constant_inputs(
                    {"weight": graph.net.encoder.embedding.weight.detach().numpy()}
                )
                for count, length in ((1, 8), (2, 19)):
                    args = (
                        torch.ones((count, length), dtype=torch.int32),
                        torch.ones((count, length), dtype=torch.int32),
                        torch.tensor([[0, 1, 2, length - 1]], dtype=torch.int32),
                        torch.tensor([[0, 3, 0]], dtype=torch.int32),
                        torch.tensor([[0, 4], [0, 5]], dtype=torch.int32),
                        torch.tensor([[0, 2]], dtype=torch.int32),
                    )
                    expected = graph(*args)
                    actual = session.run(
                        dict(zip(INPUTS, [value.numpy() for value in args], strict=True))
                    )
                    for key, value in zip(OUTPUTS, expected, strict=True):
                        np.testing.assert_allclose(
                            actual[key], value.detach().numpy(), atol=1e-5, rtol=1e-5
                        )
        finally:
            await session.close()

    asyncio.run(run())
