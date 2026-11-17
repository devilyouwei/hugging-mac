"""Dynamic Vela decision-backbone graph; adapters are outside this contract."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .types import CoreAILayout, ExportGraph
from .upstream import load_upstream


def prepare_decision(root: Path, layout: CoreAILayout) -> ExportGraph:
    torch = importlib.import_module("torch")
    width = layout.export_sequence_length
    model = load_upstream(root).DecisionEncoder.load(root, device="cpu")

    class VelaGraph(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(
            self, input_ids: Any, attention_mask: Any, option_positions: Any, pool_mask: Any
        ) -> Any:
            m = self.model
            h = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = (h * pool_mask.unsqueeze(-1)).sum(1) / pool_mask.sum(1, keepdim=True)
            v = torch.nn.functional.normalize(m.w_p(m.norm(pooled)), dim=-1)
            options = torch.gather(
                h, 1, option_positions.long().unsqueeze(-1).expand(-1, -1, h.shape[-1])
            )
            q = m.w_q(m.norm(h[:, 1])).unsqueeze(1)
            u = torch.nn.functional.normalize(m.w_o(m.norm(options)) + q, dim=-1)
            return (u * v.unsqueeze(1)).sum(-1) / m.log_tau.exp() + m.cls_mlp(options).squeeze(-1)

    record = model.encode(
        "Please help me write Python code.",
        {
            "type": "choice",
            "instructions": "What is the topic?",
            "criteria": {"coding": "Software development", "cooking": "Food recipes"},
        },
    )
    args = inputs_for_record(model, record, width)
    ids, mask, _, _ = args
    graph = VelaGraph().eval()
    with torch.no_grad():
        h = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = model._logits(h, [record])[0].unsqueeze(0)
        torch.testing.assert_close(graph(*args), expected)
    sequence = torch.export.Dim("sequence", min=2, max=layout.max_sequence_length)
    options = torch.export.Dim("options", min=2, max=layout.max_sequence_length)
    shapes = ({1: sequence}, {1: sequence}, {1: options}, {1: sequence})
    return ExportGraph(
        graph,
        args,
        ("input_ids", "attention_mask", "option_positions", "pool_mask"),
        "logits",
        shapes,
    )


def inputs_for_record(model: Any, record: dict[str, Any], width: int) -> tuple[Any, ...]:
    torch = importlib.import_module("torch")
    n = len(record["ids"])
    if n > width:
        raise ValueError("Example exceeds configured sequence length")
    ids = torch.full((1, width), model.tok.pad_token_id, dtype=torch.int32)
    ids[0, :n] = torch.tensor(record["ids"], dtype=torch.int32)
    mask = torch.zeros_like(ids)
    mask[:, :n] = 1
    pool = torch.zeros((1, width))
    pool[:, record["pool"][0] : record["pool"][1]] = 1
    return (ids, mask, torch.tensor([record["opt_pos"]], dtype=torch.int32), pool)
