"""D3 text graph with explicit noncausal attention and position inputs.

Adapted from hugging-mac/d3-edge-CoreAI's Apache-2.0 conversion recipe;
Full multimodal task coverage remains required for release.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .types import ExportGraph
from .upstream import load_upstream


def inputs_for_tokens(tokens: list[int], pad_id: int, width: int) -> tuple[Any, ...]:
    torch = importlib.import_module("torch")
    if not tokens or len(tokens) > width:
        raise ValueError(f"Prompt has {len(tokens)} tokens; capacity is {width}; no truncation")
    count = len(tokens)
    ids = torch.full((1, width), pad_id, dtype=torch.int32)
    ids[0, -count:] = torch.tensor(tokens, dtype=torch.int32)
    mask = torch.zeros_like(ids)
    mask[0, -count:] = 1
    positions = torch.zeros((4, 1, width), dtype=torch.int32)
    positions[:, :, -count:] = torch.arange(count, dtype=torch.int32)
    return ids, mask, positions


def prepare_decision(root: Path, sequence_length: int, max_sequence_length: int) -> ExportGraph:
    torch = importlib.import_module("torch")
    upstream = load_upstream(root)
    runtime = upstream.D3(root, device="cpu")
    from .coreai_attention import dynamic_language_model

    if runtime.attention_mode != "noncausal_full_attention":
        raise ValueError("D3 text adapter requires the declared noncausal full attention mode")

    class D3Graph(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.source_runtime = runtime
            self.language_model = dynamic_language_model(
                runtime.backbone.language_model, compute_dtype=torch.float32
            )
            self.register_buffer("readout", runtime.readout.float())

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            position_ids: Any,
            visual_embeddings: Any,
            visual_mask: Any,
        ) -> Any:
            embeddings = self.language_model.embed_tokens(input_ids)
            embeddings = torch.where(
                visual_mask, visual_embeddings.to(embeddings.dtype), embeddings
            )
            hidden = self.language_model(
                inputs_embeds=embeddings,
                attention_mask={
                    "full_attention": attention_mask[:, None, None, :].bool(),
                    "linear_attention": attention_mask,
                },
                position_ids=position_ids,
                use_cache=False,
            ).last_hidden_state
            return hidden[:, -1].float() @ self.readout.T

    # Full-length inputs avoid specializing to the padding pattern of one prompt.
    args = inputs_for_tokens([1] * sequence_length, runtime.pad_id, sequence_length)
    hidden = runtime.backbone.config.text_config.hidden_size
    args += (
        torch.zeros(1, sequence_length, hidden),
        torch.zeros(1, sequence_length, 1, dtype=torch.bool),
    )
    sequence = torch.export.Dim("sequence", min=2, max=max_sequence_length)
    shapes = ({1: sequence}, {1: sequence}, {2: sequence}, {1: sequence}, {1: sequence})
    return ExportGraph(
        D3Graph().eval(),
        args,
        ("input_ids", "attention_mask", "position_ids", "visual_embeddings", "visual_mask"),
        "logits",
        shapes,
    )
