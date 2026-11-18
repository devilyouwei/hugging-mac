"""Complete Vela graphs with original FP32 weights supplied as immutable Core AI inputs.

Separating immutable weight storage from the graph avoids duplicating the shared
backbone in every adapter asset. LoRA stays as its original two linear operations;
no merging, quantization, head removal, or alternate neural runtime is involved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from hugging_mac_conversion.models.vela.utils.full_upstream import load_full_upstream
from hugging_mac_conversion.shared.coreai_export import convert_pytorch_to_coreai

INPUTS = ("input_ids", "attention_mask", "q_index", "opt_index", "unit_index", "ent_index")
OUTPUTS = ("logits", "span_logits")


def architecture_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "configuration_ultrabert.py",
        "modeling_ultrabert.py",
        "vela3_encoder.py",
        "vela3_schema.py",
    ):
        digest.update(name.encode())
        digest.update((source / name).read_bytes())
    return digest.hexdigest()


def graph_kind(header: dict[str, Any] | None) -> str:
    if header is None:
        return "decision"
    meta = header["__metadata__"]
    span = "span" if "w_t.weight" in header else "router"
    rank, alpha, layers = (
        meta["lora_rank"],
        float(meta["lora_alpha"]),
        meta.get("lora_layers", "all"),
    )
    return f"{span}-r{rank}-a{alpha:g}-{layers}"


class ComponentGraph(torch.nn.Module):
    def __init__(self, net: Any, heads: Any) -> None:
        super().__init__()
        self.net = net
        self.heads = heads

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        q_index: Any,
        opt_index: Any,
        unit_index: Any,
        ent_index: Any,
    ) -> Any:
        hidden = self.net.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.heads(self.net, hidden, q_index, opt_index, unit_index, ent_index)


class WeightInputsGraph(torch.nn.Module):
    def __init__(self, graph: ComponentGraph) -> None:
        super().__init__()
        self.neural = graph
        self.weight_names = tuple(dict(graph.named_parameters()))

    def forward(self, *args: Any) -> Any:
        count = len(self.weight_names)
        weights = dict(zip(self.weight_names, args[:count], strict=True))
        return torch.func.functional_call(self.neural, weights, args[count:])


def remove_unused_parameters(exported: Any) -> None:
    """functional_call leaves unused shadow parameters; do not serialize them."""
    from torch.export.graph_signature import InputKind

    nodes = {node.name: node for node in exported.graph_module.graph.nodes}
    retained = []
    for spec in exported.graph_signature.input_specs:
        node = nodes[spec.arg.name]
        if spec.kind == InputKind.PARAMETER and not node.users:
            exported.graph_module.graph.erase_node(node)
            exported.state_dict.pop(spec.target)
        else:
            retained.append(spec)
    exported.graph_signature.input_specs[:] = retained
    exported.graph_module.recompile()
    exported.validate()


def sample_inputs(length: int = 32) -> tuple[Any, ...]:
    return (
        torch.ones((2, length), dtype=torch.int32),
        torch.ones((2, length), dtype=torch.int32),
        torch.tensor([[0, 1, 10, length - 1], [1, 1, 10, length - 1]], dtype=torch.int32),
        torch.tensor([[0, 3, 0], [0, 5, 0], [1, 3, 1]], dtype=torch.int32),
        torch.tensor([[0, 10], [0, 11], [1, 10], [1, 11]], dtype=torch.int32),
        torch.tensor([[0, 7], [1, 7]], dtype=torch.int32),
    )


def export_component(source: Path, header: dict[str, Any] | None, output: Path) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModel

    upstream = load_full_upstream(source)
    cfg = AutoConfig.from_pretrained(source, trust_remote_code=True)
    if header is None:
        net = upstream.DecisionEncoder(cfg, cfg.decision["proj_dim"])
        net.span = False
    else:
        encoder = AutoModel.from_config(  # type: ignore[no-untyped-call]
            cfg, trust_remote_code=True, attn_implementation="sdpa", dtype=torch.float32
        )
        net = upstream.schema.SchemaNet(
            encoder, cfg.hidden_size, cfg.decision["proj_dim"], "w_t.weight" in header
        )
        meta = header["__metadata__"]
        upstream.schema.inject_lora(
            net, int(meta["lora_rank"]), float(meta["lora_alpha"]), meta.get("lora_layers", "all")
        )
    neural = ComponentGraph(net.eval(), upstream.schema.SchemaNet.heads).eval()
    graph = WeightInputsGraph(neural).eval()
    parameters = tuple(value.detach() for value in neural.parameters())
    sequence = torch.export.Dim("sequence", min=2, max=32768)
    batch = torch.export.Dim("batch", min=1)
    dynamic = (
        {0: batch, 1: sequence},
        {0: batch, 1: sequence},
        *(
            {0: torch.export.Dim(name, min=1)}
            for name in ("questions", "options", "units", "labels")
        ),
    )
    shapes = (*({} for _ in parameters), *dynamic)
    exported = torch.export.export(graph, (*parameters, *sample_inputs()), dynamic_shapes=(shapes,))
    import coreai_torch

    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    remove_unused_parameters(exported)
    if exported.graph_signature.buffers_to_mutate:
        raise ValueError("Immutable Vela inference must not mutate weights or buffers")
    states = {
        f"weight_{i:04d}": {"name": name.removeprefix("net."), "shape": list(value.shape)}
        for i, (name, value) in enumerate(zip(graph.weight_names, parameters, strict=True))
    }
    output.mkdir(parents=True, exist_ok=False)
    convert_pytorch_to_coreai(
        exported,
        output / "model.aimodel",
        input_names=(*states, *INPUTS),
        output_names=OUTPUTS,
        optimize=False,
    )
    report = {
        "kind": graph_kind(header),
        "weight_binding": "inputs",
        "architecture_sha256": architecture_digest(source),
        "inputs": INPUTS,
        "outputs": OUTPUTS,
        "states": states,
        "sequence_bound": 32768,
        "dynamic_axes": ["batch", "sequence", "questions", "options", "units", "labels"],
        "precision": "float32",
        "lora_merged": False,
    }
    with (output / "model.aimodel/main.mlirb").open("rb") as stream:
        report["graph_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "graph.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
