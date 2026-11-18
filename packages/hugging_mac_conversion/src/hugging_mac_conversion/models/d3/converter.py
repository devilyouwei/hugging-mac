"""Compose conversion-owned D3 export graphs for conversion workbench evaluation."""

from pathlib import Path

from ...shared.config import VariantConfig
from ...shared.contracts import PreparedGraph


def prepare(config: VariantConfig, root: Path) -> PreparedGraph:
    from hugging_mac_conversion.models.d3.utils.coreai_export import prepare_decision

    if config.probe.dynamic_sequence_length is None:
        raise ValueError("D3 requires a dynamic sequence bound")
    graph = prepare_decision(
        root, config.probe.sequence_length, config.probe.dynamic_sequence_length
    )
    return PreparedGraph(
        graph.module, graph.args, graph.input_names, graph.output_name, graph.dynamic_shapes
    )
