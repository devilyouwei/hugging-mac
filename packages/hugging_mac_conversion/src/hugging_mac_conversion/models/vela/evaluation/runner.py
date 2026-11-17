"""Vela-specific marker/pooling inputs and upstream decision probabilities."""

from __future__ import annotations

import importlib
from pathlib import Path
from time import perf_counter

from ....shared.config import VariantConfig
from ....shared.evaluation.contracts import EvaluationCase, EvaluationDefinition, EvaluationSample
from ..converter import inputs_for_record, prepare


def prepare_samples(
    config: VariantConfig, source: Path, cases: tuple[EvaluationCase, ...]
) -> list[EvaluationSample]:
    torch = importlib.import_module("torch")
    graph = prepare(config, source)
    model = graph.module.model
    samples = []
    for case in cases:
        try:
            question = case.payload["question"]
            if question.get("type") not in {"choice", "noul"}:
                raise ValueError("Current diagnostic artifact has no adapter for this task")
            record = model.encode(case.payload["state"], question)
            width = (
                len(record["ids"])
                if config.probe.dynamic_sequence_length is not None
                else config.probe.sequence_length
            )
            args = inputs_for_record(model, record, width)
            started = perf_counter()
            with torch.no_grad():
                reference = model.forward_recs([record])[0]
            duration = (perf_counter() - started) * 1000
            samples.append(
                EvaluationSample(
                    case,
                    {name: arg.numpy() for name, arg in zip(graph.input_names, args, strict=True)},
                    reference,
                    tuple(record["names"]),
                    graph.output_name,
                    reference_ms=duration,
                )
            )
        except (ValueError, KeyError) as error:
            samples.append(
                EvaluationSample(case, {}, None, (), graph.output_name, error=str(error))
            )
    return samples


EVALUATION = EvaluationDefinition(Path(__file__).with_name("config.yaml"), prepare_samples)
