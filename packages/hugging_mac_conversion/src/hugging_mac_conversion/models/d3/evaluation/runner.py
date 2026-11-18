"""D3-specific prompt rendering, left padding, answer codes and temperature."""

from __future__ import annotations

import importlib
from pathlib import Path
from time import perf_counter

from hugging_mac_conversion.models.d3.utils.coreai_export import inputs_for_tokens

from ....shared.config import VariantConfig
from ....shared.evaluation.contracts import EvaluationCase, EvaluationDefinition, EvaluationSample
from ..converter import prepare


def prepare_samples(
    config: VariantConfig, source: Path, cases: tuple[EvaluationCase, ...]
) -> list[EvaluationSample]:
    torch = importlib.import_module("torch")
    graph = prepare(config, source)
    runtime = graph.module.source_runtime
    samples = []
    width = config.probe.sequence_length
    for case in cases:
        try:
            if set(case.payload) - {"state", "question"}:
                raise ValueError(
                    "Current diagnostic artifact supports text only; multimodal is untested"
                )
            prepared = runtime.prepare(case.payload["state"], {case.id: case.payload["question"]})
            if prepared.errors:
                raise ValueError(str(prepared.errors))
            tokens = prepared.sequences[case.id]
            labels = tuple(prepared.questions[case.id].keys)
            current_width = len(tokens) if config.probe.dynamic_sequence_length else width
            args = inputs_for_tokens(tokens, runtime.pad_id, current_width)
            args += (
                torch.zeros(1, current_width, runtime.backbone.config.text_config.hidden_size),
                torch.zeros(1, current_width, 1, dtype=torch.bool),
            )
            started = perf_counter()
            with torch.no_grad():
                reference = runtime.probabilities([tokens], [len(labels)])[0]
            duration = (perf_counter() - started) * 1000
            samples.append(
                EvaluationSample(
                    case,
                    {name: arg.numpy() for name, arg in zip(graph.input_names, args, strict=True)},
                    reference,
                    labels,
                    graph.output_name,
                    temperature=runtime.temperature,
                    reference_ms=duration,
                )
            )
        except (ValueError, KeyError) as error:
            samples.append(
                EvaluationSample(case, {}, None, (), graph.output_name, error=str(error))
            )
    return samples


EVALUATION = EvaluationDefinition(Path(__file__).with_name("config.yaml"), prepare_samples)
