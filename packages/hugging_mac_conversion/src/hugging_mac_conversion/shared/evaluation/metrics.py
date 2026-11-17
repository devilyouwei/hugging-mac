"""Numerics and decision comparisons, independent of source model architecture."""

from __future__ import annotations

from typing import Any

from .contracts import EvaluationConfig, EvaluationSample


def compare(sample: EvaluationSample, output: Any, config: EvaluationConfig) -> dict[str, Any]:
    import numpy as np

    reference = np.asarray(sample.reference, dtype=np.float64)
    logits = np.asarray(output, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < len(sample.labels):
        raise ValueError("Converted output does not match the declared decision interface")
    scores = logits[0, : len(sample.labels)] / sample.temperature
    if reference.shape != scores.shape:
        raise ValueError(
            f"Decision shape mismatch: original {reference.shape}, converted {scores.shape}"
        )
    if not np.isfinite(reference).all():
        raise ValueError("Non-finite original-model probabilities")
    if not np.isfinite(scores).all():
        raise ValueError(
            "Non-finite converted decision logits: "
            f"{int((~np.isfinite(scores)).sum())}/{scores.size}"
        )
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    original = sample.labels[int(reference.argmax())]
    converted = sample.labels[int(probabilities.argmax())]
    numeric = bool(np.allclose(reference, probabilities, atol=config.atol, rtol=config.rtol))
    agreement = original == converted
    return {
        "passed": numeric and (agreement or not config.require_decision_agreement),
        "numeric_passed": numeric,
        "decision_agreement": agreement,
        "max_absolute_error": float(np.max(np.abs(reference - probabilities))),
        "mean_absolute_error": float(np.mean(np.abs(reference - probabilities))),
        "reference_probabilities": reference.tolist(),
        "converted_probabilities": probabilities.tolist(),
        "reference_decision": original,
        "converted_decision": converted,
        "expected": sample.case.expected,
        "reference_correct": original == sample.case.expected
        if sample.case.expected is not None
        else None,
        "converted_correct": converted == sample.case.expected
        if sample.case.expected is not None
        else None,
    }
