"""Verify native graph outputs and detections against the pinned FP32 ONNX release."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from .converter import reference_sessions
from .publication.spotter import Array, CoreML, read_keywords, read_pcm, spot

COMPUTE_UNITS = ("CPU_ONLY", "ALL")
EXPECTED = {
    "en_0.wav": ["LIGHT_UP"],
    "en_1.wav": ["LOVELY_CHILD"],
    "zh_0.wav": [],
    "zh_1.wav": [],
    "zh_2.wav": [],
    "zh_3.wav": ["文森特卡索"],
    "zh_4.wav": [],
    "zh_5.wav": ["周望军"],
    "zh_6.wav": [],
    "silence": [],
}


class ONNX:
    def __init__(self, reference: Path) -> None:
        self.sessions = reference_sessions(reference)
        self.metadata = {part: session.get_inputs() for part, session in self.sessions.items()}

    def __call__(self, part: str, values: tuple[Array, ...]) -> tuple[Array, ...]:
        inputs = {
            info.name: np.asarray(value, dtype=np.int64 if info.type == "tensor(int64)" else None)
            for info, value in zip(self.metadata[part], values, strict=True)
        }
        return tuple(self.sessions[part].run(None, inputs))


def check_tensors(
    native: CoreML, oracle: ONNX, graphs: dict[str, Any], count: int = 32
) -> dict[str, Any]:
    rng = np.random.default_rng(721)
    report = {}
    for part, contract in graphs.items():
        timing = []
        errors = dict.fromkeys(contract["outputs"], 0.0)
        inputs = contract["inputs"]
        caches = [np.zeros(meta["shape"], dtype=meta["dtype"]) for meta in inputs[1:]]
        for index in range(count + 2):
            if part == "encoder":
                values = (rng.normal(size=inputs[0]["shape"]).astype(np.float32), *caches)
            elif part == "decoder":
                values = (rng.integers(-1, 263, size=inputs[0]["shape"], dtype=np.int32),)
            else:
                values = tuple(rng.normal(size=meta["shape"]).astype(np.float32) for meta in inputs)
            expected = oracle(part, values)
            start = perf_counter_ns()
            actual = native(part, values)
            if index >= 2:
                timing.append((perf_counter_ns() - start) / 1_000_000)
            for name, got, want in zip(contract["outputs"], actual, expected, strict=True):
                if got.shape != want.shape or not np.isfinite(got).all():
                    raise ValueError(f"Invalid {part}/{name} output: {got.shape}")
                np.testing.assert_allclose(
                    got, want, atol=1e-4, rtol=1e-4, err_msg=f"{part}/{name}"
                )
                errors[name] = max(errors[name], float(np.max(np.abs(got - want))))
            if part == "encoder":
                # Compare successive updates using the native state carried by the stream.
                caches = list(actual[1:])
        report[part] = {
            "max_abs_error": max(errors.values()),
            "max_abs_error_by_output": errors,
            "invocations": count,
            "inference_p95_ms": float(np.percentile(timing, 95)),
        }
    return report


def validate(bundle: Path, reference: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    keywords = read_keywords(bundle / "keywords.txt", bundle / "tokens.txt")
    oracle = ONNX(reference)
    audio = {
        name: read_pcm(reference / "test_wavs" / name)
        if name != "silence"
        else np.zeros(16000, dtype=np.float32)
        for name in EXPECTED
    }
    baseline = {name: spot(oracle, manifest, keywords, pcm) for name, pcm in audio.items()}
    for name, hits in baseline.items():
        if [hit.label for hit in hits] != EXPECTED[name]:
            raise ValueError(f"Reference keyword outcome changed on {name}: {hits}")

    backends = {}
    for compute_unit in COMPUTE_UNITS:
        native = CoreML(bundle, manifest["graphs"], compute_unit)
        tensors = check_tensors(native, oracle, manifest["graphs"])
        for name, pcm in audio.items():
            hits = spot(native, manifest, keywords, pcm)
            if hits != baseline[name]:
                raise ValueError(
                    f"{compute_unit} keyword parity failed on {name}: {hits} != {baseline[name]}"
                )
        backends[compute_unit] = {"tensors": tensors, "audio_cases_passed": len(audio)}
    return {
        "status": "passed",
        "audio_cases": {name: [asdict(hit) for hit in hits] for name, hits in baseline.items()},
        "backends": backends,
        "scope": "Bilingual keyword and FP32 graph parity; P95 measures individual predict calls.",
    }
