# Qwen3-TTS

## Summary

Canonical SDK package for `qwen/qwen3-tts-12hz`. Model scale, Base edition, MLX/Core ML, and quantization are represented as variant/runtime/artifact data.

## Runtimes and variants

- Variant: `0.6b-base`
- Runtimes: `mlx` and `coreml`
- Artifacts: pinned MLX 4-bit snapshot and FluidInference's pinned six-graph W8A16 Core ML pipeline

The Core ML runtime uses `TextProjector`, two code embedders, the cached talker/code-predictor decoders, and the fixed 125-frame `SpeechDecoder`. It supports English and Chinese, the bundled official speaker, normal speed, and at most 125 codec frames (about 10 seconds). Reference-audio cloning remains available through MLX; the Core ML artifact does not ship a speaker encoder and rejects that option explicitly. FluidAudio currently classifies this conversion as evaluated but not officially supported because generation is slow.

## Capability API

- `SpeechSynthesis.synthesize(SpeechSynthesisRequest) -> SpeechSynthesisResponse`

Reference audio and its transcript may be supplied through the shared request schema.

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: canonical metadata and MLX artifact declaration.
- `config.py`: typed MLX/resource options.
- `definition.py`: MLX factory and registration.
- `instance.py`: lifecycle and runtime-neutral synthesis capability.
- `mlx.py`: MLX generation engine.
- `coreml.py`: six-graph Core ML autoregressive generation engine.
- `resources.py`: snapshot validation and lifecycle.
- `utils/`: private engine output types.
