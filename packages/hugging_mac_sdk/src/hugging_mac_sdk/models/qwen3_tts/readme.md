# Qwen3-TTS

## Summary

Canonical SDK package for `qwen/qwen3-tts-12hz`. Model scale, Base edition, MLX/Core ML, and quantization are represented as variant/runtime/artifact data.

## Runtimes and variants

- Variant: `0.6b-base`
- Runtimes: `mlx` and `coreml`
- Artifacts: pinned MLX 4-bit snapshot and aufklarer's pinned six-model W8A16 Core ML pipeline

The Core ML runtime uses `TextProjector`, two code embedders, the cached talker/code-predictor decoders, and the fixed 125-frame `SpeechDecoder`. It reuses the model package's shared tokenizer artifact; the speaker and dedicated TTS BOS/EOS/PAD embeddings come from the aufklarer snapshot. The small embedding graphs stay on CPU for stable FP32 accumulation while the decoder chain defaults to CPU + Neural Engine.

Core ML supports Chinese, English, German, Italian, Portuguese, Spanish, Japanese, Korean, French, and Russian with the bundled speaker, normal speed, and at most 125 codec frames (about 10 seconds). Reference-audio cloning remains available through MLX; the Core ML artifact has no speaker encoder and rejects reference audio explicitly. The precompiled bundle requires macOS 15 or newer.

## Capability API

- `SpeechSynthesis.synthesize(SpeechSynthesisRequest) -> SpeechSynthesisResponse`

Reference audio and its transcript may be supplied through the shared request schema.

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: canonical metadata and MLX/Core ML artifact declarations.
- `config.py`: typed runtime and resource options.
- `definition.py`: MLX/Core ML factories and registration.
- `instance.py`: lifecycle and runtime-neutral synthesis capability.
- `mlx.py`: MLX generation engine.
- `coreml.py`: six-graph Core ML autoregressive generation engine.
- `resources.py`: runtime-specific snapshot validation and lifecycle.
- `utils/`: lightweight BPE tokenizer for the shared tokenizer files and private engine output types.
