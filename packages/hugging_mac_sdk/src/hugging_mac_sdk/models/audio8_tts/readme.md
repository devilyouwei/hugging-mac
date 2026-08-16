# Audio8 TTS

## Summary

Canonical SDK package for `audio8/audio8-tts-preview`. PyTorch FP32 and MLX BF16 builds are artifacts of the same model, not separate packages or model IDs.

## Runtimes and variants

- Variant: `0.6b-preview`
- Runtimes: `pytorch`, `mlx`
- Runtime-specific source snapshots and precision are represented by resources and artifacts in `model.yaml`.

## Capability API

- `SpeechSynthesis.synthesize(SpeechSynthesisRequest) -> SpeechSynthesisResponse`

The request may include reference audio/text or a managed voice profile when supported by the selected runtime.

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: canonical identity and both runtime artifacts.
- `config.py`: shared and runtime-specific typed options.
- `definition.py`: PyTorch/MLX factories and unified registration.
- `instance.py`: shared lifecycle and `SpeechSynthesis` implementation.
- `torch.py`: PyTorch CPU engine.
- `mlx.py`: MLX engine.
- `resources.py`: aggregate model resource provider and PyTorch resources.
- `mlx_resources.py`: MLX artifact validation and lifecycle.
- `utils/`: private output types and reference-voice storage.

