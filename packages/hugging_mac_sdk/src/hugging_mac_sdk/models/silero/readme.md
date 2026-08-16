# Silero VAD

## Summary

Canonical SDK package for `snakers4/silero-vad`, providing stateful voice-activity detection for streaming or complete audio input.

## Runtimes and variants

- Variant: `v6.2.1`
- Runtimes: `coreml`, `onnx`

## Capability API

- `VoiceActivityDetection.detect_voice_activity(VoiceActivityRequest) -> VoiceActivityResponse`

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: canonical metadata and Core ML/ONNX artifacts.
- `config.py`: typed runtime and state options.
- `definition.py`: runtime factories and registration.
- `instance.py`: lifecycle, inference state, and VAD capability.
- `coreml.py`: Core ML engine.
- `onnx.py`: ONNX Runtime engine.
- `resources.py`: artifact validation and lifecycle.
- `utils/`: private audio, postprocessing, and state types.

