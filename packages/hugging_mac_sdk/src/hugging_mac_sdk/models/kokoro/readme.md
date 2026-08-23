# Kokoro

## Summary

Canonical SDK package for `hexgrad/kokoro`, a compact multilingual text-to-speech model. Model size and compiled formats remain variant/artifact metadata.

The downloadable Core ML runtime uses FluidInference's Float32 `kokoro_21_5s.mlmodelc` artifact from the repository's `main` branch. It accepts 124 input IDs and exposes a 175,800-sample output buffer. The SDK keeps each pass below a 6.8-second safety threshold, recursively retrying overlong predictions at punctuation or whitespace boundaries, then trusts the model-reported sample length and applies only a 5 ms fade. The graph does not expose speed control.

## Runtimes and variants

- Variant: `v1.0`
- Runtimes: `pytorch-mps`, `coreml`

## Capability API

- `SpeechSynthesis.synthesize(SpeechSynthesisRequest) -> SpeechSynthesisResponse`

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: canonical metadata and PyTorch/Core ML artifacts.
- `config.py`: typed runtime, resource, and voice options.
- `definition.py`: runtime factories and registration.
- `instance.py`: shared lifecycle and synthesis capability.
- `torch.py`: PyTorch/MPS engine.
- `coreml.py`: end-to-end Core ML engine.
- `resources.py`: source and prebuilt artifact lifecycle.
- `utils/`: private output types and upstream notices.
