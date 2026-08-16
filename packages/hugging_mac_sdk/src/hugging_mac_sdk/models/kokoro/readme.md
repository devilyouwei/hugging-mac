# Kokoro

## Summary

Canonical SDK package for `hexgrad/kokoro`, a compact multilingual text-to-speech model. Model size and compiled formats remain variant/artifact metadata.

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

