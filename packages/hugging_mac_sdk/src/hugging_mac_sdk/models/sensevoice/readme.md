# SenseVoice

## Summary

Canonical SDK package for `funaudiollm/sensevoice`. The current `small` scale is a model variant rather than part of the package or canonical model ID.

## Runtimes and variants

- Variant: `small`
- Runtimes: `pytorch-mps`, `coreml`

## Capability APIs

- `SpeechTranscription.transcribe(TranscriptionRequest) -> TranscriptionResponse`
- `SpeechUnderstanding.understand_speech(SpeechUnderstandingRequest) -> SpeechUnderstandingResponse`

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: canonical metadata, runtimes, sources, and artifacts.
- `config.py`: typed instance/resource configuration.
- `definition.py`: runtime factories, converter binding, and registration.
- `instance.py`: shared transcription and rich-understanding capabilities.
- `torch.py`: PyTorch/MPS engine.
- `coreml.py`: Core ML engine.
- `resources.py`: source and converted-artifact lifecycle.
- `converter.py`: SenseVoice Core ML conversion policy.
- `utils/`: private audio frontend, model, postprocessing, and types.

