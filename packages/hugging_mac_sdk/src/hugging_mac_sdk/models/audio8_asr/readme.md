# Audio8-ASR

## Summary

Canonical SDK package for `audio8/audio8-asr`. It provides multilingual short-form speech transcription with the `0.1b` model variant.

## Runtimes and variants

- Variant: `0.1b`
- Runtimes: `pytorch-mps`, `coreml`
- Artifacts and quantization details are declared in `model.yaml`; they are not separate model identities.

## Capability API

- `SpeechTranscription.transcribe(TranscriptionRequest) -> TranscriptionResponse`

Applications must obtain this capability from a loaded model handle and must not import this package's runtime classes.

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: canonical metadata, variants, runtimes, and artifacts with their owned sources.
- `config.py`: typed instance and resource configuration.
- `definition.py`: runtime factories, converter binding, and registry integration.
- `instance.py`: lifecycle, capability registration, and runtime-neutral transcription flow.
- `torch.py`: PyTorch/MPS inference engine.
- `coreml.py`: Core ML inference engine.
- `resources.py`: source and converted-artifact lifecycle.
- `converter.py`: Audio8-ASR to Core ML conversion policy.
- `utils/`: private audio, model, Core ML, and output helpers.
