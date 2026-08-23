# Nemotron 3.5 ASR

## Summary

Canonical package for `nvidia/nemotron-3.5-asr-streaming-0.6b`. It consumes FluidInference's precompiled Core ML derivative and provides short-form multilingual transcription through the SDK capability API.

## Runtimes and variants

- Runtime: `coreml` (`cpu-and-neural-engine` by default; preprocessor and RNN-T control flow also use CPU).
- Variants: `latin-{560,1120,2240,4480}ms` and `multilingual-{560,1120,2240,4480}ms`.
- Default: `multilingual-2240ms`, the upstream recommended throughput/quality tier.
- `latin` uses the pruned 2,828-token vocabulary for en/es/fr/it/pt/de. `multilingual` retains all 13,087 tokens and supports 100+ language prompts. The current capability schema has no language field, so inference uses the upstream `auto` prompt (ID 101).

## Capability API

`SpeechTranscription.transcribe(TranscriptionRequest) -> TranscriptionResponse` accepts mono/resampled 16 kHz audio, chunks it at the selected tier, maintains mel/Conformer/LSTM caches, performs greedy fused RNN-T decoding, removes emitted language tags, and returns SDK timing metadata.

## Package structure

- `__init__.py`: definition, manifest, and registration exports.
- `model.yaml`: canonical identity, trusted Core ML sources, download selections, variants, runtime, and artifacts.
- `config.py`: frozen Core ML/resource options and variant parsing.
- `definition.py`: Core ML factory and registry binding.
- `instance.py`: lifecycle, locking, preprocessing, and capability response.
- `coreml.py`: graph loading, streaming caches, greedy RNN-T decode, and tokenization.
- `resources.py`: bundle validation, explicit download/status/delete lifecycle.
- `utils/audio.py`: shared-schema audio decoding/resampling adapter.
- `utils/types.py`: private prepared-audio and engine-output values.

The implementation follows the artifact's `metadata.json` rather than hard-coding tier tensor shapes. Model loading never downloads; installation remains an explicit SDK resource operation.
