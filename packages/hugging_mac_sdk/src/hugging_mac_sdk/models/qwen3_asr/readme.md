# Qwen3-ASR

## Summary

Canonical package for `qwen/qwen3-asr`. It currently manages a prebuilt Core ML encoder, embedding graph, and stateful decoder graph set.

## Runtimes and variants

- Variant: `0.6b`
- Runtime: `coreml`
- Runtime artifact: prebuilt INT8 multi-graph Core ML bundle
- Shared artifact: the official Qwen tokenizer, reused by every variant and runtime

## Capability API

The package implements `SpeechTranscription.transcribe(TranscriptionRequest) -> TranscriptionResponse` with mono 16 kHz preprocessing, Whisper-compatible 128-bin features, the fixed-shape Core ML encoder, batched split-decoder prefill, MLState KV caches, greedy generation, and local tokenizer decoding.

## Package structure

- `__init__.py`: definition, manifest, and registration exports.
- `model.yaml`: canonical identity and graph artifact metadata.
- `config.py`: typed Core ML and resource options.
- `definition.py`: Core ML factory and registry binding.
- `instance.py`: graph lifecycle and transcription capability composition.
- `coreml.py`: multi-graph session loading and unloading.
- `resources.py`: graph-and-tokenizer bundle validation, status, download, and deletion.

The catalog exposes one install action for `coreml-int8`. Before downloading that runtime artifact,
the shared-artifact layer checks and installs the three tokenizer files from the official
`Qwen/Qwen3-ASR-0.6B` repository. The tokenizer remains reusable if another runtime or variant is
installed later.
