# Qwen3 TTS Technical Notes

This package exposes `SpeechSynthesis` over MLX and Core ML engine protocols.
`model.yaml` is the only owner of runtime compatibility and static resource facts.

`mlx.py` owns autoregressive generation, speech tokenization, and optional
reference-audio conditioning. It omits both reference arguments for direct
generation and supplies both for voice cloning. `coreml.py` owns the compiled
autoregressive state, embedding and decoder execution order, supported request
checks, and waveform decoding.
`instance.py` keeps capability validation, lifecycle, locking, and response
construction common to both paths.

The definition injects the selected runtime files and reusable tokenizers.
`resources.py` reports readiness for the exact selection and does not infer
resources from model names. BPE helpers and private engine outputs remain under
`utils/`.
