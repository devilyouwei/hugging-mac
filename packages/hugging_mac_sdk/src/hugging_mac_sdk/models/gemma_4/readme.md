# Gemma 4

## Summary

Canonical SDK package for the `google/gemma-4` family. Effective parameter scale and QAT OptiQ quantization are variant/artifact configuration, not package or model-ID suffixes.

## Runtimes and variants

- Variants: `e4b` (default), `e2b`
- Runtime: `mlx`
- Current artifacts: pinned MLX QAT OptiQ mixed-precision snapshots

## Capability API

- `Chat.chat(ChatRequest) -> ChatResponse`
- `Chat.stream_chat(ChatRequest) -> AsyncIterator[ChatStreamEvent]`
- Both LLM text-only and VLM text-plus-image requests use the shared `Chat` capability. Images are supplied on chat messages; runtime tensors and processors remain private.

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: canonical identity, variants, MLX runtime, pinned sources, and artifacts.
- `config.py`: typed variant, resource, and runtime options.
- `definition.py`: variant-aware MLX factory and registration.
- `instance.py`: lifecycle, shared chat capability, and engine protocol.
- `mlx.py`: MLX text/vision generation engine and OptiQ vision-sidecar loading.
- `resources.py`: per-variant snapshot validation and lifecycle.
- `utils/`: private generation result types.
