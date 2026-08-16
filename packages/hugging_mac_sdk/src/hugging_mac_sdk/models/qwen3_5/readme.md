# Qwen3.5

## Summary

Canonical SDK package for the `qwen/qwen3.5` family. Parameter scale and OptiQ quantization are variant/artifact configuration, not package or model-ID suffixes.

## Runtimes and variants

- Variants: `9b`, `4b`, `2b`
- Runtime: `mlx`
- Current artifacts: pinned MLX OptiQ 4-bit snapshots

## Capability APIs

- `Chat.chat(ChatRequest) -> ChatResponse`
- `Chat.stream_chat(ChatRequest) -> AsyncIterator[ChatStreamEvent]`
- Vision-language input is accepted through the shared chat request schemas; backend tensors never cross the API boundary. The instance currently registers `Chat`; the separate manifest capability name `vision-language-generation` has no standalone protocol and should not be interpreted as an additional handle capability.

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: variants, MLX runtime, pinned sources, and artifacts.
- `config.py`: typed variant and MLX options.
- `definition.py`: variant-aware MLX factory and registration.
- `instance.py`: lifecycle, chat capabilities, and engine protocol.
- `mlx.py`: MLX language/vision engine.
- `resources.py`: per-variant snapshot validation and lifecycle.
- `utils/`: private generated-output types.
