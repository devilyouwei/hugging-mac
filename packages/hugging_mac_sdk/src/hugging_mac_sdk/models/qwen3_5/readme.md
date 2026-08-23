# Qwen3.5 Technical Notes

The package implements text and vision-language requests through the shared
`Chat` capability. Declarative model, variant, runtime, tokenizer, and source
facts live only in `model.yaml`.

`definition.py` injects the selected MLX weights and reusable tokenizer into one
factory. `instance.py` owns lifecycle, locking, chat capability registration,
and stable error translation. `mlx.py` owns text/image processor setup, prompt
construction, generation, streaming, and vision-side model loading.

Public responses contain SDK chat schemas only. Processors, MLX arrays, caches,
and backend generation objects remain private to the engine and `utils/`.
