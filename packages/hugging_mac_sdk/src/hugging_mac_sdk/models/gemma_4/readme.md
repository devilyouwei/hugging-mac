# Gemma 4 Technical Notes

The package implements the shared `Chat` protocol for text and image-bearing
messages. `model.yaml` owns model identity, variant selection, sources, and file
contracts.

`definition.py` resolves the selected MLX artifact and reusable tokenizer before
injecting them into the factory. `instance.py` owns lifecycle, request
serialization, and chat/stream capability registration. `mlx.py` owns processor
loading, text and vision preparation, generation, cancellation, and conversion
to private output events.

The public boundary contains chat messages, image references, responses, and
stream events only. MLX arrays, processors, prompt templates, and generation
state stay inside the package.
