# Audio8 TTS Technical Notes

This package exposes `SpeechSynthesis` through one instance type composed with a
PyTorch or MLX engine. Model identity and the compatibility matrix are read from
`model.yaml`, not repeated in Python or this document.

Both variants use the `pytorch` runtime with CPU as the default device because
it currently performs better than MPS for these model implementations. MPS
remains available through an explicit instance option, and the engine falls
back to CPU when MPS is unavailable or raises during loading or inference. The
`0.6b-preview` variant also supports Apple GPU execution through `mlx`.

## Engine boundary

`instance.py` validates reference-audio/reference-text pairing, owns synthesis
serialization, and maps the private engine result to the public little-endian
float waveform response. `torch.py` owns PyTorch device selection and stability
setup. Its load boundary also adapts the removed Transformers 4
Falcon-H1 cache constructor to the Transformers 5 lazy `DynamicCache`; this is
kept local because the MLX dependency set requires Transformers 5. `mlx.py` owns
MLX model loading, reference conditioning, and Apple GPU generation.

`resources.py` and `mlx_resources.py` resolve the injected runtime artifact and
required reusable tokenizer files. Neither resolver chooses a model variant from
hard-coded strings.

Voice-profile persistence and engine-only output values stay under `utils/`.
Capability callers never receive codec tokens, framework arrays, or model
objects.
