# Audio8 TTS Technical Notes

This package exposes `SpeechSynthesis` through one instance type composed with a
PyTorch, MLX, or hybrid Core AI engine. Model identity and the compatibility matrix are read from
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

## Core AI pipeline

The Core AI build is deliberately hybrid. `coreai.py` delegates fast-AR cached
steps and codec waveform decoding to `CoreAIProvider`. Falcon-H1 slow AR,
sampling, tokenization, and optional reference-audio encoding remain PyTorch CPU.
The runtime reports `coreai`; device selects only the Core AI stages. It does
not imply that every stage uses ANE/GPU or that PyTorch can be uninstalled.

`converter.py` receives the parsed package and its namespaced YAML extension.
`utils/coreai_export.py` provides tensor-only adapters with explicit KV cache
inputs/outputs. The fast adapter exports one step; its caches reset for every
frame and request. Codec export replaces complex RoPE construction with real
sin/cos and floating-point padding arithmetic with equivalent integer shapes.
Original weights and module behavior are preserved. Conversion retains declared
source files for CPU stages, but never copies the shared tokenizer. Every graph
file and retained source file is listed in the target's required inventory.

`definition.py` injects artifacts and graph paths into factories and converters;
`resources.py` provides read-only complete-file status, explicit conversion,
overwrite protection and safe deletion. The registration function optionally
accepts a converter registry. Unsupported variant/format pairs fail explicitly.
There is no model migration or duplicate ID for the upstream organization rename.

Explicit resource download with `options.runtime="coreai"` selects the target's
remote source and shared tokenizer without separately downloading the PyTorch
source artifact. Remote prefixes are stripped by the generic downloader, keeping
downloaded and locally converted directories interchangeable. The default resource
download still prepares PyTorch for inference or local conversion.

Conversion uses a sibling staging directory. Failure or cancellation before
publication leaves the old output intact. Native inference and cleanup are
serialized, and cancellation waits for native work before releasing the lock.
Failed loading closes already loaded sessions and partial PyTorch state. Plain
synthesis does not load the PyTorch codec; reference cloning loads its encoder
on demand. The unused PyTorch fast decoder weights are released after hooking
the Core AI sessions into the original generation loop.

The conversion environment and reproducible package versions are documented in
`docs/sdk/models/audio8-tts.md`. Optional dependencies import only on the selected
runtime/export path. `coreai-torch` uses BSD-3-Clause; model weights retain their
upstream license declared by the manifest/variant. Converted artifacts include
CPU weights as well as native graphs, adding disk usage and native session memory.
