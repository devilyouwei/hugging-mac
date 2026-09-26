# Audio8 TTS Technical Notes

This package exposes `SpeechSynthesis` through one instance type composed with a
PyTorch, MLX, or native Core AI engine. Model identity and the compatibility matrix are read from
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

All neural stages run through CoreAIProvider: Falcon-H1 slow AR, fast AR,
reference-audio codec encoding, and waveform decoding. No PyTorch, Transformers,
or torchaudio import is performed during native inference. Host orchestration
uses NumPy sampling, the Rust tokenizer, SoundFile, and SciPy resampling.

`utils/coreai_slow.py` exports a recurrent token step with explicit mutable
KV, convolution, and SSM buffers. Prompt ingestion uses the same step from zero
states; this supports arbitrary prompt lengths within the declared context, but
is not a parallel prefill optimization. `utils/coreai_export.py` exports native
fast embeddings with per-frame KV reset and both codec directions, replacing
complex RoPE and floating-point shape arithmetic with exportable equivalents.
`utils/coreai_generation.py` preserves prompt segment boundaries, filtering order,
and repetition-aware sampling. RNG streams differ from PyTorch, so sampled speech
is not expected to be bit-identical. Non-native-rate reference audio is resampled
with SciPy rather than torchaudio.

`converter.py` derives runtime dimensions and state shapes from loaded weights,
exports four graphs, and writes their runtime metadata. It never retains the
original checkpoint or copies shared tokenizer files. Model paths, remote file
selection, and graph names come from the YAML declarations. Explicit resource
download with `options.runtime="coreai"` selects the native artifact and shares;
the default resource download still prepares PyTorch for local conversion.

`coreai.py` validates local files, lazily opens the reference encoder when needed,
and serializes inference. Persistent state is native and scoped to one request.
Cancellation waits for native calls before clearing states and releasing the
lock. Failed load closes partial sessions. Native conversion uses sibling staging
and publishes only after full inventory validation; failure preserves old files.

The native artifact has a separate canonical path from the former hybrid build.
Existing hybrid files are not deleted or treated as native. Core AI auto selection
allows acceleration but does not imply all operations execute on ANE; some
operations in the current graph fail ANE validation. Native export still requires
PyTorch in the isolated conversion environment documented in
`docs/sdk/models/audio8-tts.md`.
