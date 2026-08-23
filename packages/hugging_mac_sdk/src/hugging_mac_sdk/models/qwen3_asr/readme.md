# Qwen3 ASR Technical Notes

The package implements `SpeechTranscription` with a stateful Core ML generation
path. `model.yaml` owns identity, runtime, sources, and required files.

`coreml.py` owns session loading and state objects for the audio encoder,
embeddings, and decoder. `instance.py` owns audio preprocessing, generation
control, local token decoding, lifecycle, and public response construction.
Runtime-private graph inputs and states never leave the package.

The definition injects the selected runtime artifact and reusable tokenizer.
`resources.py` evaluates their combined readiness and manages explicit
download/status/delete operations without maintaining another file inventory.
