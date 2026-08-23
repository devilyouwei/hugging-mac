# MOSS-TTS-Nano Technical Notes

This package exposes `SpeechSynthesis` through the MLX runtime. `model.yaml` is
the only owner of model identity, runtime compatibility, and resource layout.

`mlx.py` owns MLX-Audio loading, reference-audio conditioning, autoregressive
generation, codec decoding, and stereo-to-mono response normalization.
`instance.py` owns lifecycle, synthesis serialization, error mapping, and the
public response. Reference audio is optional, and the paired reference text
accepted by the shared request schema is intentionally not passed to MLX-Audio.
When reference audio is supplied the engine uses voice-clone mode and passes
`ref_audio`; otherwise it omits that argument and uses direct continuation mode.

The autoregressive model runs through MLX on the Apple GPU. MOSS Audio Tokenizer
encoding and decoding use its CPU execution path, so this is a hybrid pipeline
reported as `mlx`. Streaming is not exposed because the current MLX-Audio model
implementation rejects streaming generation.

`resources.py` downloads and validates the model and its separately declared
shared audio tokenizer. Loading and inference never initiate a network download.
