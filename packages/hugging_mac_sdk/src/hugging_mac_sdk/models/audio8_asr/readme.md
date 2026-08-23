# Audio8 ASR Technical Notes

This package implements the `SpeechTranscription` capability with one shared
instance contract across its PyTorch and Core ML engines. `model.yaml` is the
only source for identity, variants, runtimes, files, sources, and presentation.

## Engine boundary

`instance.py` decodes caller audio, serializes inference, delegates prepared
audio to the selected engine, and maps the engine result to the public response.
`torch.py` assembles the local model implementation with the Torch provider.
`coreml.py` combines the converted audio tower with the cached language-model
path while preserving the same capability output.

The package does not execute model-repository Python. Its local implementation
constructs the supported architecture from declared configuration and loads only
the selected data files.

## Conversion and resources

`converter.py` owns the model-specific split between the audio tower and language
components, fixed audio-duration functions, weight export, and atomic publication
of the target directory. `resources.py` consumes the injected artifacts and
checks their complete required-file contracts; it does not duplicate source or
path facts.

Audio decoding, model layers, Core ML helpers, and private engine result types
remain under `utils/` and never enter public schemas.
