# SenseVoice Technical Notes

The package implements `SpeechTranscription` and `SpeechUnderstanding` with a
shared instance over PyTorch and Core ML engines. `model.yaml` owns all
declarative facts.

The package does not execute repository Python. `utils/frontend.py` implements
the audio feature frontend, `utils/modeling.py` implements the supported
inference network, and `utils/postprocess.py` converts model tokens into text,
language, emotion, and acoustic-event values.

`torch.py` and `coreml.py` implement the same private engine protocol.
`converter.py` owns the model-specific export boundary. `instance.py` selects the
plain or rich public response without exposing logits, tensors, or control
tokens. `resources.py` validates only injected artifact contracts.
