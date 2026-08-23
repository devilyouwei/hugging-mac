# Kokoro Technical Notes

The package provides one `SpeechSynthesis` instance contract over PyTorch and
Core ML engines. All static resource and runtime facts are read from `model.yaml`.

`torch.py` owns the upstream-compatible synthesis path and speed handling.
`coreml.py` owns compiled-model loading, bounded text segmentation, prediction
length checks, recursive retry at natural boundaries, and final waveform joins.
The compiled path trusts the model-reported valid sample length and applies only
a short boundary fade.

`instance.py` validates capability input and serializes synthesis. Phonemization,
voice data, upstream notices, and private output types remain under `utils/`.
`resources.py` manages the exact artifact selected by the definition without
declaring repositories or file lists in Python.
