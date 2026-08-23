# Silero VAD Technical Notes

The package implements `VoiceActivityDetection` with Core ML and ONNX engines.
Identity, runtime compatibility, and resource layout come from `model.yaml`.

`instance.py` owns audio decoding, request thresholds, state reset, segment
assembly, lifecycle, and serialization. `coreml.py` and `onnx.py` own backend
session state and implement the same frame-level private engine contract.

Streaming state is scoped to the model instance and reset between unrelated
inputs. Private recurrent tensors and sample-level probabilities do not cross the
capability boundary. `resources.py` manages the injected selected artifact.
