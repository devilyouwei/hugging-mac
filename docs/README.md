# Hugging Mac Documentation

Hugging Mac is a local model studio for Apple Silicon. It brings vision, speech,
audio, language, and multimodal models into one application while keeping model
inputs and generated data on your Mac.

This documentation is for people who want to run the Studio or call its Python
SDK. It describes supported workflows and public interfaces. Implementation
notes and contributor standards live beside the code they describe.

## Start here

- [Getting started](getting-started.md) installs the project and runs the Studio.
- [Model Library](model-library.md) explains downloads, conversions, variants,
  runtimes, and loaded instances.
- [Studio guide](studio.md) introduces the available Apps and Games.
- [Python SDK](sdk/README.md) shows how to manage and call models from code.
- [Model SDK reference](sdk/models/README.md) provides a complete example for
  every integrated model family.

## What you can run

Hugging Mac currently includes local workflows for object detection, pose
estimation, instance segmentation, face and hand detection, speech recognition,
speech enhancement, voice activity detection, text-to-speech, chat, and
vision-language prompts. The Model Library shows the exact variants and runtimes
available on the current installation.

## A few useful terms

- A **model** is one stable model family, such as YOLOv8 or Qwen3.5.
- A **variant** is a selectable size or edition of that model.
- A **runtime** is the execution backend, such as Core ML, MLX, PyTorch MPS, or
  ONNX Runtime.
- An **artifact** is a set of files needed by a runtime. It can be downloaded or,
  when supported, converted locally.
- A **capability** is the operation exposed by a model, such as transcription,
  chat, or object detection.

## Local-first behavior

The Studio runs its models locally. Files selected in the UI are sent only to
the local Hugging Mac backend unless a feature explicitly says otherwise. Model
downloads contact their declared upstream source. Hugging Mac does not enable
external telemetry by default.

Model files and third-party dependencies retain their own licenses. Check the
license shown for a model before redistributing its files or using it in a
commercial product.
