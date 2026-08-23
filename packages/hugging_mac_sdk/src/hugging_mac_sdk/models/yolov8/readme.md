# YOLOv8 Object Detection

## Summary

SDK package for `ultralytics/yolov8`. It is separate from Pose and Seg because each exposes a different public capability contract.

## Runtimes and variants

- Variants: `n`, `s`, `m`
- Runtimes: `pytorch-mps`, `coreml`, `onnx`

## Capability API

- `ObjectDetection.detect(DetectionRequest) -> DetectionResponse`

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: variants, runtimes, and artifacts with their owned sources.
- `config.py`: typed runtime/conversion options.
- `definition.py`: runtime factories, converter binding, and registration.
- `instance.py`: lifecycle and object-detection capability.
- `torch.py`, `coreml.py`, `onnx.py`: backend engines.
- `resources.py`: source/converted artifact lifecycle.
- `converter.py`: model-specific export policy.
- `utils/`: private checkpoint, preprocessing, postprocessing, and types.
