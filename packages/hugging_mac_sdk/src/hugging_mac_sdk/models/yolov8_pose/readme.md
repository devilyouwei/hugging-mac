# YOLOv8 Pose

## Summary

SDK package for `ultralytics/yolov8-pose`. It owns the person-keypoint contract and pose-specific preprocessing/postprocessing.

## Runtimes and variants

- Variants: `n`, `s`, `m`
- Runtimes: `pytorch-mps`, `coreml`, `onnx`

## Capability API

- `PoseEstimation.estimate_pose(PoseRequest) -> PoseEstimationResponse`

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: variants, runtimes, and artifacts with their owned sources.
- `config.py`: typed runtime/conversion options.
- `definition.py`: runtime factories, converter binding, and registration.
- `instance.py`: lifecycle and pose-estimation capability.
- `torch.py`, `coreml.py`, `onnx.py`: backend engines.
- `resources.py`: source/converted artifact lifecycle.
- `converter.py`: pose export policy.
- `utils/`: private checkpoint, preprocessing, keypoint postprocessing, and types.
