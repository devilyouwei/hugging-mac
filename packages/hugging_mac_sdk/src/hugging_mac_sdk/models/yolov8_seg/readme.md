# YOLOv8 Instance Segmentation

## Summary

SDK package for `ultralytics/yolov8-seg`. It remains separate because instance masks and polygons form a different public capability contract from detection and pose.

## Runtimes and variants

- Variants: `n`, `s`, `m`
- Runtimes: `pytorch-mps`, `coreml`, `onnx`

## Capability API

- `InstanceSegmentation.segment(SegmentationRequest) -> SegmentationResponse`

## Package structure

- `__init__.py`: public definition, manifest, and registration exports.
- `model.yaml`: variants, runtimes, and artifacts with their owned sources.
- `config.py`: typed runtime/conversion options.
- `definition.py`: runtime factories, converter binding, and registration.
- `instance.py`: lifecycle and segmentation capability.
- `torch.py`, `coreml.py`, `onnx.py`: backend engines.
- `resources.py`: source/converted artifact lifecycle.
- `converter.py`: segmentation export policy.
- `utils/`: private checkpoint, preprocessing, mask postprocessing, and types.
