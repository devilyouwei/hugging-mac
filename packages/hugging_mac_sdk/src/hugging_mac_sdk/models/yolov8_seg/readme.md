# YOLOv8 Instance Segmentation Technical Notes

This package owns the `InstanceSegmentation` contract and its model-private YOLO
mask implementation. `model.yaml` owns static configuration.

The PyTorch, Core ML, and ONNX engines produce a common private raw-output shape.
`utils/preprocess.py` performs letterboxing. `utils/postprocess.py` decodes boxes,
applies class-aware NMS, combines mask coefficients with prototypes, crops and
resizes masks, and extracts immutable source-coordinate polygons.

`converter.py` exports raw detection and prototype outputs so postprocessing is
identical across runtimes. Binary masks, NumPy arrays, framework tensors, and
checkpoint structures never enter the public response.
