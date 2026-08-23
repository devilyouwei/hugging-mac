# YOLOv8 Object Detection Technical Notes

This package owns the `ObjectDetection` contract. Pose and segmentation are
separate packages because their public outputs and postprocessing differ.
`model.yaml` owns all static configuration.

`torch.py`, `coreml.py`, and `onnx.py` adapt their providers to one private engine
protocol. `utils/preprocess.py` owns letterboxing and tensor preparation;
`utils/postprocess.py` owns output decoding, class filtering, class-aware NMS,
and restoration to source-image coordinates. Runtime engines return raw values
before this common postprocessing boundary.

`converter.py` exports raw model outputs so every runtime uses the same SDK
postprocessing. Converted targets are checked and published at the exact artifact
path injected by the definition. No engine imports the Ultralytics Python
package.
