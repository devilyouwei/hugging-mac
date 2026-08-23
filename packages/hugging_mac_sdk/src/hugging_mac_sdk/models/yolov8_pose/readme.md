# YOLOv8 Pose Technical Notes

This package owns the `PoseEstimation` contract and its model-private YOLO pose
implementation. All static declarations live in `model.yaml`.

The PyTorch, Core ML, and ONNX engines share one private raw-output protocol.
`utils/preprocess.py` prepares letterboxed input. `utils/postprocess.py` decodes
person boxes and keypoint coordinates, applies filtering/NMS, and maps results to
source pixels.

`converter.py` preserves raw pose output and leaves keypoint decoding inside the
package, ensuring the same public response across runtimes. Checkpoint loading,
intermediate tensors, and keypoint layouts remain private under `utils/`.
