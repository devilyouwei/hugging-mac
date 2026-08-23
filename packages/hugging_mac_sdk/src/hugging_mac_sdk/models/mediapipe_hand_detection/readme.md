# MediaPipe Hand Detection Technical Notes

This package exposes one `HandDetection` capability over a two-stage palm and
landmark pipeline. `model.yaml` is canonical for all static facts.

## Pipeline

The detector engine is loaded eagerly. The landmark engine is loaded lazily only
when a request enables landmarks and at least one palm passes filtering. Both the
ONNX and Core ML engines implement the same private engine protocol, so
`instance.py` owns stage orchestration and the public response mapping.

`utils/preprocess.py` performs detector letterboxing and rotated hand crops.
`utils/postprocess.py` decodes palm anchors, applies weighted suppression,
projects landmarks into source-image coordinates, and corrects handedness for
mirrored input.

`converter.py` converts both stages as one atomic operation. The target becomes
available only when both converted models satisfy the required-file contract.
