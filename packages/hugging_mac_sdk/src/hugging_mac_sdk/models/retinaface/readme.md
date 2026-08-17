# RetinaFace

## Summary

SDK package for the lightweight `py-feat/retinaface` MobileNet0.25 checkpoint. A single
forward pass returns face boxes, confidence scores, and five alignment landmarks (eyes,
nose, and mouth corners); no additional landmark model is required.

## Canonical identity, runtimes, and conversion

- Model ID: `py-feat/retinaface`
- Revision: `31702389094fccc7060c15299e6ad712ee880de6`
- Variant: `mobilenet0.25`
- Source runtime: `pytorch-mps`
- Converted runtime: `coreml`
- License: MIT

The downloaded resource contains only `config.json` and
`mobilenet0.25_Final.pth`. The Models page can run the checkpoint directly with
PyTorch/MPS or convert it locally to an FP16 Core ML `mlprogram`. Old prebuilt
RetinaFace Core ML packages are neither discovered nor supported.

## Capability API

- `FaceDetection.detect_faces(DetectionRequest) -> FaceDetectionResponse`
- Coordinates are restored to pixels in the original image.
- Confidence threshold and NMS IoU are applied after either runtime's raw outputs.

## Package structure

- `model.yaml`: pinned source, artifacts, runtimes, capability, and license.
- `config.py`: frozen instance and resource options.
- `definition.py`: PyTorch/Core ML factories, converter, and model registration.
- `instance.py`: lifecycle and runtime-independent face detection.
- `torch.py`: PyTorch/MPS loading and inference.
- `converter.py`: local `.pth` to Core ML conversion.
- `coreml.py`: converted Core ML loading and inference.
- `resources.py`: source download, integrity checks, conversion, status, and deletion.
- `utils/modeling.py`: MobileNet0.25 RetinaFace architecture.
- `utils/preprocess.py`: fixed-size letterbox and mean subtraction.
- `utils/postprocess.py`: prior decoding, filtering, NMS, and coordinate restoration.
