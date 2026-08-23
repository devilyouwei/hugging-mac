# MediaPipe Hand Detection

## Summary

SDK package for `qualcomm/MediaPipe-Hand-Detection`. To callers it is one model with one `HandDetection` capability. Internally it contains the two-stage MediaPipe pipeline: a palm detector followed, when requested, by a cropped hand-landmark detector.

## Canonical identity, runtime, and variant

- Model ID: `qualcomm/mediapipe-hand-detection`
- Hugging Face source: follows the repository's `main` branch without content-hash validation
- Upstream release: `0.60.0`
- Variant: `float`
- Input size for both graphs: `256x256`
- Runtimes: downloaded `onnx` and locally converted native `coreml`
- License: `other`, as declared by the upstream Hugging Face repository; upstream licensing obligations still apply to downloaded artifacts

Each runtime artifact is a directory containing both components:

- `hand_detector`: palm boxes, confidence scores, and seven detector keypoints
- `hand_landmark_detector`: presence score, left/right score, and 21 three-dimensional landmarks

Native Core ML artifacts use FP32. A single conversion request converts both ONNX graphs into `hand_detector.mlpackage` and `hand_landmark_detector.mlpackage`, stages them together, and atomically publishes the containing `coreml` directory. Resource status considers the Core ML runtime available only when both packages are complete.

## Capability API

```python
capability = instance.get_capability(HandDetection)
response = await capability.detect_hands(
    HandDetectionRequest(
        image=image,
        include_landmarks=True,
        input_mirrored=False,
        confidence=0.5,
        landmark_confidence=0.5,
    )
)
```

`HandDetectionRequest.include_landmarks` defaults to `False`:

- `False`: loads and executes only the palm detector. The landmark model does not consume runtime memory.
- `True`: lazily loads the landmark model on the first qualifying palm and executes it once per retained palm.

The response always returns source-image pixel boxes. With landmarks enabled and a hand-presence score above `landmark_confidence`, each hand also returns 21 landmarks, handedness, handedness confidence, and landmark confidence. Landmark `x` and `y` are projected into source-image pixels; `z` is the model-relative depth value.

Set `input_mirrored=True` when the caller horizontally flips a camera frame before inference. The SDK then restores handedness to the original physical left/right hand while leaving the mirrored landmark coordinates unchanged.

## Core ML conversion verification

The two-graph archive was converted as one operation and compared with ONNX Runtime on the same random input. Maximum absolute differences were:

- detector box coordinates: `6.866e-5`
- detector scores: `1.526e-4`
- landmark presence score: `3.92e-8`
- landmark handedness score: `4.59e-6`
- landmark coordinates: `4.47e-7`

## Package structure

- `model.yaml`: identity, dual artifacts, download selections, runtime metadata, capability, and license.
- `config.py`: frozen runtime and resource options.
- `definition.py`: one public model definition across ONNX and Core ML.
- `converter.py`: one atomic conversion workflow for both graphs.
- `instance.py`: unified `HandDetection` capability and optional second-stage orchestration.
- `onnx.py`: eager detector and lazy landmarker ONNX Runtime sessions.
- `coreml.py`: eager detector and lazy landmarker Core ML models.
- `resources.py`: archive validation, dual-package completeness checks, status, and deletion.
- `utils/preprocess.py`: detector letterbox and rotated landmark ROI extraction.
- `utils/postprocess.py`: palm anchor decoding, weighted NMS, landmark projection, and handedness decoding.
- `utils/types.py`: private prepared-image, palm-candidate, and landmark-crop types.
