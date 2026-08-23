# MediaPipe Hand Detection

## Purpose

MediaPipe Hand Detection returns hand boxes and can optionally estimate 21 hand
landmarks and handedness.

## Support

- Model ID: `qualcomm/mediapipe-hand-detection`
- Variants: `float` (default)
- Runtimes: `onnx`, `coreml` (default)
- Capabilities: `hand-detection` through `HandDetection`
- Platform: ONNX supports CPU execution; Core ML requires Apple Silicon
- Optional dependency set: `yolo`

## Prepare resources

The ONNX model is downloadable. Core ML can be prepared with the registered
converter.

## Example

```python
import asyncio

from hugging_mac_sdk import HandDetectionRequest, ImageInput, ModelSdk
from hugging_mac_sdk.capabilities import HandDetection
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.mediapipe_hand_detection import (
    register_mediapipe_hand_detection,
)


async def main() -> None:
    sdk = ModelSdk()
    converters = ConverterRegistry()
    register_mediapipe_hand_detection(sdk.registry, converters)
    model_id = "qualcomm/mediapipe-hand-detection"

    await sdk.resources.download_source(
        model_id,
        variant="float",
        options={"runtime": "onnx"},
    )
    async with await sdk.load(model_id, runtime="onnx") as handle:
        detector = handle.require(HandDetection)
        result = await detector.detect_hands(
            HandDetectionRequest(
                image=ImageInput(path="hands.jpg"),
                include_landmarks=True,
                input_mirrored=False,
            )
        )
        print(result.hands)


asyncio.run(main())
```

## Notes and limits

Landmarks add work after each detected hand. Set `input_mirrored=True` when the
image was horizontally mirrored before inference so handedness is interpreted
correctly.

## License

The upstream source declares a model-specific `other` license. Review its terms
before redistribution.
