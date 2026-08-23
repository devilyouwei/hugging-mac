# RetinaFace MobileNet0.25

## Purpose

RetinaFace detects faces and returns five alignment landmarks for each face.

## Support

- Model ID: `py-feat/retinaface`
- Variants: `mobilenet0.25` (default)
- Runtimes: `pytorch-mps`, `coreml` (default)
- Capabilities: `face-detection` through `FaceDetection`
- Platform: macOS; Core ML requires Apple Silicon
- Optional dependency set: `yolo`

## Prepare resources

Download the PyTorch model directly. Register the converter when preparing Core
ML locally.

## Example

```python
import asyncio

from hugging_mac_sdk import DetectionRequest, ImageInput, ModelSdk
from hugging_mac_sdk.capabilities import FaceDetection
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.retinaface import register_retinaface


async def main() -> None:
    sdk = ModelSdk()
    converters = ConverterRegistry()
    register_retinaface(sdk.registry, converters)
    model_id = "py-feat/retinaface"

    await sdk.resources.download_source(model_id, variant="mobilenet0.25")
    async with await sdk.load(model_id, runtime="pytorch-mps") as handle:
        detector = handle.require(FaceDetection)
        result = await detector.detect_faces(
            DetectionRequest(
                image=ImageInput(path="people.jpg"),
                confidence=0.7,
            )
        )
        for face in result.faces:
            print(face.box, face.landmarks)


asyncio.run(main())
```

## Notes and limits

The five points represent the eyes, nose, and mouth corners. This model does not
return a dense face mesh.

## License

The model is licensed under `MIT`.
