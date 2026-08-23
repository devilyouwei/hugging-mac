# YOLOv8 Object Detection

## Purpose

YOLOv8 detects COCO objects in images, video frames, and camera frames.

## Support

- Model ID: `ultralytics/yolov8`
- Variants: `n` (default), `s`, `m`
- Runtimes: `pytorch-mps`, `coreml` (default), `onnx`
- Capabilities: `object-detection` through `ObjectDetection`
- Platform: macOS; accelerated runtimes target Apple Silicon
- Optional dependency set: `yolo`

## Prepare resources

Download a PyTorch source model. Core ML and ONNX can then be prepared through
the registered converter; Core ML may also be downloadable when a prebuilt copy
is available.

## Example

```python
import asyncio

from hugging_mac_sdk import DetectionRequest, ImageInput, ModelSdk
from hugging_mac_sdk.capabilities import ObjectDetection
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.yolov8 import register_yolov8


async def main() -> None:
    sdk = ModelSdk()
    converters = ConverterRegistry()
    register_yolov8(sdk.registry, converters)
    model_id = "ultralytics/yolov8"

    await sdk.resources.download_source(model_id, variant="n")
    async with await sdk.load(
        model_id,
        variant="n",
        runtime="pytorch-mps",
    ) as handle:
        detector = handle.require(ObjectDetection)
        result = await detector.detect(
            DetectionRequest(
                image=ImageInput(path="street.jpg"),
                confidence=0.4,
                iou_threshold=0.5,
                classes=(0, 2),
            )
        )
        print(result.detections)


asyncio.run(main())
```

## Notes and limits

`n`, `s`, and `m` increase in model size and typical accuracy. Use `classes` to
filter COCO class IDs and `max_detections` to cap returned results.

## License

The model is licensed under `AGPL-3.0`. Commercial distribution may require a
separate Ultralytics license.
