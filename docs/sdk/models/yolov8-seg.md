# YOLOv8 Instance Segmentation

## Purpose

YOLOv8 Seg detects objects and returns a visible polygon for each object instance.

## Support

- Model ID: `ultralytics/yolov8-seg`
- Variants: `n` (default), `s`, `m`
- Runtimes: `pytorch-mps`, `coreml` (default), `onnx`
- Capabilities: `instance-segmentation` through `InstanceSegmentation`
- Platform: macOS; accelerated runtimes target Apple Silicon
- Optional dependency set: `yolo`

## Prepare resources

Download the source for the selected size. Core ML and ONNX can be created
through the registered converter.

## Example

```python
import asyncio

from hugging_mac_sdk import ImageInput, ModelSdk, SegmentationRequest
from hugging_mac_sdk.capabilities import InstanceSegmentation
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.yolov8_seg import register_yolov8_seg


async def main() -> None:
    sdk = ModelSdk()
    converters = ConverterRegistry()
    register_yolov8_seg(sdk.registry, converters)
    model_id = "ultralytics/yolov8-seg"

    await sdk.resources.download_source(model_id, variant="n")
    async with await sdk.load(
        model_id,
        variant="n",
        runtime="pytorch-mps",
    ) as handle:
        segmenter = handle.require(InstanceSegmentation)
        result = await segmenter.segment(
            SegmentationRequest(
                image=ImageInput(path="objects.jpg"),
                confidence=0.4,
            )
        )
        for segment in result.segments:
            print(segment.label, segment.box, segment.polygons)


asyncio.run(main())
```

## Notes and limits

Polygons describe visible regions in source-image coordinates. Rasterize them in
the calling application when a pixel mask is required.

## License

The model is licensed under `AGPL-3.0`. Commercial distribution may require a
separate Ultralytics license.
