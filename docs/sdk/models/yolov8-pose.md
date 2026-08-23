# YOLOv8 Pose

## Purpose

YOLOv8 Pose detects people and estimates 17 body keypoints per person.

## Support

- Model ID: `ultralytics/yolov8-pose`
- Variants: `n` (default), `s`, `m`
- Runtimes: `pytorch-mps`, `coreml` (default), `onnx`
- Capabilities: `pose-estimation` through `PoseEstimation`
- Platform: macOS; accelerated runtimes target Apple Silicon
- Optional dependency set: `yolo`

## Prepare resources

Download the source for the selected size. Core ML and ONNX can be created
through the registered converter.

## Example

```python
import asyncio

from hugging_mac_sdk import ImageInput, ModelSdk, PoseRequest
from hugging_mac_sdk.capabilities import PoseEstimation
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.yolov8_pose import register_yolov8_pose


async def main() -> None:
    sdk = ModelSdk()
    converters = ConverterRegistry()
    register_yolov8_pose(sdk.registry, converters)
    model_id = "ultralytics/yolov8-pose"

    await sdk.resources.download_source(model_id, variant="n")
    async with await sdk.load(
        model_id,
        variant="n",
        runtime="pytorch-mps",
    ) as handle:
        estimator = handle.require(PoseEstimation)
        result = await estimator.estimate_pose(
            PoseRequest(image=ImageInput(path="person.jpg"), confidence=0.4)
        )
        for pose in result.poses:
            print(pose.box, pose.keypoints)


asyncio.run(main())
```

## Notes and limits

The response contains person boxes and COCO body keypoints. Face and hand detail
requires separate face or hand models.

## License

The model is licensed under `AGPL-3.0`. Commercial distribution may require a
separate Ultralytics license.
