# Hugging Mac Python SDK

`hugging_mac_sdk` provides one lifecycle and capability interface across local
model runtimes. Resource preparation is explicit: loading or inference never
downloads or converts model files.

## Install

In this repository:

```bash
uv sync --all-packages
```

When consuming only the SDK package, install the optional extras required by the
models you use, such as `yolo`, `asr`, `tts`, `vad`, or `llm`.

## Basic workflow

The SDK is asynchronous. A typical application registers models, prepares files,
loads a handle, obtains a capability, and closes the handle when finished.

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
            DetectionRequest(image=ImageInput(path="example.jpg"))
        )
        print(result.detections)


asyncio.run(main())
```

## Discover registered models

`ModelSdk()` starts with an empty registry. Register only the models your program
needs, then inspect the catalog:

```python
snapshot = await sdk.catalog.snapshot()
for model in snapshot.models:
    print(model.model_id, model.capabilities, model.variants, model.runtimes)
```

## Manage resources

```python
from hugging_mac_sdk import ArtifactFormat

status = await sdk.resources.status(model_id, variant="n")
status = await sdk.resources.download_source(
    model_id,
    variant="n",
    overwrite=True,
)
status = await sdk.resources.convert(
    model_id,
    ArtifactFormat.COREML,
    variant="n",
    overwrite=True,
)
status = await sdk.resources.delete(
    model_id,
    variant="n",
    runtime="coreml",
)
```

Conversion is available only for model/runtime combinations that declare it.
Prepare the source before converting.

Instance-level options can override managed storage or provide credentials:

```python
options = {
    "model_home": "/Volumes/Models/hugging-mac",
    "hf_token": "...",
}
```

Never commit tokens or write them to logs.

## Select a variant, runtime, and device

- Omit `variant` to use the model default.
- Use `runtime="auto"` to select a compatible installed runtime.
- Pass a runtime name to require a specific backend.
- Use `device` only with a device value supported by that runtime.

```python
handle = await sdk.load(
    model_id,
    variant="n",
    runtime="coreml",
    device="cpu-and-neural-engine",
)
```

## Reuse and release instances

Shared handles reuse an equivalent loaded instance. Dedicated handles unload
their instance when the final handle closes.

```python
from hugging_mac_sdk import ReusePolicy

async with await sdk.load(
    model_id,
    reuse=ReusePolicy.DEDICATED,
) as handle:
    print(handle.info())
```

## Common inputs

- Audio: `AudioInput(path=...)` or `AudioInput(data=...)`
- Images: `ImageInput(path=...)` or `ImageInput(data=...)`
- Chat images: `ChatImage(path=...)` or `ChatImage(url=...)`

## Handle errors

Stable SDK errors inherit from `HuggingMacSdkError`.

```python
from hugging_mac_sdk import HuggingMacSdkError

try:
    async with await sdk.load(model_id) as handle:
        print(handle.info())
except HuggingMacSdkError as error:
    print(error.code, error.details)
```

See the [model reference](models/README.md) for registration, resource, and
capability examples for every integrated model.
