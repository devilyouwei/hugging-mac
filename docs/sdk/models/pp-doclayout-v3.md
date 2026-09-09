# PP-DocLayoutV3

## Purpose

PP-DocLayoutV3 classifies document regions such as text, titles, tables,
formulas, and images, and returns their geometry in reading order.

## Support

- Model ID: `paddlepaddle/pp-doclayout-v3`
- Variants: `base` (default)
- Runtimes: `pytorch-mps`, `coreml` (default)
- Capabilities: `document-layout-analysis` through `DocumentLayoutAnalysis`
- Platform: macOS on Apple Silicon
- Optional dependency set: `layout`

## Prepare resources

Download the safetensors source before using PyTorch or creating the local Core
ML package. The Models page provides both actions. Core ML conversion keeps the
model's fixed 800 by 800 input and publishes raw prediction tensors; the SDK
then restores geometry and reading order consistently for both runtimes.

## Example

```python
import asyncio

from hugging_mac_sdk import ArtifactFormat, DocumentLayoutRequest, ImageInput, ModelSdk
from hugging_mac_sdk.capabilities import DocumentLayoutAnalysis
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.pp_doclayout_v3 import register_pp_doclayout_v3


async def main() -> None:
    sdk = ModelSdk()
    converters = ConverterRegistry()
    register_pp_doclayout_v3(sdk.registry, converters)
    model_id = "paddlepaddle/pp-doclayout-v3"

    await sdk.resources.download_source(model_id, variant="base")
    await sdk.resources.convert(model_id, ArtifactFormat.COREML, variant="base")
    async with await sdk.load(model_id, runtime="coreml") as handle:
        analyzer = handle.require(DocumentLayoutAnalysis)
        result = await analyzer.analyze_layout(
            DocumentLayoutRequest(image=ImageInput(path="document-page.png"))
        )
        for region in result.regions:
            print(region.order, region.label, region.box)


asyncio.run(main())
```

## Notes and limits

The network analyzes page layout; it does not transcribe text inside the
regions. Images are resized to the checkpoint's fixed square input, while all
returned boxes and polygons use source-image pixel coordinates.

## License

The model is licensed under `Apache-2.0`.
