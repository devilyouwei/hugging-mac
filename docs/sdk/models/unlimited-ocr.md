# Unlimited-OCR

## Purpose

`baidu/unlimited-ocr` parses text, layout, tables, formulas, and other document
content from one image or an ordered set of page images.

## Support

- Variant: `4bit` (default)
- Runtime: `mlx` (default)
- Capability: `document-parsing`
- Device: Apple Silicon GPU through MLX

## Prepare resources

Register the model and download the MLX Community checkpoint before loading it.
The four-bit download is about 2.45 GB.

## Example

```python
import asyncio
from pathlib import Path

from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.capabilities import DocumentParsing
from hugging_mac_sdk.models.unlimited_ocr import register_unlimited_ocr
from hugging_mac_sdk.schemas.document_parsing import DocumentImage, DocumentParsingRequest


async def main() -> None:
    sdk = ModelSdk()
    register_unlimited_ocr(sdk.registry)
    await sdk.resources.download_source("baidu/unlimited-ocr", variant="4bit")
    handle = await sdk.load("baidu/unlimited-ocr", variant="4bit", runtime="mlx")
    parser = handle.require(DocumentParsing)
    response = await parser.parse_document(
        DocumentParsingRequest(images=(DocumentImage(path=Path("document.png")),))
    )
    print(response.text)

    async for event in parser.stream_document(
        DocumentParsingRequest(images=(DocumentImage(path=Path("document.png")),))
    ):
        print(event.delta, end="", flush=True)
    await handle.close()


asyncio.run(main())
```

## Notes and limits

This integration requires Apple Silicon and `mlx-vlm` 0.6.8 or newer. The model
integration automatically uses the upstream `gundam` cropped configuration for
single images and the uncropped `base` configuration for multi-image requests.
It also applies the upstream sliding-window repetition guard. These correctness
settings are internal and cannot be overridden by callers. Both buffered and
incremental parsing use the same serialized model instance. Four-bit quantization
reduces memory and disk usage, with the usual quantization quality tradeoff.

## License

The upstream checkpoint is licensed under `MIT`.
