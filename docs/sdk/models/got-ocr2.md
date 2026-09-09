# GOT-OCR2.0

## Purpose

`stepfun-ai/got-ocr2.0` extracts formatted text, tables, formulas, and other
document content from page images.

## Support

- Variant: `8bit` (default)
- Runtime: `mlx` (default)
- Capability: `document-parsing`
- Device: Apple Silicon GPU through MLX

## Prepare resources

Register the model and download the MLX Community checkpoint before loading it.
The eight-bit download is about 689 MB.

## Example

```python
import asyncio
from pathlib import Path

from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.capabilities import DocumentParsing
from hugging_mac_sdk.models.got_ocr2 import register_got_ocr2
from hugging_mac_sdk.schemas.document_parsing import DocumentImage, DocumentParsingRequest


async def main() -> None:
    sdk = ModelSdk()
    register_got_ocr2(sdk.registry)
    await sdk.resources.download_source("stepfun-ai/got-ocr2.0", variant="8bit")
    handle = await sdk.load("stepfun-ai/got-ocr2.0", variant="8bit", runtime="mlx")
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

This integration requires Apple Silicon and `mlx-vlm` 0.6.14 or newer. The
runtime uses GOT-OCR2.0's native formatted OCR instruction because the model is
not trained as a general-purpose chat model. Multiple input pages are processed
sequentially and receive Markdown page provenance markers. Both buffered and
incremental parsing are serialized on each loaded model instance.

## License

The upstream checkpoint is licensed under `Apache-2.0`.
