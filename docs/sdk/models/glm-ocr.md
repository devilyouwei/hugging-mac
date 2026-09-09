# GLM-OCR

## Purpose

`zai-org/glm-ocr` extracts multilingual text, formulas, tables, and document
layout from page images using the MLX Community eight-bit checkpoint.

## Support

- Variant: `8bit` (default)
- Runtime: `mlx` (default)
- Capability: `document-parsing`
- Device: Apple Silicon GPU through MLX

## Prepare resources

Register the model and download the approximately 1.59 GB checkpoint before
loading it.

## Example

```python
import asyncio
from pathlib import Path

from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.capabilities import DocumentParsing
from hugging_mac_sdk.models.glm_ocr import register_glm_ocr
from hugging_mac_sdk.schemas.document_parsing import DocumentImage, DocumentParsingRequest


async def main() -> None:
    sdk = ModelSdk()
    register_glm_ocr(sdk.registry)
    await sdk.resources.download_source("zai-org/glm-ocr", variant="8bit")
    handle = await sdk.load("zai-org/glm-ocr", variant="8bit", runtime="mlx")
    parser = handle.require(DocumentParsing)
    response = await parser.parse_document(
        DocumentParsingRequest(images=(DocumentImage(path=Path("document.png")),))
    )
    print(response.text)
    await handle.close()


asyncio.run(main())
```

## Notes and limits

The runtime maps public document parsing requests to GLM-OCR's supported
`Text Recognition:` instruction. Multiple pages are processed sequentially,
and both buffered and streaming inference are serialized per model instance.

## License

The upstream checkpoint is licensed under `MIT`.
