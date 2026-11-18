# Vela

## Purpose

`vllm-sr/vela` makes choice and yes/no (`noul`) decisions from text or JSON state
using the Vela3-fast or Vela3.1-fast decision backbone and the same capability interface as D3.

## Support

- Variants: `3-fast` (default), `3_1-fast` (Vela3.1-fast decision backbone), `3_1-fast-full` (all adapters, Core AI only)
- Runtimes: `pytorch` (default), `coreai`
- Capabilities: `decision-making`, `structured-prediction`
- Devices: PyTorch CPU; Core AI CPU or automatic placement on Apple Silicon macOS
- Sources: [original weights](https://huggingface.co/vllm-sr/vela3-fast) and
  [experimental Core AI export](https://huggingface.co/hugging-mac/vela3-fast-CoreAI)

## Prepare resources

Install the SDK `vela` extra for PyTorch or `vela-coreai` for native inference.
Register Vela before resource preparation. Download original weights with
`sdk.resources.download_source("vllm-sr/vela")`. Download the Core AI bundle with
`sdk.resources.download_source("vllm-sr/vela", options={"runtime": "coreai"})`.
Both runtimes share a managed tokenizer within each variant; 3.1 has its own tokenizer resource.

For `3-fast` local conversion, first prepare the original source, then call
`sdk.resources.convert("vllm-sr/vela", ArtifactFormat.COREAI)` in an environment
with `coreai-torch`, PyTorch >=2.8 and the Vela source dependencies. Conversion
supports `overwrite=True`. Pass the same `model_home` in options during
preparation and inference. Complete external checkpoints can be supplied with
`source_path` (PyTorch) or `artifact_path` (Core AI).

## Example

```python
import asyncio
from pathlib import Path
from hugging_mac_sdk import ModelSdk, StructuredPredictionRequest
from hugging_mac_sdk.capabilities import StructuredPrediction
from hugging_mac_sdk.models.vela import register_vela

async def main():
    sdk = ModelSdk()
    register_vela(sdk.registry)
    async with await sdk.acquire(
        "vllm-sr/vela", variant="3_1-fast-full", runtime="coreai",
        options={"artifact_path": Path("/path/to/bundle")},
    ) as handle:
        response = await handle.require(StructuredPrediction).predict_structure(
            StructuredPredictionRequest(
                state="Contact Alice at alice@example.com.",
                questions={"pii": {"type": "span", "criteria": {"EMAIL_ADDRESS": "email"}}},
            )
        )
        print(response.result)

asyncio.run(main())
```


## Notes and limits

The full edition supports original adapter tasks. Decision-only editions support choice/noul only.

## License

The upstream model is licensed under `Apache-2.0`.
