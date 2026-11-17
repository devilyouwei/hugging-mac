# Vela 3.0

## Purpose

`vllm-sr/vela` makes choice and yes/no (`noul`) decisions from text or JSON state
using the Vela3-fast decision backbone and the same capability interface as D3.

## Support

- Variant: `3-fast` (default)
- Runtimes: `pytorch` (default), `coreai`
- Capability: `decision-making`
- Devices: PyTorch CPU; Core AI CPU or automatic placement on Apple Silicon macOS
- Sources: [original weights](https://huggingface.co/vllm-sr/vela3-fast) and
  [experimental Core AI export](https://huggingface.co/hugging-mac/vela3-fast-CoreAI)

## Prepare resources

Install the SDK `vela` extra for PyTorch or `vela-coreai` for native inference.
Register Vela before resource preparation. Download original weights with
`sdk.resources.download_source("vllm-sr/vela")`. Download the Core AI bundle with
`sdk.resources.download_source("vllm-sr/vela", options={"runtime": "coreai"})`.
Both runtimes share one managed tokenizer.

For local conversion, first prepare the original source, then call
`sdk.resources.convert("vllm-sr/vela", ArtifactFormat.COREAI)` in an environment
with `coreai-torch`, PyTorch >=2.8 and the Vela source dependencies. Conversion
supports `overwrite=True`. Pass the same `model_home` in options during
preparation and inference. Complete external checkpoints can be supplied with
`source_path` (PyTorch) or `artifact_path` (Core AI).

## Example

```python
import asyncio

from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.models.vela import register_vela
from hugging_mac_sdk.schemas.decision import DecisionRequest


async def main() -> None:
    sdk = ModelSdk()
    register_vela(sdk.registry)
    await sdk.resources.download_source("vllm-sr/vela", options={"runtime": "coreai"})
    async with await sdk.acquire("vllm-sr/vela", runtime="coreai", device="cpu") as handle:
        result = await handle.require(DecisionMaking).decide(
            DecisionRequest(
                state="The customer was charged twice.",
                questions={
                    "route": {
                        "type": "choice",
                        "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Payments", "shipping": "Delivery"},
                    },
                    "refund": {"type": "noul", "instructions": "A refund is needed."},
                },
            )
        )
        print(result.answers)


asyncio.run(main())
```

## Notes and limits

Both runtimes expose the decision-only backbone. Span, NER, relation, evidence,
trained router adapters, images, and videos are not supported. Router-like choice
questions use the decision backbone. Input length is dynamic up to 32,768 tokens;
oversized requests fail without truncation. Requests are serialized per instance,
and cancellation waits for native work to finish before releasing resources.
Core AI inference uses NumPy/tokenizers on the host and needs no PyTorch weights.

## Decision contract

The shared capability exposes `decision_support()` so callers can inspect limits
before sending requests. Answers are typed: inspect `answer.type`, then read
`answer.choice`, `answer.noul`, or `answer.error`. Token counts use
`response.usage.input_tokens`. A malformed or unsupported question returns a
`DecisionError` without discarding other valid questions in the request.
Vela supports `choice` and `noul` with text/JSON state. `score`, media, and custom
false/true descriptions return explicit unsupported-feature errors.

## License

The upstream model is licensed under `Apache-2.0`.

