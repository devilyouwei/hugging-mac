# Laya Multilingual

## Purpose

`convaiinnovations/laya` makes choice, score and noul decisions over text or JSON. The SDK preserves
`action.act_probability` (probability of answering directly), rather than
silently dropping the act head.

## Support

Capability: `decision-making`. Defaults: `multilingual-legacy-s512` (default) and `coreai` (default).

| Variant | Runtime | Source and context |
| --- | --- | --- |
| `multilingual` | `pytorch` | [Current original](https://huggingface.co/convaiinnovations/laya-multilingual); default 1024, configurable up to 8192 |
| `multilingual-legacy-s256` | `coreai` | [Third-party export](https://huggingface.co/mlboydaisuke/Laya-Multilingual-CoreAI); fixed 256 tokens |
| `multilingual-legacy-s512` | `coreai` | Same export; fixed 512 tokens, SDK default |

The exported checkpoint is the older `convaiinnovations/laya/multilingual`,
checkpoint `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, exported with Laya 0.3.4.
It is not claimed equivalent to the current independent multilingual checkpoint.
WFP16 means FP16 weight storage and FP32 computation. This integration exposes CPU
Core AI on Apple Silicon macOS, not the repository's compiled iOS assets.

## Prepare resources

Install `hugging-mac-sdk[laya-coreai]` for Core AI or `[laya]` for PyTorch.
Loading never downloads; prepare resources explicitly first:

## Example

```python
import asyncio
from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.models.laya import register_laya
from hugging_mac_sdk.schemas.decision import DecisionRequest

async def main() -> None:
    sdk = ModelSdk()
    register_laya(sdk.registry)
    variant = "multilingual-legacy-s512"
    await sdk.resources.download_source(
        "convaiinnovations/laya", variant=variant, options={"runtime": "coreai"}
    )
    async with await sdk.acquire(
        "convaiinnovations/laya", variant=variant, runtime="coreai", device="cpu"
    ) as handle:
        response = await handle.require(DecisionMaking).decide(DecisionRequest(
            state={"body": "我被重复扣款，请退款。"},
            questions={"department": {
                "type": "choice", "instructions": "Which team should handle this request?",
                "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"},
            }},
        ))
        print(response.answers)

asyncio.run(main())
```

For the original model, select `variant="multilingual", runtime="pytorch"` for
both download and acquisition. Set `options={"max_length": 8192}` on acquisition
for its full context. The Core AI window cannot be enlarged by that option.

## Notes and limits

Core AI uses the publisher's sequence layout (head budget 256, option descriptions
up to 48 tokens), original source temperature 1 and both main/act functions.
The optional third-party fitted calibration is not enabled. State overflow is
reported as an error rather than silently cut off. PyTorch rejects answers whose upstream usage reports truncated state; its probabilities are normalized for
the typed SDK, and score is recomputed from those probabilities. Act probability,
confidence and answer confidence are retained. These scores do not establish
calibrated task accuracy. No local conversion is registered.

## License

`Apache-2.0`; upstream and conversion attribution is retained in the SDK package NOTICE.
