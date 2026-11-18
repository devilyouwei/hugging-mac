# Full Vela Core AI

[中文](README.zh-CN.md)

This bundle retains the original Vela decision backbone and every task adapter. Its source repository, pinned revision, exact tensor inventory and graph interfaces are in `manifest.json`. Check `build.json` and `evaluation.json` for validation status; exporting a graph alone is not a successful model evaluation.

The original `system_one(state, questions, model=None)` and `predict(state, questions)` input/output structures are preserved. This includes text/JSON state, multiple questions, choice/noul, built-in router choice/set, generic spans, NER, relations, PII, unsupported claims, evidence, answer probabilities, thresholds, character offsets, scores, errors and usage. Original rendering, routing, thresholds, tokenization, window aggregation and decoding are reused.

All neural operations execute in Core AI. Original FP32 tensors are stored once under `weights/` and cached as immutable Core AI weight inputs. Graphs are shared by compatible adapter architectures. LoRA remains its original two projections; weights are neither merged nor quantized. The runtime keeps one component resident at a time, preserving public behavior while reducing memory use. PyTorch is imported by upstream host code but is not used for neural inference in this runtime.

Batch, sequence, question, option, word-unit and label dimensions are dynamic. Original defaults remain: a 32768-token decision limit, 8192-token schema windows with 512-token overlap, and 32768-token evidence windows. This does not imply the maximum length has been stress-tested on every device; see the measured lengths in the evaluation report. Memory exhaustion remains possible at large sizes.

## Python SDK

Install the current SDK with the `vela` and `coreai` extras in its own environment.
The independent conversion environment does not install or import the SDK.

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
        "vllm-sr/vela",
        variant="3_1-fast-full",
        runtime="coreai",
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

Use `method="predict"` for the original predict result. `options={"entity_route": "span"}`
selects the original relation routing policy. SDK calls are asynchronous and serialized
per instance. CPU is the validation target; GPU/ANE and iOS execution are unverified.

For independent native bundle validation, run:

```sh
USE_OS_COREAI=1 uv run --project packages/hugging_mac_conversion --locked \
  python -m hugging_mac_conversion.models.vela.evaluation.bundle_validation --artifact-path /path/to/bundle --check
```
