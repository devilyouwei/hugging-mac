# Full Vela Core AI

[中文](README.zh-CN.md)

This bundle retains the original Vela decision backbone and every task adapter. Its source repository, pinned revision, exact tensor inventory and graph interfaces are in `manifest.json`. Check `build.json` and `evaluation.json` for validation status; exporting a graph alone is not a successful model evaluation.

The original `system_one(state, questions, model=None)` and `predict(state, questions)` input/output structures are preserved. This includes text/JSON state, multiple questions, choice/noul, built-in router choice/set, generic spans, NER, relations, PII, unsupported claims, evidence, answer probabilities, thresholds, character offsets, scores, errors and usage. Original rendering, routing, thresholds, tokenization, window aggregation and decoding are reused.

All neural operations execute in Core AI. Original FP32 tensors are stored once under `weights/` and cached as immutable Core AI weight inputs. Graphs are shared by compatible adapter architectures. LoRA remains its original two projections; weights are neither merged nor quantized. The runtime keeps one component resident at a time, preserving public behavior while reducing memory use. PyTorch is imported by upstream host code but is not used for neural inference in this runtime.

Batch, sequence, question, option, word-unit and label dimensions are dynamic. Original defaults remain: a 32768-token decision limit, 8192-token schema windows with 512-token overlap, and 32768-token evidence windows. This does not imply the maximum length has been stress-tested on every device; see the measured lengths in the evaluation report. Memory exhaustion remains possible at large sizes.

## Python SDK

From the Hugging Mac repository, use its locked conversion environment (which includes the local SDK):

```sh
USE_OS_COREAI=1 uv run --project packages/hugging_mac_conversion --locked \
  python -m hugging_mac_conversion.models.vela.evaluation.sdk_validation --artifact-path /path/to/bundle --check
```

```python
from hugging_mac_sdk.models.vela.full import VelaFullCoreAI

with VelaFullCoreAI('/path/to/bundle', device='cpu') as model:
    result = model.system_one(
        'Contact Alice at alice@example.com.',
        {'pii': {'type': 'span',
                 'criteria': {'EMAIL_ADDRESS': 'an e-mail address'}}},
    )
    print(result)
```

`entity_route='span'` retains the upstream relation-pipeline option. The API is synchronous; async applications should call it from a worker thread. CPU is the validation target. Core AI GPU/ANE, Swift preprocessing and iOS execution are not claimed as tested.

Install the SDK’s `vela` and `coreai` extras for this full loader. Use the current source SDK or a wheel built from it; this new loader is not assumed to exist in an older published wheel. No `requirements.txt` is needed: dependencies and locking belong to the project's `uv` packages. `weights/` and all graph directories are required; do not download only the decision graph.
