# Vela conversion

[中文](readme.zh-CN.md)

One model family owns its variants. `model.yaml` declares `3-fast` (vllm-sr/vela3-fast)
and `3_1-fast` (vllm-sr/vela3.1-fast). The underscore is the local variant identifier;
the upstream repository retains its dot.

- `converter.py`: the existing dynamic decision-graph diagnostic, used by the shared `probe` command.
- `components.py`: complete decision/schema graphs, original LoRA operations, dynamic axes and immutable external FP32 weight inputs.
- `full.py`: verifies a complete immutable source, exports/reuses compatible component graphs, accounts for every checkpoint tensor, and assembles a standalone full bundle.
- `evaluation/full_cases.py`, `evaluation/full.py`: complete responses and all component tensor comparisons against original upstream execution.
- `publication/full/`: English/Chinese full-bundle documentation; existing publication files retain the older decision-only release recipe.

The SDK owns `models.vela.full.VelaFullCoreAI`, snapshot isolation, host routing and native
component lifecycle. The SDK acceptance CLI lives in `evaluation/sdk_validation.py`.
Conversion imports SDK helpers; the SDK does not import conversion code. Apple export/runtime
mechanics remain in shared SDK facilities.

Vela 3.1 includes the decision checkpoint plus nine adapter checkpoints. None are excluded. Exact FP32 tensors are stored once and cached as immutable Core AI inputs, avoiding repeated copies of the shared backbone. The model's original choice/noul, router choice/set, spans, NER, relations, PII, evidence, structured state, multi-question processing and response fields remain in the contract. Dynamic dimensions include batch, sequence, questions, options, units and labels. Original context/window defaults remain unchanged.

## Build and evaluate

From the repository root:

```sh
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert fetch vela --variant 3_1-fast --revision 0ed6ed301fbbf0c312f11a13ad17e3a6f8887262
USE_OS_COREAI=1 uv run --project packages/hugging_mac_conversion --locked python -m hugging_mac_conversion.models.vela.full --variant 3_1-fast --source tmp/originals/vela/3_1-fast/0ed6ed301fbbf0c312f11a13ad17e3a6f8887262
USE_OS_COREAI=1 uv run --project packages/hugging_mac_conversion --locked python -m hugging_mac_conversion.models.vela.evaluation.full --source <original> --bundle <build> --output <new-evaluation-directory>
USE_OS_COREAI=1 uv run --project packages/hugging_mac_conversion --locked python -m hugging_mac_conversion.models.vela.evaluation.sdk_validation --artifact-path <build> --check
```

`full --resume <build>` reuses unchanged graphs only when source/variant and architecture hashes match.
Originals, builds and machine-readable evaluations stay under project-local, Git-ignored
`tmp/originals/vela/<variant>`, `tmp/conversions/vela/<variant>/coreai`, and
`tmp/evaluations/vela/<variant>`. Original and converted weights may share immutable local
file storage through hardlinks; bundles contain no symlinks to external caches.

Complete-response validation compares every field, including probabilities, span offsets/scores,
thresholds, errors and usage. Every component must be exercised and raw tensor checks must pass.
Only then can the build enter the existing SDK/publication gates. Symbolic 32K support is not a
claim that 32K execution has been stress-tested. Failed or incomplete builds remain unpublishable.

[Latest Vela 3.1 results](evaluation/coreai-3_1-fast-2026-11-17.md).
[Earlier Vela 3.0 diagnostic results](evaluation/coreai-dynamic-2026-11-16.md).
