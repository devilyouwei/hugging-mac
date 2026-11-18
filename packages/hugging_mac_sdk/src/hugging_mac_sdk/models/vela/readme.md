# Vela model package

`register_vela` exposes canonical `vllm-sr/vela`, variants `3-fast` (default) and `3_1-fast`, with
`pytorch` and `coreai` runtimes. Both implement the shared `DecisionMaking`
capability (`DecisionRequest` / `DecisionResponse`), like D3.

## Scope

This integration uses the original Vela 3.0/3.1 decision backbones. The 3.0
export is `hugging-mac/vela3-fast-CoreAI`; the 3.1 download selects only the
decision graph and main FP32 weight file from `hugging-mac/vela3.1-fast`. It supports choice/noul,
structured JSON state, multiple questions, and dynamic sequence/option counts.
Inputs exceeding 32,768 tokens fail without truncation. Images, videos, spans,
sets, and trained router/task adapters are not part of this integration.
Choice questions always use the decision backbone, even for router-like text.

## Ownership

- `model.yaml`: identity, runtime requirements, artifacts, per-variant tokenizer shares and export layout.
- `definition.py`: factories, resource provider, converter registration.
- `config.py`: local storage overrides, credentials and device selection.
- `instance.py`: lifecycle, serialized capability execution and measured stage timings.
- `resources.py`: managed status/download/conversion/delete; shared files use merged views.
- `torch.py`: isolated upstream DecisionEncoder loading and CPU inference.
- `coreai.py`: native Core AI session; NumPy/tokenizers host processing, no Torch.
- `converter.py`: transactional SDK export and required-file validation.
- `utils/`: request serialization, rendering, upstream isolation and dynamic export graph.

Registration imports no optional frameworks or model weights. Loading never
downloads or converts. Managed tokenizer files live once under `_shared`;
complete external bundles may instead provide their own tokenizer. Deletion
rejects external paths. Inference is serialized per instance; cancellation drains
native workers before unload. Failed loads close partial sessions and views.

## Conversion and verification

For `3-fast`, `sdk.resources.convert(model_id, ArtifactFormat.COREAI)` converts a prepared
source into the same scoped layout as the downloadable Core AI artifact.
Run conversion in an environment with `coreai-torch`, PyTorch >=2.8 and the
Vela source dependencies. Existing SDK workspace Torch constraints may require
a separate SDK authoring environment. Unknown options are rejected;
failed or cancelled conversion preserves existing output. Explicit overwrite
is supported. The tokenizer remains a declared shared dependency.

Offline contracts are in `tests/sdk/test_vela*.py`. Explicit native parity uses
`VELA_SOURCE_PATH` and `VELA_COREAI_PATH` with `test_vela_smoke.py -m smoke`.
Set `VELA_VARIANT=3_1-fast` for the new variant (default: `3-fast`).
The Core AI upstream release remains experimental and decision-only.

Decision schemas and result validation are shared with other decision models in
`schemas/decision.py` and `core/decision.py`. The instance exposes
`decision_support()` using injected `extensions.decision` declarations, retains
its lifecycle/inference lock, and delegates per-question dispatch to the shared
helper. Runtime engines keep their native dictionary boundary; model-specific
preprocessing remains inside this package. Typed responses use attribute access,
and unsupported questions return `DecisionError` alongside valid answers.

## Vela3.1-fast

Use `variant="3_1-fast"` with either runtime. Both downloads exclude all nine
adapters. Core AI reuses `utils/native.py` for immutable external weight inputs,
while `utils/packing.py` renders only choice/noul questions. No task routing or
Torch inference is introduced. Custom true/false descriptions for noul are
preserved through `extensions.decision_variants`. Separate tokenizer shares
prevent cross-version mixing. The Core AI prebuilt selection is approximately
1.52 GB plus tokenizer/configuration; local SDK conversion remains 3.0-only.

See [SDK usage](../../../../../../docs/sdk/models/vela.md) for download and acquisition.

## Complete structured tasks

`3_1-fast-full` is the Core AI edition with every original task adapter. Acquire it
through `ModelSdk` and call `StructuredPrediction.predict_structure` with a
`StructuredPredictionRequest`. The `result` contains the original response without
projecting spans, scores, errors or usage into decision-only answers. `method` selects
`system_one` or `predict`; `entity_route` is an instance option. The decision-only
editions continue to download only the backbone. Full-model inference shares the
instance inference lock and drained cancellation with `DecisionMaking`.

`coreai.py` owns runtime integration; `utils/full.py`, `utils/full_host.py` and
`utils/native.py` are private. Conversion maintains its own implementations and
comparison tools. No external caller imports these SDK utilities.
