# Vela model package

`register_vela` exposes canonical `vllm-sr/vela`, variant `3-fast`, with
`pytorch` and `coreai` runtimes. Both implement the shared `DecisionMaking`
capability (`DecisionRequest` / `DecisionResponse`), like D3.

## Scope

This integration uses the original `vllm-sr/vela3-fast` decision backbone and
`hugging-mac/vela3-fast-CoreAI` decision-only export. It supports choice/noul,
structured JSON state, multiple questions, and dynamic sequence/option counts.
Inputs exceeding 32,768 tokens fail without truncation. Images, videos, spans,
sets, and trained router/task adapters are not part of this integration.
Choice questions always use the decision backbone, even for router-like text.

## Ownership

- `model.yaml`: identity, runtime requirements, artifacts, shared tokenizer and export layout.
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

`sdk.resources.convert(model_id, ArtifactFormat.COREAI)` converts a prepared
source into the same scoped layout as the downloadable Core AI artifact.
Run conversion in an environment with `coreai-torch`, PyTorch >=2.8 and the
Vela source dependencies. Existing SDK workspace Torch constraints may require
the repository's isolated conversion environment. Unknown options are rejected;
failed or cancelled conversion preserves existing output. Explicit overwrite
is supported. The tokenizer remains a declared shared dependency.

Offline contracts are in `tests/sdk/test_vela*.py`. Explicit native parity uses
`VELA_SOURCE_PATH` and `VELA_COREAI_PATH` with `test_vela_smoke.py -m smoke`.
The Core AI upstream release remains experimental and decision-only.

Decision schemas and result validation are shared with other decision models in
`schemas/decision.py` and `core/decision.py`. The instance exposes
`decision_support()` using injected `extensions.decision` declarations, retains
its lifecycle/inference lock, and delegates per-question dispatch to the shared
helper. Runtime engines keep their native dictionary boundary; model-specific
preprocessing remains inside this package. Typed responses use attribute access,
and unsupported questions return `DecisionError` alongside valid answers.
