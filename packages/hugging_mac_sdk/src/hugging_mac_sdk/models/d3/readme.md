# Decision 3.0

Canonical model ID: `vllm-sr/d3`. The `edge` variant implements decision making
for text, images, and videos with the original `pytorch` runtime (default) and
the experimental `coreai` runtime. PyTorch currently exposes CPU execution and
retains the upstream BF16 backbone / FP32 readout policy.

## Ownership

- `model.yaml` owns identity, runtime requirements, variants, and the complete
  scoped and shared file inventories. `definition.py` injects these declarations into the
  engine and resource provider. `extensions.d3_coreai` owns the native asset
  layout; variant metadata owns the upstream response model name.
- `config.py` defines frozen instance options (`model_home`, optional
  `artifact_path`, `source_path`, Hub token, device policy, and permutation averaging).
- `instance.py` owns lifecycle, capability registration, serialization, error
  mapping, and `DecisionRequest` / `DecisionResponse` mapping.
- `torch.py` loads the original upstream runtime and weights without export or
  quantization. Both engines implement the same runtime-neutral instance protocol.
- `converter.py` exports both neural stages and publishes the scoped Core AI artifact
  only after its declared inventory passes validation. Cancellation drains the
  worker and discards staged files; overwrite preserves the old bundle on failure.
- `coreai.py` owns decision and vision sessions. Both neural stages execute in
  Core AI. Torch, Transformers, and the upstream processor perform CPU host
  preprocessing and response formatting; they do not execute neural weights.
- `utils/adapter.py`, `utils/preprocess.py`, and `utils/upstream.py` contain private
  host processing and isolated loading of the bundle's upstream Python code.
- `utils/coreai_export.py`, `utils/coreai_attention.py`, and `utils/coreai_vision.py`
  own export graphs. The conversion package owns independent copies and does not import SDK internals.
- `utils/execution.py` drains blocking work before propagating cancellation.
- `utils/types.py` validates the immutable native asset layout.
- `resources.py` validates local bundles, reports status, and deletes only the
  canonical managed artifact. External `artifact_path` bundles cannot be deleted.
- The conversion package owns native bundle evaluation independently. Public SDK
  acceptance is in `tests/sdk/d3_bundle_validation.py`; application examples live in docs.

## Capability flow

Register `register_d3(sdk.registry)`, then acquire `vllm-sr/d3` through `ModelSdk`
with `runtime="pytorch"` after `sdk.resources.download_source(...)`. A local original
checkpoint can be supplied with `options={"source_path": source}`. For converted
bundles select `runtime="coreai"` and optionally `options={"artifact_path": bundle}`. Invoke
`handle.require(DecisionMaking).decide(DecisionRequest(...))`. Images use shared
`ImageInput` paths or encoded bytes; videos use local paths or
`DecisionVideoFrames` containing encoded/path-based pre-sampled frames. Questions and answers
retain the upstream choice/noul/score vocabulary. Responses add SDK identity,
instance, runtime, device policy, and separate preprocessing, inference, and response-mapping timings.

Bundle acceptance calls the same typed capability as applications. The conversion
package translates NumPy video fixtures into lossless encoded frames; the SDK
public boundary contains no framework arrays or validation-only instance methods. Optional frameworks are imported only when loading the runtime.

## Shared resources

Both scoped artifacts declare `required_shares: [tokenizer, processor, runtime]`.
Each share has its own downloadable source and lives under
`<model_home>/vllm-sr/d3/_shared/<artifact_id>`:

- `tokenizer`: vocabulary, tokenizer settings, merges, and chat template.
- `processor`: image and video preprocessing settings.
- `runtime`: the upstream Python host contract (`d3_runtime.py`, `d3_format.py`).
  Its `pytorch` artifact format describes the original Python model code; it does
  not declare a separate execution runtime.

Weights, model/decision configuration, and PyTorch-only fast-path code remain
scoped. The shared declarations contain no variant, runtime, conversion flag,
or dependencies. Future incompatible variants must declare different shared
artifacts instead of overwriting these files.

Core resource services select and download the required shares, aggregate their
availability and count their size once. Model resolvers validate all selected
shares before loading. Engines use `merged_directory_view()` and retain its
context until unload so upstream lazy path access remains valid. Conversion
keeps the view open until its worker finishes, including cancellation.
Deleting a scoped runtime leaves shares intact for other consumers.

Explicit external complete bundles can supply embedded shared files without
being copied or modified. Managed artifacts always use canonical shared paths;
legacy duplicate files do not satisfy a missing managed share.

## Resources and limits

Managed original weights resolve to
`<model_home>/vllm-sr/d3/edge/pytorch/source`; converted bundles resolve to
`<model_home>/vllm-sr/d3/edge/coreai/coreai-fp32`. Load validates local files and
never downloads or converts. Explicit source download supports overwrite and
uses the YAML-declared upstream source.

Call `sdk.resources.convert("vllm-sr/d3", ArtifactFormat.COREAI, ...)` for SDK
conversion. `register_d3(models, converters)` also registers the same converter
with `ConverterRegistry`. SDK conversion requires its own authoring dependencies
with compatible Torch, Transformers, and `coreai-torch`, as with Audio8-TTS.
It exports dynamic decision and vision graphs and copies only variant-owned
configuration files. Shared files are not copied into the converted artifact.
The Core AI artifact is still experimental: successful export is not numerical
parity approval. There is no compatible Core AI download source.

Status covers both runtimes without exposing paths. Delete can select one runtime
or remove both managed artifacts; externally supplied paths are protected. Native
`main.hash` files are asset contents, not configured expected digests.

Inference is serialized per instance. Cancellation waits for native execution to
finish before releasing the lock and propagates `CancelledError`. Unload waits
for inference; failed loads close both partially created sessions. Repeated
load/unload follows the SDK base lifecycle (an unloaded instance is terminal).
`cpu` selects CPU execution; `auto` reports the framework's automatic device
policy, not a claimed hardware placement.

The current conversion evaluation still has a confidence-parity failure. Offline
contract tests do not establish native numerical parity or publication readiness.
Run `python tests/sdk/d3_bundle_validation.py --artifact-path <bundle>`
for the explicit native bundle check. See the conversion package's D3 evaluation
report for evidence and limits.

Decision schemas and result validation are shared with other decision models in
`schemas/decision.py` and `core/decision.py`. The instance exposes
`decision_support()` using injected `extensions.decision` declarations, retains
its lifecycle/inference lock, and delegates per-question dispatch to the shared
helper. Runtime engines keep their native dictionary boundary; model-specific
preprocessing remains inside this package. Typed responses use attribute access,
and unsupported questions return `DecisionError` alongside valid answers.
