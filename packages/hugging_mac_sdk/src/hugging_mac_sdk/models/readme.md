# Model Package Integration Standard

This document is the required checklist for adding a model to `hugging_mac_sdk.models`.
It summarizes the SDK architecture and turns it into an implementation and review contract.
New model packages must follow this standard before they are exposed to applications.

## 1. Canonical model identity

One upstream model has one canonical Python package and one stable model ID.

- Name the package after the original model or model family, normalized to Python snake case.
- Use the upstream organization and canonical model name for `manifest.model_id`.
- Do not encode a runtime, framework, file format, precision, quantization, parameter size,
  conversion method, or deployment target in the package name or model ID.
- Names such as `_coreml`, `_mlx`, `_onnx`, `_int8`, `_4bit`, `_bf16`, or a size-only suffix
  such as `_8b` are prohibited when they merely describe another build of the same model.
- Put interchangeable weight scales or editions in `manifest.variants`.
- Put execution backends in `manifest.runtimes` and `runtime_factories`.
- Put formats, quantization, precision, and physical files in `artifacts` and variant metadata.
- Use a separate package/model ID only when the public capability contract or model architecture
  is genuinely different, not because its weights were converted or quantized.

For example, PyTorch BF16, MLX 4-bit, ONNX INT8, and Core ML builds of one upstream model belong
in the same directory and definition:

```text
models/example_model/
model_id: upstream/example-model
variants: [base, large]
runtimes: [pytorch-mps, mlx, onnx, coreml]
artifacts: source-bf16, mlx-4bit, onnx-int8, coreml-int8
```

## 2. Architecture boundary

Applications program against capability protocols, never model classes, runtime adapters, or
artifact names:

```text
Application -> ModelSdk/ModelHandle -> Capability -> ModelInstance -> Engine -> Runtime provider
```

The SDK package must be usable without FastAPI, Vue, application databases, or business queues.
Runtime-specific tensors, sessions, processors, cache state, and file paths never cross the public
capability schema boundary. Application code must not contain special cases for a model's artifact
IDs, tokenizer layout, quantization, or runtime implementation.

## 3. Required package layout

```text
models/<canonical_model>/
├── __init__.py
├── readme.md
├── model.yaml
├── config.py
├── definition.py
├── instance.py
├── resources.py          # required when the model owns managed files
├── torch.py              # only for a real PyTorch runtime
├── mlx.py                # only for a real MLX runtime
├── coreml.py             # only for a real Core ML runtime
├── onnx.py               # only for a real ONNX Runtime runtime
├── converter.py          # only for supported local conversions
└── utils/                # model-private preprocessing, postprocessing, loaders, and types
    └── __init__.py
```

Responsibilities are strict:

- `__init__.py` exports only the definition, manifest, and registration function.
- `readme.md` summarizes the model, canonical identity, supported variants and runtimes,
  capability APIs, implementation status, and the responsibility of every package file.
- `model.yaml` is the only source of truth for declarative identity, license, capabilities,
  variants, runtimes, sources, artifacts, download selections, and required files.
  YAML must not import or execute Python.
- `config.py` defines only frozen, strongly typed instance/resource/runtime input options such as
  storage overrides, credentials, device policy, and inference controls. It must not repeat model
  IDs, repository IDs, variant inventories, or artifact filename inventories.
- `definition.py` binds every declared runtime to a concrete factory and registers resources and
  converters. It is the package composition root: it loads `model.yaml`, selects artifacts by
  semantic key, and injects the parsed manifest/artifacts into instances, engines, and resource
  providers. Converter requests must be built from that same injected context.
- `instance.py` owns lifecycle, concurrency, capability registration, common pre/postprocessing,
  and composition with an engine protocol. It must not import concrete runtime engines.
- Runtime files load and execute one backend through the shared runtime providers where possible.
- `resources.py` resolves, validates, reports, downloads, and deletes only this model's assets. It
  consumes injected manifest/artifact objects and must not redeclare static model facts.
- `converter.py` contains model-specific conversion logic; general framework mechanics belong in
  `converters/`.
- `utils/` is private to the package. A model package must not import another model package.

## 4. Manifest, variants, runtimes, and artifacts

Declare the upstream repository, license, and trust boundary. Hugging Face sources intentionally
follow the repository's default `main` revision and are trusted without content-hash validation. Each
variant must describe a weight/model choice that preserves the same public task contract. Each
runtime must describe an implementation that actually exists on the instance factory path.

Every supported `(variant, runtime)` combination must have:

1. a concrete runtime factory in `runtime_factories`;
2. all artifacts required by that combination;
3. a resource status whose aggregated runtime availability is accurate;
4. a load/unload implementation and an inference contract test.

A runtime does not have to support every variant. The presence of a scoped artifact declares a
supported combination; a missing combination must be rejected during definition creation rather
than passed to the runtime factory or reported as ready.

An artifact is a physical input to a runtime. It may be a file or directory and may contain weight
shards, tokenizer files, processors, configuration, or compiled graphs. A runtime may require more
than one artifact. `source` makes an artifact downloadable; `convert: true` makes it a supported
local conversion target. Both may be declared on the same artifact when a prebuilt download and a
local conversion are alternative ways to provision exactly the same files at exactly the same
path. `provisioning` is derived resource-status output, not a `model.yaml` field. Artifact IDs are
SDK data and must not become model-specific application logic.

### `model.yaml` standard fields

The common template contains only fields that have the same meaning for every model package:

- `manifest`: canonical identity, presentation, license, capabilities, variants,
  runtimes, defaults, and upstream source links;
- `artifacts`: runtime/variant ownership, format, canonical managed path, kind, optional source,
  conversion support, sharing, required files, and required shares;
- `extensions`: an optional namespaced area for model-private declarative data that does not belong
  in the common schema.

A representative declaration is:

```yaml
manifest:
  schema_version: '1'
  model_id: upstream/example-model
  display_name: Example Model
  family: example
  capabilities: [embedding]
  runtimes:
  - name: coreml
    required_modules: [coremltools, numpy]
  variants:
  - name: base
    display_name: Base
  default_variant: base
  default_runtime: coreml

artifacts:
- artifact_id: coreml-fp16
  variant: base
  runtime: coreml
  format: coreml
  path: upstream/example-model/base/coreml/coreml-fp16
  kind: directory
  required_files: [Manifest.json, Data/model.mlmodel]
  required_shares: [tokenizer]
  convert: true
  source:
    kind: huggingface
    repo_id: upstream/example-model-coreml
    allow_patterns:
    - example.mlpackage/**
    strip_prefix: example.mlpackage

- artifact_id: tokenizer
  format: tokenizer
  path: upstream/example-model/_shared/tokenizer
  kind: directory
  shared: true
  required_files: [tokenizer.json, tokenizer_config.json]
  source:
    kind: huggingface
    repo_id: upstream/example-model
    allow_patterns:
    - tokenizer.json
    - tokenizer_config.json
```

Every directory artifact declares its complete `required_files` inventory as a YAML flow array,
for example `required_files: [config.json, model.safetensors]`. Every runtime declares
`required_modules` in the same array form. A Hugging Face repository snapshot declares the exact
download selection through a block-style `allow_patterns` list. A single-file source may use
`filename`, because it is downloaded through the Hub file API rather than as a snapshot.
File artifacts do not declare `required_files`; their canonical file path is the availability
check. Put `required_shares` beside `required_files` so the artifact's complete dependency contract
is readable in one place.

Within one package, `(variant, runtime, artifact_id)` must be unique. Every `required_shares` entry
must name a declared shared artifact. The package loader validates both rules and rejects YAML that
depends on ordering or an undeclared share.

Do not add `revision`, `expected_sha256`, `file_sha256`, or other digest fields. Resource
availability is based on the complete required-file inventory; downloaded bytes are trusted from
the declared upstream source. The SDK may still calculate a digest for result metadata, but it must
not compare that digest with a configured expected value.

Download sources have exactly one owner: `artifacts[*].source`. Neither `manifest` nor a variant
may declare `resources`. Definitions, resource providers, converters, and application inventory
must select the artifact semantically and consume its injected `source`; they must not maintain a
second variant-to-source mapping.

Shared tokenizers and other reusable downloads are unscoped `shared: true` artifacts with their own
source. A shared artifact must not declare `variant`, `runtime`, or another shared dependency.
Dependencies belong to the consuming scoped artifact through `required_shares: [tokenizer]`, not to
the variant. This forms an explicit many-to-many relation: one runtime artifact may require several
shares, and one share may be reused by artifacts from several variants or runtimes. Status, download,
conversion preparation, and application readiness checks resolve the shares of the exact artifact(s)
being operated on. Direct download/delete operations still address a shared artifact by its
`artifact_id`.

Do not promote model-specific concepts such as sample rate, default voice, decoder graph names,
vocabulary shape, or architecture knobs into new common manifest/artifact fields. Put a simple fact
that belongs to one variant under `manifest.variants[].metadata`. Put package-level or structured
model-private declarative data under a namespaced `extensions` entry. Pure implementation and
algorithm constants stay in model-private Python modules. Instance-overridable values stay in
`config.py`.

Static facts must have one owner. Python code may derive local values from the parsed package
config, but must not copy the same literal into `config.py`, `resources.py`, a converter, or an
application service.

## 5. Capability contract: no declaration without implementation

The static manifest and every runtime instance must report the same real capabilities.

For every capability listed in `model.yaml`, the instance must:

1. implement the exact typed protocol method;
2. register it with `register_capability(...)` while the instance is in `CREATED` state;
3. return the shared request/response schema without backend objects;
4. reject invalid state or input with stable SDK errors;
5. have a contract test that loads or constructs each runtime and calls `supports()` and
   `require()`;
6. have an inference test proving that the protocol method reaches the engine and returns a valid
   response.

Loading weights successfully is not capability support. A package that only resolves or loads
artifacts must not advertise inference capabilities. Placeholder methods, `NotImplementedError`,
and manifest-only capability declarations are prohibited.

## 6. Resource lifecycle

Resources live outside the Python package under the configured model storage root. Every
`artifacts[*].path` must use the compact canonical layout below; package loading rejects any other
layout:

```text
scoped directory:
<model_home>/<model_id>/<variant>/<runtime>/<artifact_id>/

scoped file:
<model_home>/<model_id>/<variant>/<runtime>/<artifact_id>/<filename>

shared directory:
<model_home>/<model_id>/_shared/<artifact_id>/

shared file:
<model_home>/<model_id>/_shared/<artifact_id>/<filename>
```

- `model_id` keeps its organization/name hierarchy. `variant`, `runtime`, and `artifact_id` use
  their exact declared values; `_shared` is the only reserved path segment.
- Do not put `revision`, `main`, repository hashes, source kind, format aliases, or duplicated model
  filenames into directory levels. The source repository may advance while the local storage
  address remains stable.
- `ModelArtifact.resolve(model_home)` resolves the exact artifact, while
  `ModelArtifact.storage_path(model_home)` resolves its owning directory for safe cleanup. Resource
  providers, runtime engines, application inventory, download endpoints, and deletion logic must
  consume these helpers instead of rebuilding paths from strings.
- A conversion reads the declared source artifact and writes exactly to the declared target
  artifact path. In particular, a Core ML converter must not invent a `.mlpackage`, `.mlmodelc`, or
  `coreml/model` sibling path. Its `ConversionRequest.output_path`, subsequent status check, and
  runtime loader must all refer to the same target artifact.
- Hugging Face `strip_prefix` must be used when selected repository files live below a wrapper
  directory, so the bytes committed at the artifact root match `required_files` exactly.
- `status()` is read-only and validates every required file, not just the directory.
- Downloads and conversions are explicit SDK operations; `load()` must never download or convert.
- Downloads/conversions use staging and atomic commit where possible.
- Hugging Face downloads use the source schema's internal default branch (`main`) without a YAML
  revision or configured hash check. The revision is not part of the canonical local path.
- `delete()` is idempotent and must reject paths outside the model's managed root.
- Resource responses expose availability and size, never absolute local paths.
- Tokenizers, processors, vocoders, auxiliary graphs, and generation configs are first-class
  required resources. Do not assume that loading the main weight file is enough.

## 7. Runtime engine and instance rules

Define an engine protocol in `instance.py` containing `resolve`, `load`, inference methods,
`close`, `runtime_name`, and `device` as needed. Concrete runtime modules implement it. Shared
preprocessing and response construction belong in the runtime-independent instance or private
utilities so switching runtime does not change the public behavior.

- Use an inference lock when mutable caches or sessions are not concurrency-safe.
- Preserve `asyncio.CancelledError`; wrap backend failures in `InferenceError`.
- Report actual runtime/device and SDK-measured preprocessing, inference, and postprocessing time.
- `load()` is idempotent, `unload()` is idempotent, and failed loads release partial resources.
- Do not label a hybrid pipeline as fully Core ML/ANE, MLX, or another backend. Document where
  every major stage executes.
- Stateful decoders must create/reset independent state per instance or request as appropriate.

## 8. Registration and application integration

Expose `register_<canonical_model>(registry, converters=None)` from the package. Add that call only
to the platform composition root. Registration must not load weights or touch the network.

An application may select models by capability and optional product policy. It may request
`runtime="auto"` or a user-selected runtime and inspect SDK catalog/resource availability. It must
not duplicate model variants, artifact IDs, file layouts, preprocessing, or fallback
rules. Adding a conforming model should not modify an application's inference pipeline.
Application business modules must not import `models.<package>.config`; model selection belongs to
the application manifest/composition root and discovery uses the SDK registry/catalog.

## 9. Required tests

A model is not complete until all applicable checks pass:

- manifest/config parsing and canonical identity;
- package layout and no cross-model imports;
- all declared runtimes have factories and artifacts;
- manifest capabilities equal runtime instance capabilities;
- resource missing/present/corrupt status, download, overwrite, and safe delete;
- lifecycle transitions, repeated load/unload, failed-load cleanup, and cancellation;
- typed capability request/response contract for every runtime;
- deterministic preprocessing/postprocessing unit tests;
- golden-sample correctness for every runtime and variant exposed as usable;
- cross-runtime output parity within documented tolerances;
- catalog visibility without downloading or loading;
- application smoke test through the capability interface, with no model-specific branch.

Mocks may verify orchestration, but they do not replace at least one end-to-end inference test using
real managed artifacts before a capability is advertised to applications.

## 10. Implementation sequence

1. Identify the canonical upstream model, repository, license, trust boundary, and public capability.
2. Decide whether differences are variants, runtimes, or artifacts; do not create derived IDs.
3. Create the canonical package and `model.yaml`.
4. Add typed config and complete resource lifecycle support.
5. Define the runtime-neutral engine protocol and instance lifecycle.
6. Implement one correctness-reference runtime end to end.
7. Register and test the real capability.
8. Add optimized runtimes one at a time, each with its own factory, artifacts, and parity tests.
9. Add conversion only when the model needs a supported, reproducible conversion path.
10. Register the definition at the composition root and verify catalog discovery.
11. Integrate applications only through the existing capability pipeline.
12. Run package, SDK, resource, application smoke, and real inference tests.

## 11. Definition of done

A model integration is done only when its identity is canonical, resources are reproducible,
every declared runtime loads, every declared capability performs real inference, all runtime
details remain inside the SDK, and an application can use it without importing or branching on
the model implementation.
