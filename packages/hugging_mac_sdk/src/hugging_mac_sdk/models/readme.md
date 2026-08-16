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

The existing tree contains legacy packages that do not yet satisfy this identity rule. They are
migration targets, not examples to copy.

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
- `model.yaml` contains declarative identity, revision, license, capabilities, variants, runtimes,
  sources, and artifacts. YAML must not import or execute Python.
- `config.py` defines frozen, strongly typed instance/resource/runtime options.
- `definition.py` binds every declared runtime to a concrete factory and registers resources and
  converters.
- `instance.py` owns lifecycle, concurrency, capability registration, common pre/postprocessing,
  and composition with an engine protocol. It must not import concrete runtime engines.
- Runtime files load and execute one backend through the shared runtime providers where possible.
- `resources.py` resolves, validates, reports, downloads, and deletes only this model's assets.
- `converter.py` contains model-specific conversion logic; general framework mechanics belong in
  `converters/`.
- `utils/` is private to the package. A model package must not import another model package.

## 4. Manifest, variants, runtimes, and artifacts

Pin reproducible upstream revisions and declare the upstream license and trust boundary. Each
variant must describe a weight/model choice that preserves the same public task contract. Each
runtime must describe an implementation that actually exists on the instance factory path.

Every declared runtime must have:

1. a concrete factory in `runtime_factories`;
2. all required artifacts declared for every supported variant;
3. a resource status whose aggregated runtime availability is accurate;
4. a load/unload implementation and an inference contract test.

An artifact is a physical input to a runtime. It may be a file or directory and may contain weight
shards, tokenizer files, processors, configuration, or compiled graphs. A runtime may require more
than one artifact. Prebuilt conversions are downloadable artifacts; locally produced conversions
use `provisioning: convert`. Artifact IDs are SDK data and must not become application logic.

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

Resources live outside the Python package under the configured model storage root. Use the layout:

```text
<model_home>/<model_id>/<revision>/<variant>/<runtime-or-source>/<artifact>/
```

- `status()` is read-only and validates every required file, not just the directory.
- Downloads and conversions are explicit SDK operations; `load()` must never download or convert.
- Downloads/conversions use staging and atomic commit where possible.
- Fix revisions and hashes for reproducibility, especially for executable or pickle-backed files.
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
not duplicate model revisions, variants, artifact IDs, file layouts, preprocessing, or fallback
rules. Adding a conforming model should not modify an application's inference pipeline.

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

1. Identify the canonical upstream model, revision, license, trust boundary, and public capability.
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
