# Model Package Integration Standard

## Purpose

`models` contains canonical model integrations. Each package binds one model
identity to its variants, runtime implementations, artifacts, resource provider,
capabilities, optional converters, and artifact inspectors.

Applications use model packages through `ModelSdk` and capability protocols.
They do not import model-private engines, configuration, artifact IDs, or paths.

## Canonical identity

One upstream architecture and public task contract use one package and one model
ID. Framework, format, precision, quantization, parameter size, and deployment
target do not create separate model IDs.

| Difference | Representation |
| --- | --- |
| Interchangeable weight size or edition | `manifest.variants` |
| Execution backend | `manifest.runtimes` and definition runtime factory |
| Physical format, precision, quantization, or files | `artifacts` |
| Different public capability or incompatible architecture | Separate canonical model package when required |

Package directories use Python `snake_case`. The manifest model ID uses the
canonical upstream organization/name and remains stable across runtime builds.

## Package layout

```text
models/<canonical_model>/
├── __init__.py
├── readme.md
├── model.yaml
├── config.py
├── definition.py
├── instance.py
├── resources.py          # when managed resources are supported
├── torch.py              # when PyTorch is implemented
├── mlx.py                # when MLX is implemented
├── coreml.py             # when Core ML is implemented
├── onnx.py               # when ONNX Runtime is implemented
├── converter.py          # when local conversion is implemented
└── utils/                # model-private loaders, preprocessing, and types
```

Files are added only when the integration uses them.

| File | Responsibility |
| --- | --- |
| `__init__.py` | Export the definition, parsed manifest, and registration function. |
| `readme.md` | Describe implemented runtimes, capability flow, file ownership, concurrency, and limits. |
| `model.yaml` | Single source for static model identity, variants, runtimes, artifacts, sources, files, and shares. |
| `config.py` | Frozen instance-variable options such as model home, credentials, device, compute policy, and inference controls. |
| `definition.py` | Package composition root; load YAML and inject manifest/artifacts into factories, providers, converters, and inspectors. |
| `instance.py` | Runtime-neutral lifecycle, capability registration, engine protocol, and shared request/response mapping. |
| Runtime module | Load and execute one backend through a model-private engine. |
| `resources.py` | Report, download, convert, and delete declared artifacts using injected declarations. |
| `converter.py` | Model-specific conversion request construction or architecture-aware export. |
| `utils/` | Private implementation shared only inside the package. |

A model package does not import another model package.

## Static configuration ownership

`model.yaml` owns:

- model ID, presentation, family, license, tags, capabilities, and source URL;
- variants, runtimes, devices, dtypes, quantizations, platforms, architectures,
  and required modules;
- artifact IDs, ownership, formats, kinds, canonical paths, required files,
  sources, conversion support, and shared dependencies.

`config.py` does not repeat these values. It contains only values that can differ
per instance or operation. Python code receives parsed manifest/artifact objects
from `definition.py` instead of reconstructing static facts.

Model-specific declarative facts use:

- `manifest.variants[].metadata` for a fact belonging to one variant;
- a namespaced top-level `extensions` entry for package-wide structured data;
- Python implementation constants for algorithm-only values;
- `config.py` for caller-overridable values.

Sample rate, default voice, graph/function names, vocabulary shape, and similar
model-private facts are not common manifest fields.

## `model.yaml` structure

The top level contains:

```yaml
manifest: {}
artifacts: []
extensions: {}  # optional
```

`manifest` and `artifacts` are separated by one blank line.

### Manifest field order

The identity section uses this order:

```text
schema_version
model_id
display_name
description
tags
family
capabilities
source_url
license
```

`license` is the final identity field, followed by one blank line. The remaining
manifest fields are consecutive:

```text
runtimes
default_runtime
variants
default_variant
```

Runtime entries use:

```text
name
devices
dtypes
quantizations          # optional
platforms
architectures
required_modules
```

Variant entries use:

```text
name
display_name
description            # optional
metadata               # optional
```

`required_modules` uses YAML flow-array syntax.

### Artifact field order

A scoped artifact uses:

```text
artifact_id
convert: true          # optional; immediately after artifact_id
variant
runtime
format
kind
path
required_files         # directory artifacts
required_shares        # optional; immediately after required_files
source                 # optional
metadata               # optional
```

A shared artifact uses:

```text
artifact_id
shared: true           # immediately after artifact_id
format
kind
path
required_files
source
metadata               # optional
```

Shared artifacts do not declare `variant`, `runtime`, `convert`, or
`required_shares`. Artifact mappings are consecutive without blank lines.
`kind` always precedes `path`.

`required_files`, `required_shares`, and `required_modules` use flow arrays:

```yaml
required_files: [config.json, model.safetensors]
required_shares: [tokenizer]
required_modules: [torch, transformers]
```

Hugging Face `allow_patterns` uses a block list:

```yaml
allow_patterns:
- config.json
- model.safetensors
```

### Source field order

A source starts with `kind`, followed by:

1. locator: `repo_id` or `url`;
2. selection: `filename` or `allow_patterns`;
3. extraction: `format`, `strip_prefix`, or `strip_components` when applicable.

Artifact source declarations have one owner: `artifacts[*].source`. Manifests and
variants do not contain a second resource/source mapping.

### Representative package

```yaml
manifest:
  schema_version: '1'
  model_id: upstream/example-model
  display_name: Example Model
  description: Example model for Apple Silicon.
  tags: [example, coreml]
  family: example
  capabilities: [example-task]
  source_url: https://huggingface.co/upstream/example-model
  license: Apache-2.0

  runtimes:
  - name: coreml
    devices: [all, cpu-and-neural-engine, cpu-and-gpu, cpu-only]
    dtypes: [float16]
    platforms: [darwin]
    architectures: [arm64]
    required_modules: [coremltools, numpy]
  default_runtime: coreml
  variants:
  - name: base
    display_name: Base
    description: Base model.
    metadata:
      parameters_millions: 100
  default_variant: base

artifacts:
- artifact_id: coreml-fp16
  convert: true
  variant: base
  runtime: coreml
  format: coreml
  kind: directory
  path: upstream/example-model/base/coreml/coreml-fp16
  required_files: [Manifest.json, Data/model.mlmodel]
  required_shares: [tokenizer]
  source:
    kind: huggingface
    repo_id: upstream/example-model-coreml
    allow_patterns:
    - example.mlpackage/**
    strip_prefix: example.mlpackage
- artifact_id: tokenizer
  shared: true
  format: tokenizer
  kind: directory
  path: upstream/example-model/_shared/tokenizer
  required_files: [tokenizer.json, tokenizer_config.json]
  source:
    kind: huggingface
    repo_id: upstream/example-model
    allow_patterns:
    - tokenizer.json
    - tokenizer_config.json
```

## Artifact semantics

A scoped artifact belongs to exactly one `(variant, runtime)` combination. Its
semantic key is `(variant, runtime, artifact_id)` and must be unique. The
presence of scoped artifacts declares which variant/runtime combinations are
supported.

Directory artifacts declare the complete `required_files` inventory. File
artifacts do not declare `required_files`; their declared file path is the
availability target. Required-file paths are safe relative paths.

`source` makes an artifact downloadable. `convert: true` makes it a supported
conversion target. Both may be present when download and conversion publish the
same declared files at the same canonical path. `provisioning` is derived status,
not a YAML field.

Do not declare `revision`, `expected_sha256`, `file_sha256`, or another expected
digest. Hugging Face sources use their schema default branch and are trusted
without configured content-hash comparison. Calculated digests may still appear
in operation results.

## Shared artifacts

A shared artifact is variant- and runtime-independent. It has its own directly
downloadable source and cannot be converted or depend on another share.

Dependencies are declared by consuming scoped artifacts:

```yaml
required_shares: [tokenizer, processor]
```

This forms a many-to-many relation. Core resource status, download preparation,
and conversion preparation resolve shares from the exact scoped artifacts being
operated on. A shared artifact ID must be unique, must not overlap a scoped ID,
and must be referenced by at least one scoped artifact.

Some third-party loaders require scoped and shared files in one directory. Use
`resources.merged_directory_view()` and materialize lazy data before the context
closes; do not copy shared files into every artifact.

## Canonical storage paths

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

`model_id` preserves its organization/name hierarchy. `_shared` is the only
reserved segment. Revision, source kind, repository branch, hash, and format
aliases are not path segments.

All code uses `ModelArtifact.resolve(model_home)` for the exact artifact and
`storage_path(model_home)` for its owning directory. Resource providers,
converters, engines, inspection, and applications do not rebuild paths from
string fragments.

When a Hub snapshot stores files below a wrapper directory, `strip_prefix`
publishes that directory directly at the artifact root so its layout matches
`required_files`.

## Definition and runtime composition

`definition.py` loads `model.yaml`, selects artifacts by semantic key, constructs
runtime factories, and creates one immutable `ModelDefinition`. Every declared
runtime has a concrete factory. Artifacts and inspectors reference only declared
variants/runtimes.

An instance defines a runtime-neutral engine protocol for the operations it
needs, such as resolve, load, inference, close, runtime name, and device. Concrete
runtime modules implement that protocol. The instance owns capability
registration and shared response mapping; engines own framework-specific loading
and execution.

Runtime behavior follows these rules:

- validate local artifacts before framework loading;
- import optional frameworks only on the selected runtime path;
- serialize inference when the session or mutable state is not concurrency-safe;
- preserve `asyncio.CancelledError`;
- convert unexpected inference failures to `InferenceError`;
- report actual runtime/device and measured stage timings;
- release partial state after failed load;
- isolate stateful decoder/session state at the correct instance or request scope.

Hybrid implementations document where each major stage executes and report a
runtime name that matches the actual pipeline.

## Capabilities

Every capability declared by the manifest is implemented and registered by the
runtime instance while it is in `CREATED` state. The implementation accepts and
returns the shared SDK schema and does not expose framework values.

Registration tests compare the manifest capability set with instance
capabilities for every runtime. Inference tests invoke the protocol through a
`ModelHandle`; loading weights alone is not capability implementation.

## Resource provider and conversion

The resource provider consumes injected manifest/artifact objects. It implements
read-only status plus explicit source download, conversion, and idempotent delete.

- Status validates complete required files for each declared artifact.
- Download writes the artifact's declared source to its canonical path.
- Conversion reads a declared source artifact and writes the declared target
  artifact path.
- Download and conversion support explicit overwrite.
- Load does not call download or conversion.
- Delete rejects paths outside the managed model root.
- Status responses expose availability and size, not local paths.

Model-specific converters own architecture-sensitive export and output
validation. Generic selection and common framework mechanics use the SDK
`converters` module.

## Registration and application boundary

The package exports `register_<model>(registry, converters=None)` as applicable.
Registration is called only from the consuming platform composition root and has
no weight, network, or framework side effects.

Applications select registered definitions through catalog, resource, runtime,
variant, and capability APIs. They do not import `models.<package>.config`, use
artifact IDs for business branching, reproduce preprocessing, or inspect runtime
engines.

## Required verification

Each integration covers the applicable contracts:

- YAML parsing, canonical identity, field order, and path validation;
- runtime factory and artifact coverage for declared combinations;
- manifest/instance capability equality;
- resource missing/present/incomplete status, download, overwrite, conversion,
  and safe idempotent deletion;
- required-share selection for status and provisioning;
- lifecycle transitions, repeated load/unload, failed-load cleanup, and
  cancellation;
- typed capability inference for every exposed runtime and variant;
- preprocessing/postprocessing unit behavior and cross-runtime parity where
  multiple runtimes implement the same task;
- catalog and registration without downloading, loading, or importing unrelated
  frameworks;
- application smoke coverage through capability APIs without model-private
  imports.

Real managed artifacts are used in explicit smoke/integration coverage before a
runtime is treated as usable. Default unit and contract tests remain offline.
