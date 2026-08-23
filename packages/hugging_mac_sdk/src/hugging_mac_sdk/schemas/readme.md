# Schema Module

## Purpose

`schemas` contains the validated data contracts shared by callers, core
services, model instances, resource providers, converters, and inspectors. The
schemas are frozen Pydantic models unless a field explicitly contains an opaque
runtime-independent value such as bytes or a local `Path`.

The module contains no loading, downloading, inference, or transport behavior.

## File ownership

| File | Contracts |
| --- | --- |
| `manifest.py` | Model identity, variants, runtimes, defaults, and capabilities. |
| `artifact.py` | Scoped/shared artifact identity, path, format, source, files, and shares. |
| `resources.py` | Download sources, resolved resources, artifact/runtime status, and conversion targets. |
| `catalog.py` | Model summaries, instance snapshots, and lifecycle metrics. |
| `conversion.py` | Artifact formats and conversion request/result data. |
| `model_structure.py` | Runtime-neutral component, tensor, layer, and operator summaries. |
| `health.py` | Instance health state and optional failure message. |
| `chat.py` | Chat messages, images, requests, responses, stream events, and timing. |
| `detection.py` | Image input, detection request, boxes, faces, landmarks, and timing. |
| `hand.py` | Hand detection request, handedness, boxes, and optional 21-point landmarks. |
| `pose.py` | Pose keypoints and pose response; reuses `DetectionRequest`. |
| `segmentation.py` | Instance polygons and segmentation response; reuses `DetectionRequest`. |
| `transcription.py` | Encoded audio input and short-form transcription contracts. |
| `streaming_transcription.py` | Stateful ASR session, chunk request, and partial/final response. |
| `speech_understanding.py` | Transcription plus language, emotion, and acoustic events. |
| `voice_activity.py` | VAD settings, half-open speech intervals, and timing. |
| `speech_enhancement.py` | Encoded audio input and enhanced WAV output. |
| `speech_synthesis.py` | TTS generation/reference controls and Float32LE waveform output. |

`schemas/__init__.py` is intentionally empty. Stable application-facing exports
are curated by the package root `hugging_mac_sdk.__init__`.

## Common schema rules

- Models use `ConfigDict(frozen=True)` so snapshots and requests are not mutated
  after validation.
- Task responses include stable model, instance, runtime, and device identity
  when inference produced the result.
- Timing values are milliseconds and reject negative values.
- Raw audio/image fields set `repr=False` where applicable.
- SDK schemas do not contain NumPy arrays, PyTorch tensors, Core ML feature
  providers, ONNX values, MLX arrays, or runtime session objects.
- Application HTTP schemas map to these contracts; they are not aliases of them.

## Media inputs

`ImageInput` and `AudioInput` accept exactly one of a local path or in-memory
bytes. `ChatImage` accepts exactly one local path or HTTP(S) URL. Path access and
remote URL policy are enforced by the caller or model integration; the schema
only validates the mutually exclusive representation.

Vision coordinates are source-image pixel coordinates. Bounding boxes require
`x2 >= x1` and `y2 >= y1`. Confidence fields use `[0, 1]`. VAD segments are
half-open intervals whose end must follow their start.

## Manifest and artifact validation

`ModelManifest` requires at least one capability, runtime, and variant. Runtime
and variant names are unique; defaults must name declared entries.
`get_variant()` resolves the default and raises `ResourceNotFoundError` for an
unknown selection.

`ModelArtifact` enforces two ownership forms:

| Form | Required fields | Forbidden fields |
| --- | --- | --- |
| Scoped | `variant`, `runtime`, artifact ID, format, kind, path | `shared=true` |
| Shared | `shared=true`, artifact ID, format, kind, path | variant, runtime, `required_shares` |

Relative artifact and required-file paths reject empty values and `..`.
Required files and shares are unique. File artifacts cannot declare
`required_files`; the file itself is the readiness target. `resolve()` confines
relative paths to the supplied storage root. `canonical_path()` and
`storage_path()` provide the standard artifact and owning-directory locations.

Cross-artifact rules such as canonical model paths, declared share existence,
and unique semantic keys are validated by `core.ModelPackageConfig` and
`core.ModelDefinition`.

## Resource source contracts

`ResourceSource` is a discriminated union:

| Kind | Meaning |
| --- | --- |
| `huggingface` | One Hub file or a selected repository snapshot. |
| `url_file` | One HTTP(S) file. |
| `url_archive` | One ZIP/TAR directory artifact with optional component stripping. |
| `composite` | Several leaf sources installed at non-overlapping relative paths. |

For a Hugging Face snapshot, `strip_prefix` must be a safe relative path, requires
`allow_patterns`, and must contain every allowed pattern. Composite paths are
safe, relative, non-empty, and cannot overlap or contain one another.

`ResolvedResource` contains the materialized path, original source declaration,
calculated digest, and byte size. It is an operation result, not an artifact
declaration.

## Derived resource status

`ModelArtifactStatus` records availability and provisioning for one artifact.
Its internal `shared` and `required_shares` fields are excluded from serialized
output but are used by `ModelResourceStatus` aggregation.

During validation, `ModelResourceStatus` derives:

- one `RuntimeResourceStatus` per runtime represented by scoped artifacts;
- runtime availability from all scoped artifacts and their required shares;
- runtime size from scoped artifacts plus the shares required by that runtime;
- conversion targets from artifacts whose provisioning is `convert`;
- total size from every artifact status in the model response.

The aggregate is recomputed from `artifacts`; callers do not supply authoritative
runtime or total fields.

## Task-specific validation

- Chat requires at least one message; roles are system, user, or assistant.
- TTS reference audio and reference text must be provided together.
- Hand detection can request landmarks and records whether the source image was
  mirrored for handedness mapping.
- Streaming ASR carries a stable session ID across chunk and final responses.
- Speech enhancement returns encoded PCM WAV bytes.
- Speech synthesis returns mono little-endian Float32 bytes with sample rate and
  duration metadata.

Model-specific input requirements beyond these shared contracts are validated by
the implementing model instance.
