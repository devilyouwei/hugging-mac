# Capability Module

## Purpose

`capabilities` defines the task operations exposed by loaded model instances.
Each capability is a structural `Protocol` with typed SDK request and response
schemas. The module contains no lifecycle, resource, runtime, or transport logic.

## Files

| File | Responsibility |
| --- | --- |
| `protocols.py` | Capability protocol definitions. |
| `__init__.py` | Public exports for all capability types. |

## Protocols

| Capability | Method contract |
| --- | --- |
| `DecisionMaking` | `decision_support() -> DecisionSupport`; `decide(DecisionRequest) -> DecisionResponse` |
| `ObjectDetection` | `detect(DetectionRequest) -> DetectionResponse` |
| `FaceDetection` | `detect_faces(DetectionRequest) -> FaceDetectionResponse` |
| `HandDetection` | `detect_hands(HandDetectionRequest) -> HandDetectionResponse` |
| `PoseEstimation` | `estimate_pose(PoseRequest) -> PoseEstimationResponse` |
| `InstanceSegmentation` | `segment(SegmentationRequest) -> SegmentationResponse` |
| `Chat` | `chat(ChatRequest) -> ChatResponse` and `stream_chat(...) -> AsyncIterator[ChatStreamEvent]` |
| `DocumentParsing` | `parse_document(DocumentParsingRequest) -> DocumentParsingResponse` and `stream_document(...) -> AsyncIterator[DocumentParsingStreamEvent]` |
| `DocumentLayoutAnalysis` | `analyze_layout(DocumentLayoutRequest) -> DocumentLayoutResponse` |
| `SpeechTranscription` | `transcribe(TranscriptionRequest) -> TranscriptionResponse` |
| `StreamingSpeechTranscription` | Start, append, finish, and cancel a stateful transcription session. |
| `SpeechEnhancement` | `enhance_speech(SpeechEnhancementRequest) -> SpeechEnhancementResponse` |
| `SpeechSynthesis` | `synthesize(SpeechSynthesisRequest) -> SpeechSynthesisResponse` |
| `SpeechUnderstanding` | `understand_speech(SpeechUnderstandingRequest) -> SpeechUnderstandingResponse` |
| `VoiceActivityDetection` | `detect_voice_activity(VoiceActivityRequest) -> VoiceActivityResponse` |
| `ImageEmbedding` | Generic typed image embedding request/response. |
| `TextEmbedding` | Generic typed text embedding request/response. |
| `ImageTextSimilarity` | Generic typed image/text similarity request/response. |

## Instance binding

A model instance registers capability implementations while its state is
`CREATED`:

```text
ModelInstance.__init__
  -> register_capability(CapabilityProtocol, implementation)
  -> load()
  -> ModelHandle.require(CapabilityProtocol)
  -> typed method call
```

`BaseModelInstance.supports()` checks registration. `require()` returns the
registered implementation or raises `UnsupportedCapabilityError`. Capability
registration is rejected after the instance leaves `CREATED`, so the capability
set is stable throughout loading and inference.

The registered implementation may be the instance itself or another object
owned by the instance. It may share that instance's engine, tokenizer, decoder
state, and inference lock.

## Stateful and streaming contracts

`Chat.stream_chat()` returns typed asynchronous events without creating a
separate session object. `StreamingSpeechTranscription` uses an explicit session
ID and four operations:

1. `start_stream()` creates model state.
2. `transcribe_stream()` appends one audio chunk.
3. `finish_stream()` flushes and removes the session.
4. `cancel_stream()` removes the session without a final result.

Session storage, locking, and cleanup are implemented by the model instance. The
capability module defines only the callable contract.

## Boundary rules

- Capability arguments and results use types from `schemas`.
- Protocols do not expose tensors, framework sessions, processors, model paths,
  or runtime-specific cache objects.
- Lifecycle operations use `ModelHandle` and `BaseModelInstance`, not capability
  methods.
- Resource download and conversion use core resource services.
- HTTP, WebSocket, and SSE mapping belongs to the application layer.
- A capability listed by a model manifest must be registered by every runtime
  instance that advertises that model definition.

## Shared decision contract

D3 and Vela implement the same `DecisionMaking` protocol. Query
`decision_support()` for supported question types, media, choice/score limits,
and custom noul descriptions; these facts come from `model.yaml` extensions.
`choice` selects one label from any supported number of classes. `noul` returns
the probability of the affirmative outcome. `score` returns the expected
zero-based index of ordered levels. Models need not implement all three types.

Requests accept typed `ChoiceQuestion`, `NoulQuestion`, and `ScoreQuestion` values
or equivalent dictionaries. Choice label lists normalize to description mappings.
Answers are `ChoiceAnswer`, `NoulAnswer`, `ScoreAnswer`, or `DecisionError`,
discriminated by `type`; access fields as attributes (for example `.choice` or
`.noul`). Use `model_dump(mode="json", exclude_none=True)` for JSON-compatible
output. Confidence is optional and is not synthesized when a backend omits it.

Malformed raw question items and unsupported model features produce per-question
errors, preserving valid siblings and input ordering. Invalid request envelopes or
explicitly constructed invalid typed questions raise schema validation errors.
The shared dispatcher calls the backend once per accepted question, aggregates
actual token usage and timings, and validates native results. Backend input
`ValueError`s become per-question `invalid_question` errors; unexpected runtime
failures raise `InferenceError`, and cancellation propagates unchanged.
