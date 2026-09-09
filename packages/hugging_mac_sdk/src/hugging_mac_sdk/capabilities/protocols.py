"""Initial capability protocols.

Request and response types are generic so concrete task schemas can evolve without
introducing framework tensors into the public SDK boundary.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, TypeVar

from hugging_mac_sdk.schemas.chat import ChatRequest, ChatResponse, ChatStreamEvent
from hugging_mac_sdk.schemas.detection import (
    DetectionRequest,
    DetectionResponse,
    FaceDetectionResponse,
)
from hugging_mac_sdk.schemas.document_layout import DocumentLayoutRequest, DocumentLayoutResponse
from hugging_mac_sdk.schemas.document_parsing import (
    DocumentParsingRequest,
    DocumentParsingResponse,
    DocumentParsingStreamEvent,
)
from hugging_mac_sdk.schemas.hand import HandDetectionRequest, HandDetectionResponse
from hugging_mac_sdk.schemas.pose import PoseEstimationResponse, PoseRequest
from hugging_mac_sdk.schemas.segmentation import SegmentationRequest, SegmentationResponse
from hugging_mac_sdk.schemas.speech_enhancement import (
    SpeechEnhancementRequest,
    SpeechEnhancementResponse,
)
from hugging_mac_sdk.schemas.speech_synthesis import (
    SpeechSynthesisRequest,
    SpeechSynthesisResponse,
)
from hugging_mac_sdk.schemas.speech_understanding import (
    SpeechUnderstandingRequest,
    SpeechUnderstandingResponse,
)
from hugging_mac_sdk.schemas.streaming_transcription import (
    StreamingTranscriptionRequest,
    StreamingTranscriptionResponse,
    StreamingTranscriptionSession,
)
from hugging_mac_sdk.schemas.transcription import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from hugging_mac_sdk.schemas.voice_activity import VoiceActivityRequest, VoiceActivityResponse

RequestT = TypeVar("RequestT", contravariant=True)
ResponseT = TypeVar("ResponseT", covariant=True)


class ObjectDetection(Protocol):
    """Detect objects in an image or image batch."""

    async def detect(self, request: DetectionRequest) -> DetectionResponse: ...


class FaceDetection(Protocol):
    """Detect faces and their canonical five alignment landmarks."""

    async def detect_faces(self, request: DetectionRequest) -> FaceDetectionResponse: ...


class HandDetection(Protocol):
    """Detect hands and optionally estimate 21 landmarks through one model interface."""

    async def detect_hands(self, request: HandDetectionRequest) -> HandDetectionResponse: ...


class PoseEstimation(Protocol):
    """Estimate person keypoints in an image."""

    async def estimate_pose(self, request: PoseRequest) -> PoseEstimationResponse: ...


class InstanceSegmentation(Protocol):
    """Detect objects and return their visible instance polygons."""

    async def segment(self, request: SegmentationRequest) -> SegmentationResponse: ...


class Chat(Protocol):
    """Generate conversational responses, optionally as a stream."""

    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]: ...


class DocumentParsing(Protocol):
    """Parse one or more document images, optionally as a text stream."""

    async def parse_document(self, request: DocumentParsingRequest) -> DocumentParsingResponse: ...

    def stream_document(
        self, request: DocumentParsingRequest
    ) -> AsyncIterator[DocumentParsingStreamEvent]: ...


class DocumentLayoutAnalysis(Protocol):
    """Locate semantic document regions and return them in reading order."""

    async def analyze_layout(self, request: DocumentLayoutRequest) -> DocumentLayoutResponse: ...


class SpeechTranscription(Protocol):
    """Transcribe speech into text."""

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse: ...


class StreamingSpeechTranscription(Protocol):
    """Maintain decoder state while accepting consecutive audio chunks."""

    async def start_stream(self) -> StreamingTranscriptionSession: ...

    async def transcribe_stream(
        self, request: StreamingTranscriptionRequest
    ) -> StreamingTranscriptionResponse: ...

    async def finish_stream(self, session_id: str) -> StreamingTranscriptionResponse: ...

    async def cancel_stream(self, session_id: str) -> None: ...


class SpeechEnhancement(Protocol):
    """Remove background noise from speech audio."""

    async def enhance_speech(
        self, request: SpeechEnhancementRequest
    ) -> SpeechEnhancementResponse: ...


class SpeechSynthesis(Protocol):
    """Synthesize speech from text, optionally cloning a reference voice."""

    async def synthesize(
        self,
        request: SpeechSynthesisRequest,
    ) -> SpeechSynthesisResponse: ...


class SpeechUnderstanding(Protocol):
    """Recognize speech and return its rich paralinguistic annotations."""

    async def understand_speech(
        self,
        request: SpeechUnderstandingRequest,
    ) -> SpeechUnderstandingResponse: ...


class VoiceActivityDetection(Protocol):
    """Detect speech intervals in audio."""

    async def detect_voice_activity(
        self, request: VoiceActivityRequest
    ) -> VoiceActivityResponse: ...


class ImageEmbedding(Protocol[RequestT, ResponseT]):
    """Create embeddings for images."""

    async def embed_images(self, request: RequestT) -> ResponseT: ...


class TextEmbedding(Protocol[RequestT, ResponseT]):
    """Create embeddings for text."""

    async def embed_texts(self, request: RequestT) -> ResponseT: ...


class ImageTextSimilarity(Protocol[RequestT, ResponseT]):
    """Calculate image/text similarity scores."""

    async def similarity(self, request: RequestT) -> ResponseT: ...
