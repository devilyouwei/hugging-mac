"""Initial capability protocols.

Request and response types are generic so concrete task schemas can evolve without
introducing framework tensors into the public SDK boundary.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, TypeVar

from hugging_mac_sdk.schemas.chat import ChatRequest, ChatResponse, ChatStreamEvent
from hugging_mac_sdk.schemas.detection import DetectionRequest, DetectionResponse
from hugging_mac_sdk.schemas.pose import PoseEstimationResponse, PoseRequest
from hugging_mac_sdk.schemas.segmentation import SegmentationRequest, SegmentationResponse
from hugging_mac_sdk.schemas.speech_synthesis import (
    SpeechSynthesisRequest,
    SpeechSynthesisResponse,
)
from hugging_mac_sdk.schemas.speech_understanding import (
    SpeechUnderstandingRequest,
    SpeechUnderstandingResponse,
)
from hugging_mac_sdk.schemas.transcription import (
    TranscriptionRequest,
    TranscriptionResponse,
)

RequestT = TypeVar("RequestT", contravariant=True)
ResponseT = TypeVar("ResponseT", covariant=True)


class ObjectDetection(Protocol):
    """Detect objects in an image or image batch."""

    async def detect(self, request: DetectionRequest) -> DetectionResponse: ...


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


class SpeechTranscription(Protocol):
    """Transcribe speech into text."""

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse: ...


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


class ImageEmbedding(Protocol[RequestT, ResponseT]):
    """Create embeddings for images."""

    async def embed_images(self, request: RequestT) -> ResponseT: ...


class TextEmbedding(Protocol[RequestT, ResponseT]):
    """Create embeddings for text."""

    async def embed_texts(self, request: RequestT) -> ResponseT: ...


class ImageTextSimilarity(Protocol[RequestT, ResponseT]):
    """Calculate image/text similarity scores."""

    async def similarity(self, request: RequestT) -> ResponseT: ...
