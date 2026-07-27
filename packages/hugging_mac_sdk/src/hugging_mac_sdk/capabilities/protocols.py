"""Initial capability protocols.

Request and response types are generic so concrete task schemas can evolve without
introducing framework tensors into the public SDK boundary.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, TypeVar

from hugging_mac_sdk.schemas.detection import DetectionRequest, DetectionResponse

RequestT = TypeVar("RequestT", contravariant=True)
ResponseT = TypeVar("ResponseT", covariant=True)
EventT = TypeVar("EventT", covariant=True)


class ObjectDetection(Protocol):
    """Detect objects in an image or image batch."""

    async def detect(self, request: DetectionRequest) -> DetectionResponse: ...


class Chat(Protocol[RequestT, ResponseT, EventT]):
    """Generate conversational responses, optionally as a stream."""

    async def chat(self, request: RequestT) -> ResponseT: ...

    def stream_chat(self, request: RequestT) -> AsyncIterator[EventT]: ...


class SpeechTranscription(Protocol[RequestT, ResponseT]):
    """Transcribe speech into timestamped text."""

    async def transcribe(self, request: RequestT) -> ResponseT: ...


class ImageEmbedding(Protocol[RequestT, ResponseT]):
    """Create embeddings for images."""

    async def embed_images(self, request: RequestT) -> ResponseT: ...


class TextEmbedding(Protocol[RequestT, ResponseT]):
    """Create embeddings for text."""

    async def embed_texts(self, request: RequestT) -> ResponseT: ...


class ImageTextSimilarity(Protocol[RequestT, ResponseT]):
    """Calculate image/text similarity scores."""

    async def similarity(self, request: RequestT) -> ResponseT: ...
