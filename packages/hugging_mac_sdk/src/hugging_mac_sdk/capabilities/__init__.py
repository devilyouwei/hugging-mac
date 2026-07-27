"""Task-specific capability contracts."""

from hugging_mac_sdk.capabilities.protocols import (
    Chat,
    ImageEmbedding,
    ImageTextSimilarity,
    ObjectDetection,
    SpeechTranscription,
    TextEmbedding,
)

__all__ = [
    "Chat",
    "ImageEmbedding",
    "ImageTextSimilarity",
    "ObjectDetection",
    "SpeechTranscription",
    "TextEmbedding",
]
