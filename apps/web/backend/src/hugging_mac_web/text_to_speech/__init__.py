"""Multi-model text-to-speech application."""

from hugging_mac_web.text_to_speech.blueprint import (
    TextToSpeechBlueprint,
    create_blueprint,
)

__all__ = ["TextToSpeechBlueprint", "create_blueprint"]
