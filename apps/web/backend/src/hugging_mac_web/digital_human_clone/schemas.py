"""HTTP schemas for the Digital Human game."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelStateView(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["asr", "llm", "tts"]
    model_id: str
    runtime: str
    resources_ready: bool
    ready_instance_id: str | None = None


class SetupView(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[ModelStateView, ...]
    llm_variants: tuple["LlmVariantView", ...]
    selected_llm_variant: str

    @property
    def ready(self) -> bool:
        return all(model.ready_instance_id is not None for model in self.models)


class LoadedModelsView(BaseModel):
    model_config = ConfigDict(frozen=True)

    asr_instance_id: str
    llm_instance_id: str
    tts_instance_id: str


class LlmVariantView(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    description: str


class TranscriptView(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    inference_ms: float | None = None


class ConversationMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)
