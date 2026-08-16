"""Shared platform dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from hugging_mac_sdk import ModelSdk, RuntimePolicy
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.audio8_asr import register_audio8_asr
from hugging_mac_sdk.models.audio8_tts import register_audio8_tts
from hugging_mac_sdk.models.deepfilternet3 import register_deepfilternet3
from hugging_mac_sdk.models.kokoro import register_kokoro
from hugging_mac_sdk.models.nemotron_3_5_asr import register_nemotron_3_5_asr
from hugging_mac_sdk.models.qwen3_5 import register_qwen3_5
from hugging_mac_sdk.models.qwen3_asr import register_qwen3_asr
from hugging_mac_sdk.models.qwen3_tts import (
    register_qwen3_tts,
)
from hugging_mac_sdk.models.sensevoice import register_sensevoice
from hugging_mac_sdk.models.silero import register_silero
from hugging_mac_sdk.models.yolov8 import register_yolov8
from hugging_mac_sdk.models.yolov8_pose import register_yolov8_pose
from hugging_mac_sdk.models.yolov8_seg import register_yolov8_seg

from hugging_mac_web.app_registry import AppRegistry
from hugging_mac_web.config import WebSettings
from hugging_mac_web.shared.cache import LocalCache
from hugging_mac_web.shared.storage import TinyDocumentStore


@dataclass(slots=True)
class PlatformContext:
    settings: WebSettings
    models: ModelSdk
    converters: ConverterRegistry
    apps: AppRegistry
    documents: TinyDocumentStore
    cache: LocalCache

    def close(self) -> None:
        self.documents.close()


def create_context(settings: WebSettings) -> PlatformContext:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    models = ModelSdk(
        runtime_policy=RuntimePolicy(settings.runtime_preferences),
    )
    converters = ConverterRegistry()
    register_audio8_asr(models.registry, converters)
    register_audio8_tts(models.registry)
    register_deepfilternet3(models.registry)
    register_kokoro(models.registry)
    register_nemotron_3_5_asr(models.registry)
    register_qwen3_asr(models.registry)
    register_qwen3_tts(models.registry)
    register_qwen3_5(models.registry)
    register_sensevoice(models.registry, converters)
    register_silero(models.registry)
    register_yolov8(models.registry, converters)
    register_yolov8_pose(models.registry, converters)
    register_yolov8_seg(models.registry, converters)
    documents = TinyDocumentStore(settings.resolved_database_path)
    return PlatformContext(
        settings=settings,
        models=models,
        converters=converters,
        apps=AppRegistry(),
        documents=documents,
        cache=LocalCache(settings.cache_dir, documents),
    )
