# ruff: noqa: RUF001
"""Text-to-speech model profiles and settings."""

from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class TtsModelProfile:
    model_id: str
    display_name: str
    short_name: str
    description: str
    variant: str
    runtime: str
    required_artifact_id: str
    voices: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    requires_reference_voice: bool = False
    max_new_tokens: int = 1024


AUDIO8_TTS_PROFILE = TtsModelProfile(
    model_id="audio8/audio8-tts-preview-0.6b",
    display_name="Audio8 TTS Preview 0.6B",
    short_name="Audio8 TTS",
    description="多语言生成与声音克隆模型；仅使用稳定的 PyTorch CPU FP32 路径。",
    variant="preview",
    runtime="pytorch",
    required_artifact_id="source",
    languages=("auto",),
)

AUDIO8_TTS_ONNX_INT4_PROFILE = TtsModelProfile(
    model_id="audio8/audio8-tts-preview-0.6b-onnx-int4",
    display_name="Audio8 TTS ONNX INT4",
    short_name="Audio8 ONNX",
    description="官方 INT4 ONNX Runtime 版本，CPU 内存更低；支持参考音频音色 profile。",
    variant="int4",
    runtime="onnx",
    required_artifact_id="source",
    languages=("auto",),
    requires_reference_voice=True,
    max_new_tokens=256,
)

KOKORO_82M_PROFILE = TtsModelProfile(
    model_id="hexgrad/kokoro-82m",
    display_name="Kokoro 82M",
    short_name="Kokoro",
    description="轻量快速的 24 kHz 本地语音，使用 PyTorch MPS 加速。",
    variant="v1.0",
    runtime="pytorch-mps",
    required_artifact_id="source",
    voices=(
        "af_heart",
        "af_bella",
        "af_nova",
        "af_sarah",
        "am_adam",
        "am_michael",
        "bf_emma",
        "bm_george",
        "jf_alpha",
        "zf_xiaoxiao",
    ),
    languages=("a", "b", "e", "f", "h", "i", "p", "j", "z"),
)

TTS_MODEL_PROFILES = {
    profile.model_id: profile
    for profile in (
        AUDIO8_TTS_PROFILE,
        AUDIO8_TTS_ONNX_INT4_PROFILE,
        KOKORO_82M_PROFILE,
    )
}


class TextToSpeechSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUGGING_MAC_APP_TEXT_TO_SPEECH_",
        env_file=".env",
        extra="ignore",
    )

    max_text_characters: int = 2000
    max_reference_audio_bytes: int = 50 * 1024 * 1024
