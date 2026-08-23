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
    requires_reference_audio: bool = False
    max_new_tokens: int = 1024


AUDIO8_TTS_PROFILE = TtsModelProfile(
    model_id="audio8/audio8-tts-preview",
    display_name="Audio8 TTS Preview",
    short_name="Audio8 TTS",
    description=(
        "多语言生成与声音克隆模型；PyTorch 默认使用 CPU，也可选择 MPS，"
        "异常时回退 CPU，0.6B 另支持 MLX。"
    ),
    variant="0.6b-preview",
    runtime="pytorch",
    required_artifact_id="source",
    languages=("auto",),
)

KOKORO_82M_PROFILE = TtsModelProfile(
    model_id="hexgrad/kokoro",
    display_name="Kokoro 82M",
    short_name="Kokoro",
    description="预编译端到端 Core ML 模型，优先使用 Apple Neural Engine。",
    variant="v1.0",
    runtime="coreml",
    required_artifact_id="coreml",
    voices=(
        "af_alloy",
        "af_aoede",
        "af_heart",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ),
    languages=("en-us", "en-gb", "es", "fr", "hi", "it", "pt", "ja", "zh"),
)

QWEN3_TTS_0_6B_BASE_4BIT_PROFILE = TtsModelProfile(
    model_id="qwen/qwen3-tts-12hz",
    display_name="Qwen3-TTS",
    short_name="Qwen3-TTS",
    description="支持 MLX 声音克隆与 Core ML 默认音色合成。",
    variant="0.6b-base",
    runtime="mlx",
    required_artifact_id="mlx-4bit",
    languages=(
        "auto",
        "chinese",
        "english",
        "japanese",
        "korean",
        "german",
        "french",
        "russian",
        "portuguese",
        "spanish",
        "italian",
    ),
    requires_reference_audio=True,
    max_new_tokens=2048,
)

TTS_MODEL_PROFILES = {
    profile.model_id: profile
    for profile in (
        AUDIO8_TTS_PROFILE,
        KOKORO_82M_PROFILE,
        QWEN3_TTS_0_6B_BASE_4BIT_PROFILE,
    )
}


class TextToSpeechSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_TEXT_TO_SPEECH_",
        env_file=".env",
        extra="ignore",
    )

    max_text_characters: int = 2000
    max_reference_audio_bytes: int = 50 * 1024 * 1024
