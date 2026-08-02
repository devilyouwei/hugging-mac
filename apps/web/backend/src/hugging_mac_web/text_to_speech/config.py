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

AUDIO8_TTS_MLX_BF16_PROFILE = TtsModelProfile(
    model_id="mlx-community/audio8-tts-preview-0.6b-bf16",
    display_name="Audio8 TTS MLX BF16",
    short_name="Audio8 MLX",
    description="Apple Silicon GPU 版本，支持多语言和零样本参考音频声音克隆。",
    variant="bf16",
    runtime="mlx",
    required_artifact_id="model",
    languages=("auto",),
    requires_reference_voice=True,
    max_new_tokens=256,
)

KOKORO_82M_PROFILE = TtsModelProfile(
    model_id="mlx-community/kokoro-82m-bf16",
    display_name="Kokoro 82M",
    short_name="Kokoro",
    description="轻量快速的 24 kHz 本地语音，使用 MLX BF16 在 Apple Silicon GPU 上推理。",
    variant="bf16",
    runtime="mlx",
    required_artifact_id="model",
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
    languages=("a", "b", "e", "f", "h", "i", "p", "j", "z"),
)

TTS_MODEL_PROFILES = {
    profile.model_id: profile
    for profile in (
        AUDIO8_TTS_PROFILE,
        AUDIO8_TTS_MLX_BF16_PROFILE,
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
