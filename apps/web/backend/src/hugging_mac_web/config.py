"""Typed platform configuration."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path, user_data_path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WEB_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    runtime_preference: str = "coreml,mlx,mps,onnx"
    data_dir: Path = Field(default_factory=lambda: user_data_path("hugging-mac") / "web")
    cache_dir: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "web")
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    database_path: Path | None = None
    max_image_upload_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=40_000_000, gt=0)
    max_document_upload_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    max_document_pages: int = Field(default=2000, gt=0)
    sse_heartbeat_seconds: float = Field(default=15.0, gt=0)

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @cached_property
    def parsed_cors_origins(self) -> tuple[str, ...]:
        return tuple(origin.strip() for origin in self.cors_origins.split(",") if origin.strip())

    @cached_property
    def runtime_preferences(self) -> tuple[str, ...]:
        return tuple(
            runtime.strip() for runtime in self.runtime_preference.split(",") if runtime.strip()
        )

    @property
    def resolved_database_path(self) -> Path:
        return self.database_path or self.data_dir / "platform.json"
