"""Hugging Face download endpoint configuration, read at operation time."""

import os
from pathlib import Path

from dotenv import dotenv_values


def download_endpoint() -> str | None:
    """Process environment overrides .env in the current working directory."""
    value = os.environ.get("HF_ENDPOINT")
    if value is None:
        value = dotenv_values(Path.cwd() / ".env").get("HF_ENDPOINT")
    return value.strip().rstrip("/") or None if value is not None else None
