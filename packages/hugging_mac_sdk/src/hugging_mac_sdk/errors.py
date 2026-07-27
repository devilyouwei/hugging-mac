"""Stable SDK error hierarchy."""

from __future__ import annotations

from typing import Any


class HuggingMacSdkError(Exception):
    """Base class for errors that may cross the SDK boundary."""

    code = "sdk_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause


class ManifestError(HuggingMacSdkError):
    code = "manifest_error"


class RegistrationConflictError(HuggingMacSdkError):
    code = "registration_conflict"


class ResourceNotFoundError(HuggingMacSdkError):
    code = "resource_not_found"


class ResourceIntegrityError(HuggingMacSdkError):
    code = "resource_integrity_error"


class DownloadError(HuggingMacSdkError):
    code = "download_error"
    retryable = True


class UnsupportedCapabilityError(HuggingMacSdkError):
    code = "unsupported_capability"


class UnsupportedRuntimeError(HuggingMacSdkError):
    code = "unsupported_runtime"


class InsufficientResourceError(HuggingMacSdkError):
    code = "insufficient_resource"
    retryable = True


class ModelLoadError(HuggingMacSdkError):
    code = "model_load_error"
    retryable = True


class InferenceError(HuggingMacSdkError):
    code = "inference_error"
