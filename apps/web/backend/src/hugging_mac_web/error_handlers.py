"""Map platform and SDK errors to stable API envelopes."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from hugging_mac_sdk.errors import (
    DownloadError,
    HuggingMacSdkError,
    InferenceError,
    InsufficientResourceError,
    ModelLoadError,
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedCapabilityError,
    UnsupportedRuntimeError,
)

from hugging_mac_web.schemas import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)

_SDK_STATUS: tuple[tuple[type[HuggingMacSdkError], HTTPStatus], ...] = (
    (DownloadError, HTTPStatus.SERVICE_UNAVAILABLE),
    (ResourceNotFoundError, HTTPStatus.NOT_FOUND),
    (ResourceIntegrityError, HTTPStatus.UNPROCESSABLE_ENTITY),
    (UnsupportedCapabilityError, HTTPStatus.CONFLICT),
    (UnsupportedRuntimeError, HTTPStatus.CONFLICT),
    (InsufficientResourceError, HTTPStatus.SERVICE_UNAVAILABLE),
    (ModelLoadError, HTTPStatus.SERVICE_UNAVAILABLE),
    (InferenceError, HTTPStatus.INTERNAL_SERVER_ERROR),
)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HuggingMacSdkError, _sdk_error_handler)
    app.add_exception_handler(HTTPException, _http_error_handler)


async def _sdk_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    assert isinstance(error, HuggingMacSdkError)
    logger.error(
        "SDK request failed method=%s path=%s trace_id=%s code=%s details=%r",
        request.method,
        request.url.path,
        getattr(request.state, "trace_id", None),
        error.code,
        error.details,
        exc_info=(type(error), error, error.__traceback__),
    )
    status = next(
        (value for error_type, value in _SDK_STATUS if isinstance(error, error_type)),
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    details = _safe_details(error.details)
    response = ErrorResponse(
        error=ErrorBody(
            code=error.code,
            message=error.message,
            retryable=error.retryable,
            trace_id=getattr(request.state, "trace_id", None),
            details=details,
        )
    )
    return JSONResponse(status_code=status, content=response.model_dump(mode="json"))


async def _http_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    assert isinstance(error, HTTPException)
    if error.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        logger.error(
            "HTTP request failed method=%s path=%s trace_id=%s status=%s",
            request.method,
            request.url.path,
            getattr(request.state, "trace_id", None),
            error.status_code,
            exc_info=(type(error), error, error.__traceback__),
        )
    message = str(error.detail)
    response = ErrorResponse(
        error=ErrorBody(
            code="http_error",
            message=message,
            trace_id=getattr(request.state, "trace_id", None),
        )
    )
    return JSONResponse(status_code=error.status_code, content=response.model_dump(mode="json"))


def _safe_details(details: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "capability",
        "artifact_id",
        "declared",
        "instance_count",
        "model_id",
        "revision",
        "reason",
        "runtime",
        "source_format",
        "target_format",
    }
    return {key: value for key, value in details.items() if key in allowed}
