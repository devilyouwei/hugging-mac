"""Bounded multipart upload readers."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile, status


async def read_upload_limited(file: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Upload exceeds {limit} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)
