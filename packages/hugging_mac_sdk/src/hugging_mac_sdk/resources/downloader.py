"""Atomic downloads for Hugging Face and arbitrary HTTP(S) model resources."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx
from huggingface_hub import hf_hub_download, snapshot_download

from hugging_mac_sdk.errors import (
    DownloadError,
    ResourceIntegrityError,
    ResourceNotFoundError,
)
from hugging_mac_sdk.resources.archives import extract_archive
from hugging_mac_sdk.resources.hashing import (
    directory_sha256,
    directory_size,
    file_sha256,
)
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ResolvedResource,
    ResourceSource,
    UrlArchiveSource,
    UrlFileSource,
)


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    downloaded_bytes: int
    total_bytes: int | None


type ProgressCallback = Callable[[DownloadProgress], None]


class ResourceDownloader:
    """Download resources into an exact destination.

    Files and directories are first materialized beside the destination under a
    unique staging name. They only become visible at the destination after
    validation succeeds.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._client = client
        self._timeout = timeout

    async def download(
        self,
        source: ResourceSource,
        destination: Path,
        *,
        overwrite: bool = False,
        token: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> ResolvedResource:
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._assert_destination_available(destination, overwrite)

        if isinstance(source, HuggingFaceSource):
            if source.filename:
                return await self._download_huggingface_file(
                    source,
                    destination,
                    overwrite=overwrite,
                    token=token,
                )
            return await self._download_huggingface_snapshot(
                source,
                destination,
                overwrite=overwrite,
                token=token,
            )
        if isinstance(source, UrlFileSource):
            return await self._download_url_file(
                source,
                destination,
                overwrite=overwrite,
                progress=progress,
            )
        if isinstance(source, UrlArchiveSource):
            return await self._download_url_archive(
                source,
                destination,
                overwrite=overwrite,
                progress=progress,
            )
        raise TypeError(f"Unsupported resource source: {type(source)!r}")

    async def _download_url_file(
        self,
        source: UrlFileSource,
        destination: Path,
        *,
        overwrite: bool,
        progress: ProgressCallback | None,
    ) -> ResolvedResource:
        staging = _staging_path(destination)
        try:
            await self._stream_url(str(source.url), staging, progress)
            digest = self._validate_file(staging, source.expected_sha256)
            self._commit(staging, destination, overwrite)
            return ResolvedResource(
                path=destination,
                source=source,
                digest=digest,
                size_bytes=destination.stat().st_size,
            )
        except (ResourceIntegrityError, ResourceNotFoundError, DownloadError):
            raise
        except Exception as error:
            raise DownloadError(
                f"Failed to download {source.url}",
                cause=error,
            ) from error
        finally:
            _remove_staging(staging)

    async def _download_url_archive(
        self,
        source: UrlArchiveSource,
        destination: Path,
        *,
        overwrite: bool,
        progress: ProgressCallback | None,
    ) -> ResolvedResource:
        staging_archive = _staging_path(destination, suffix=".archive")
        staging_directory = _staging_path(destination, suffix=".directory")
        try:
            await self._stream_url(str(source.url), staging_archive, progress)
            self._validate_file(staging_archive, source.expected_sha256)
            await asyncio.to_thread(
                extract_archive,
                staging_archive,
                staging_directory,
                archive_format=source.format,
                strip_components=source.strip_components,
            )
            digest = await asyncio.to_thread(directory_sha256, staging_directory)
            size = directory_size(staging_directory)
            self._commit(staging_directory, destination, overwrite)
            return ResolvedResource(
                path=destination,
                source=source,
                digest=digest,
                size_bytes=size,
            )
        except (ResourceIntegrityError, ResourceNotFoundError, DownloadError):
            raise
        except Exception as error:
            raise DownloadError(
                f"Failed to download model directory from {source.url}",
                cause=error,
            ) from error
        finally:
            _remove_staging(staging_archive)
            _remove_staging(staging_directory)

    async def _download_huggingface_file(
        self,
        source: HuggingFaceSource,
        destination: Path,
        *,
        overwrite: bool,
        token: str | None,
    ) -> ResolvedResource:
        staging = _staging_path(destination)
        try:
            cached_path = await asyncio.to_thread(
                partial(
                    hf_hub_download,
                    repo_id=source.repo_id,
                    filename=source.filename,
                    revision=source.revision,
                    token=token,
                )
            )
            await asyncio.to_thread(shutil.copyfile, cached_path, staging)
            digest = self._validate_file(staging, source.expected_sha256)
            self._commit(staging, destination, overwrite)
            return ResolvedResource(
                path=destination,
                source=source,
                digest=digest,
                size_bytes=destination.stat().st_size,
            )
        except ResourceIntegrityError:
            raise
        except Exception as error:
            raise DownloadError(
                f"Failed to download {source.repo_id}/{source.filename}",
                cause=error,
            ) from error
        finally:
            _remove_staging(staging)

    async def _download_huggingface_snapshot(
        self,
        source: HuggingFaceSource,
        destination: Path,
        *,
        overwrite: bool,
        token: str | None,
    ) -> ResolvedResource:
        staging = _staging_path(destination)
        try:
            await asyncio.to_thread(
                partial(
                    snapshot_download,
                    repo_id=source.repo_id,
                    revision=source.revision,
                    local_dir=staging,
                    allow_patterns=list(source.allow_patterns) or None,
                    ignore_patterns=list(source.ignore_patterns) or None,
                    token=token,
                )
            )
            # ``local_dir`` snapshots contain Hub bookkeeping under ``.cache``.
            # The atomic destination is a model artifact, not another cache, so
            # keep only files selected by the source declaration.
            await asyncio.to_thread(_remove_huggingface_metadata, staging)
            digest = await asyncio.to_thread(directory_sha256, staging)
            self._validate_digest(digest, source.expected_sha256, source.repo_id)
            size = directory_size(staging)
            self._commit(staging, destination, overwrite)
            return ResolvedResource(
                path=destination,
                source=source,
                digest=digest,
                size_bytes=size,
            )
        except ResourceIntegrityError:
            raise
        except Exception as error:
            raise DownloadError(
                f"Failed to download Hugging Face snapshot {source.repo_id}",
                cause=error,
            ) from error
        finally:
            _remove_staging(staging)

    async def _stream_url(
        self,
        url: str,
        destination: Path,
        progress: ProgressCallback | None,
    ) -> None:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            follow_redirects=True,
            timeout=self._timeout,
        )
        try:
            async with client.stream("GET", url) as response:
                if response.status_code == 404:
                    raise ResourceNotFoundError(f"Resource not found: {url}")
                response.raise_for_status()
                total = _content_length(response)
                downloaded = 0
                with destination.open("xb") as output:
                    async for chunk in response.aiter_bytes():
                        output.write(chunk)
                        downloaded += len(chunk)
                        if progress is not None:
                            progress(DownloadProgress(downloaded, total))
        except ResourceNotFoundError:
            raise
        except httpx.HTTPError as error:
            raise DownloadError(f"HTTP download failed: {url}", cause=error) from error
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def _validate_file(path: Path, expected_sha256: str | None) -> str:
        digest = file_sha256(path)
        ResourceDownloader._validate_digest(digest, expected_sha256, path.name)
        return digest

    @staticmethod
    def _validate_digest(
        digest: str,
        expected_sha256: str | None,
        resource_name: str,
    ) -> None:
        if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
            raise ResourceIntegrityError(
                f"SHA-256 mismatch for {resource_name}",
                details={"expected": expected_sha256.lower(), "actual": digest},
            )

    @staticmethod
    def _assert_destination_available(destination: Path, overwrite: bool) -> None:
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")

    @staticmethod
    def _commit(staging: Path, destination: Path, overwrite: bool) -> None:
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {destination}")
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(staging, destination)


def _staging_path(destination: Path, *, suffix: str = "") -> Path:
    return destination.parent / f".{destination.name}.partial-{uuid4().hex}{suffix}"


def _remove_staging(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _remove_huggingface_metadata(snapshot: Path) -> None:
    metadata = snapshot / ".cache" / "huggingface"
    if metadata.is_dir():
        shutil.rmtree(metadata)
    cache = snapshot / ".cache"
    with contextlib.suppress(OSError):
        cache.rmdir()


def _content_length(response: httpx.Response) -> int | None:
    value = response.headers.get("content-length")
    if value is None:
        return None
    with contextlib.suppress(ValueError):
        return int(value)
    return None
