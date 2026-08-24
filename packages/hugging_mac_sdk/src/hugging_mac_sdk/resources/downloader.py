"""Atomic downloads for Hugging Face and arbitrary HTTP(S) model resources."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from hugging_mac_sdk.errors import (
    DownloadError,
    ResourceNotFoundError,
)
from hugging_mac_sdk.resources.archives import extract_archive
from hugging_mac_sdk.resources.hashing import (
    directory_sha256,
    directory_size,
    file_sha256,
)
from hugging_mac_sdk.schemas.resources import (
    CompositeSource,
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
    phase: str = "downloading"


type ProgressCallback = Callable[[DownloadProgress], None]


def _report(
    progress: ProgressCallback | None,
    phase: str,
    downloaded_bytes: int = 0,
    total_bytes: int | None = None,
) -> None:
    if progress is not None:
        progress(DownloadProgress(downloaded_bytes, total_bytes, phase))


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
        max_workers: int = 8,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._max_workers = max_workers

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
                    progress=progress,
                )
            return await self._download_huggingface_snapshot(
                source,
                destination,
                overwrite=overwrite,
                token=token,
                progress=progress,
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
        if isinstance(source, CompositeSource):
            return await self._download_composite(
                source,
                destination,
                overwrite=overwrite,
                token=token,
                progress=progress,
            )
        raise TypeError(f"Unsupported resource source: {type(source)!r}")

    async def _download_composite(
        self,
        source: CompositeSource,
        destination: Path,
        *,
        overwrite: bool,
        token: str | None,
        progress: ProgressCallback | None,
    ) -> ResolvedResource:
        staging = _staging_path(destination)
        try:
            staging.mkdir(parents=True)
            completed = 0
            known_total = 0
            for resource in source.resources:
                child_total: int | None = None

                def child_progress(event: DownloadProgress, base: int = completed) -> None:
                    nonlocal child_total
                    if event.phase != "downloading":
                        return
                    child_total = event.total_bytes
                    total = base + event.total_bytes if event.total_bytes is not None else None
                    _report(progress, "downloading", base + event.downloaded_bytes, total)

                resolved = await self.download(
                    resource.source,
                    staging / resource.path,
                    token=token,
                    progress=child_progress,
                )
                completed += resolved.size_bytes
                if child_total is not None:
                    known_total += child_total
                _report(progress, "downloading", completed, known_total or None)
            _report(progress, "verifying", completed, completed)
            digest = await asyncio.to_thread(directory_sha256, staging)
            size = directory_size(staging)
            _report(progress, "installing", size, size)
            self._commit(staging, destination, overwrite)
            _report(progress, "completed", size, size)
            return ResolvedResource(
                path=destination,
                source=source,
                digest=digest,
                size_bytes=size,
            )
        except (ResourceNotFoundError, DownloadError):
            raise
        except Exception as error:
            raise DownloadError("Failed to download composite resource", cause=error) from error
        finally:
            _remove_staging(staging)

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
            size = staging.stat().st_size
            _report(progress, "verifying", size, size)
            digest = self._digest_file(staging)
            _report(progress, "installing", size, size)
            self._commit(staging, destination, overwrite)
            _report(progress, "completed", size, size)
            return ResolvedResource(
                path=destination,
                source=source,
                digest=digest,
                size_bytes=destination.stat().st_size,
            )
        except (ResourceNotFoundError, DownloadError):
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
            archive_size = staging_archive.stat().st_size
            _report(progress, "verifying", archive_size, archive_size)
            self._digest_file(staging_archive)
            _report(progress, "extracting", archive_size, archive_size)
            await asyncio.to_thread(
                extract_archive,
                staging_archive,
                staging_directory,
                archive_format=source.format,
                strip_components=source.strip_components,
            )
            digest = await asyncio.to_thread(directory_sha256, staging_directory)
            size = directory_size(staging_directory)
            _report(progress, "installing", size, size)
            self._commit(staging_directory, destination, overwrite)
            _report(progress, "completed", size, size)
            return ResolvedResource(
                path=destination,
                source=source,
                digest=digest,
                size_bytes=size,
            )
        except (ResourceNotFoundError, DownloadError):
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
        progress: ProgressCallback | None,
    ) -> ResolvedResource:
        staging = _staging_path(destination)
        try:
            tracker = _HuggingFaceProgressTracker(progress)
            cached_path = await asyncio.to_thread(
                partial(
                    hf_hub_download,
                    repo_id=source.repo_id,
                    filename=source.filename,
                    revision=source.revision,
                    token=token,
                    tqdm_class=tracker.tqdm_class,
                )
            )
            cached_size = Path(cached_path).stat().st_size
            _report(progress, "installing", cached_size, cached_size)
            await asyncio.to_thread(shutil.copyfile, cached_path, staging)
            _report(progress, "verifying", cached_size, cached_size)
            digest = self._digest_file(staging)
            self._commit(staging, destination, overwrite)
            _report(progress, "completed", cached_size, cached_size)
            return ResolvedResource(
                path=destination,
                source=source,
                digest=digest,
                size_bytes=destination.stat().st_size,
            )
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
        progress: ProgressCallback | None,
    ) -> ResolvedResource:
        staging = _staging_path(destination)
        selected = _staging_path(destination, suffix=".selected")
        try:
            tracker = _HuggingFaceProgressTracker(progress)
            await asyncio.to_thread(
                partial(
                    snapshot_download,
                    repo_id=source.repo_id,
                    revision=source.revision,
                    local_dir=staging,
                    allow_patterns=list(source.allow_patterns) or None,
                    ignore_patterns=list(source.ignore_patterns) or None,
                    token=token,
                    max_workers=self._max_workers,
                    tqdm_class=tracker.tqdm_class,
                )
            )
            _report(progress, "verifying", tracker.downloaded, tracker.total)
            # ``local_dir`` snapshots contain Hub bookkeeping under ``.cache``.
            # The atomic destination is a model artifact, not another cache, so
            # keep only files selected by the source declaration.
            await asyncio.to_thread(_remove_huggingface_metadata, staging)
            materialized = staging
            if source.strip_prefix is not None:
                prefix_path = staging / source.strip_prefix
                if not prefix_path.is_dir():
                    raise ResourceNotFoundError(
                        f"Hugging Face snapshot prefix not found: {source.strip_prefix}"
                    )
                await asyncio.to_thread(os.replace, prefix_path, selected)
                await asyncio.to_thread(_remove_staging, staging)
                materialized = selected
            digest = await asyncio.to_thread(directory_sha256, materialized)
            size = directory_size(materialized)
            _report(progress, "installing", size, size)
            self._commit(materialized, destination, overwrite)
            _report(progress, "completed", size, size)
            return ResolvedResource(
                path=destination,
                source=source,
                digest=digest,
                size_bytes=size,
            )
        except Exception as error:
            raise DownloadError(
                f"Failed to download Hugging Face snapshot {source.repo_id}",
                cause=error,
            ) from error
        finally:
            _remove_staging(staging)
            _remove_staging(selected)

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
                            _report(progress, "downloading", downloaded, total)
        except ResourceNotFoundError:
            raise
        except httpx.HTTPError as error:
            raise DownloadError(f"HTTP download failed: {url}", cause=error) from error
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def _digest_file(path: Path) -> str:
        return file_sha256(path)

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


class _HuggingFaceProgressTracker:
    """Aggregate concurrent Hugging Face byte progress bars into one callback."""

    def __init__(self, progress: ProgressCallback | None) -> None:
        self._progress = progress
        self._lock = threading.Lock()
        self._bars: dict[int, tuple[int, int | None]] = {}

        tracker = self

        class CallbackTqdm(tqdm):  # type: ignore[misc]
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                tracker._update(self)

            def update(self, n: float | None = 1) -> bool | None:
                changed = super().update(n)
                tracker._update(self)
                return bool(changed) if changed is not None else None

            def close(self) -> None:
                tracker._update(self)
                super().close()

        self.tqdm_class = CallbackTqdm

    @property
    def downloaded(self) -> int:
        with self._lock:
            return sum(item[0] for item in self._bars.values())

    @property
    def total(self) -> int | None:
        with self._lock:
            totals = [item[1] for item in self._bars.values()]
            return sum(total for total in totals if total is not None) if totals and all(
                total is not None for total in totals
            ) else None

    def _update(self, bar: tqdm) -> None:
        if self._progress is None or getattr(bar, "unit", None) != "B":
            return
        with self._lock:
            self._bars[id(bar)] = (int(bar.n), int(bar.total) if bar.total is not None else None)
            downloaded = sum(item[0] for item in self._bars.values())
            totals = [item[1] for item in self._bars.values()]
            total = (
                sum(value for value in totals if value is not None)
                if totals and all(value is not None for value in totals)
                else None
            )
        _report(self._progress, "downloading", downloaded, total)

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
