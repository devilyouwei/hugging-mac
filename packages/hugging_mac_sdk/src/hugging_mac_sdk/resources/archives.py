"""Safe extraction for model directory archives."""

from __future__ import annotations

import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from hugging_mac_sdk.errors import DownloadError


def extract_archive(
    archive: Path,
    destination: Path,
    *,
    archive_format: str = "auto",
    strip_components: int = 0,
) -> None:
    resolved_format = _detect_format(archive, archive_format)
    destination.mkdir(parents=True, exist_ok=False)

    if resolved_format == "zip":
        _extract_zip(archive, destination, strip_components)
    else:
        _extract_tar(archive, destination, strip_components)


def _detect_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if zipfile.is_zipfile(path):
        return "zip"
    if tarfile.is_tarfile(path):
        return "tar"
    raise DownloadError(f"Unsupported or invalid archive: {path.name}")


def _safe_relative_path(name: str, strip_components: int) -> Path | None:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise DownloadError(f"Archive member has an unsafe path: {name}")

    parts = tuple(part for part in normalized.parts if part not in {"", "."})
    if len(parts) <= strip_components:
        return None
    return Path(*parts[strip_components:])


def _extract_zip(archive: Path, destination: Path, strip_components: int) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = _safe_relative_path(member.filename, strip_components)
            if relative is None:
                continue

            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise DownloadError(f"Archive symlinks are not allowed: {member.filename}")

            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _extract_tar(archive: Path, destination: Path, strip_components: int) -> None:
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            relative = _safe_relative_path(member.name, strip_components)
            if relative is None:
                continue
            if member.issym() or member.islnk():
                raise DownloadError(f"Archive links are not allowed: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise DownloadError(f"Unsupported archive member: {member.name}")

            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise DownloadError(f"Could not read archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
