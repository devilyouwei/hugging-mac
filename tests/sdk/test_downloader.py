from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import pytest
from hugging_mac_sdk.errors import DownloadError, ResourceIntegrityError
from hugging_mac_sdk.resources import downloader as downloader_module
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    UrlArchiveSource,
    UrlFileSource,
)


def zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


async def test_downloads_url_file_atomically(tmp_path: Path) -> None:
    content = b"model weights"
    digest = hashlib.sha256(content).hexdigest()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = ResourceDownloader(client=client)
        destination = tmp_path / "model.safetensors"
        result = await downloader.download(
            UrlFileSource(
                url="https://models.example/model.safetensors",
                expected_sha256=digest,
            ),
            destination,
        )

    assert destination.read_bytes() == content
    assert result.digest == digest
    assert result.size_bytes == len(content)
    assert not tuple(tmp_path.glob("*.partial-*"))


async def test_hash_mismatch_never_commits_destination(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"corrupt", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = ResourceDownloader(client=client)
        destination = tmp_path / "model.bin"
        with pytest.raises(ResourceIntegrityError):
            await downloader.download(
                UrlFileSource(
                    url="https://models.example/model.bin",
                    expected_sha256="0" * 64,
                ),
                destination,
            )

    assert not destination.exists()


async def test_downloads_and_extracts_model_directory(tmp_path: Path) -> None:
    archive_bytes = zip_bytes(
        {
            "model/config.json": b"{}",
            "model/tokenizer.json": b'{"version": 1}',
            "model/weights/model-00001.safetensors": b"weights",
        }
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=archive_bytes, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = ResourceDownloader(client=client)
        destination = tmp_path / "llm"
        result = await downloader.download(
            UrlArchiveSource(
                url="https://models.example/llm.zip",
                strip_components=1,
            ),
            destination,
        )

    assert (destination / "config.json").read_bytes() == b"{}"
    assert (destination / "weights/model-00001.safetensors").read_bytes() == b"weights"
    assert result.size_bytes > 0


async def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive_bytes = zip_bytes({"../outside.txt": b"not allowed"})

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=archive_bytes, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = ResourceDownloader(client=client)
        destination = tmp_path / "model"
        with pytest.raises(DownloadError):
            await downloader.download(
                UrlArchiveSource(url="https://models.example/unsafe.zip"),
                destination,
            )

    assert not destination.exists()
    assert not (tmp_path.parent / "outside.txt").exists()


async def test_downloads_huggingface_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = tmp_path / "cache" / "weights.safetensors"
    cached.parent.mkdir()
    cached.write_bytes(b"hf weights")

    def fake_hf_hub_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == "org/yolo"
        assert kwargs["filename"] == "yolo-small.safetensors"
        return str(cached)

    monkeypatch.setattr(downloader_module, "hf_hub_download", fake_hf_hub_download)
    destination = tmp_path / "models" / "yolo-small.safetensors"

    result = await ResourceDownloader().download(
        HuggingFaceSource(
            repo_id="org/yolo",
            filename="yolo-small.safetensors",
        ),
        destination,
    )

    assert destination.read_bytes() == b"hf weights"
    assert result.size_bytes == len(b"hf weights")


async def test_downloads_huggingface_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**kwargs: object) -> str:
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir()
        (local_dir / "config.json").write_text("{}")
        (local_dir / "model.safetensors").write_bytes(b"weights")
        metadata = local_dir / ".cache" / "huggingface"
        metadata.mkdir(parents=True)
        (metadata / "download.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setattr(downloader_module, "snapshot_download", fake_snapshot_download)
    destination = tmp_path / "models" / "llm"

    result = await ResourceDownloader().download(
        HuggingFaceSource(repo_id="org/llm"),
        destination,
    )

    assert (destination / "config.json").read_text() == "{}"
    assert (destination / "model.safetensors").read_bytes() == b"weights"
    assert not (destination / ".cache").exists()
    assert result.digest is not None
