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
    CompositeResource,
    CompositeSource,
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


async def test_downloads_composite_resource_as_one_atomic_directory(tmp_path: Path) -> None:
    payloads = {
        "/coreml": b"compiled graph",
        "/tokenizer": b'{"tokenizer_class":"Qwen2Tokenizer"}',
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payloads[request.url.path], request=request)

    source = CompositeSource(
        resources=(
            CompositeResource(
                path="coreml/model/model.mil",
                source=UrlFileSource(url="https://models.example/coreml"),
            ),
            CompositeResource(
                path="tokenizer/tokenizer_config.json",
                source=UrlFileSource(url="https://models.example/tokenizer"),
            ),
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        destination = tmp_path / "qwen3-asr"
        result = await ResourceDownloader(client=client).download(source, destination)

    assert (destination / "coreml/model/model.mil").read_bytes() == b"compiled graph"
    assert (destination / "tokenizer/tokenizer_config.json").read_bytes() == payloads["/tokenizer"]
    assert result.size_bytes == sum(map(len, payloads.values()))
    assert result.digest is not None
    assert not tuple(tmp_path.glob("*.partial-*"))


async def test_composite_download_failure_does_not_commit_partial_bundle(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/coreml":
            return httpx.Response(200, content=b"compiled graph", request=request)
        return httpx.Response(500, content=b"failed", request=request)

    source = CompositeSource(
        resources=(
            CompositeResource(
                path="coreml/model/model.mil",
                source=UrlFileSource(url="https://models.example/coreml"),
            ),
            CompositeResource(
                path="tokenizer/tokenizer_config.json",
                source=UrlFileSource(url="https://models.example/tokenizer"),
            ),
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        destination = tmp_path / "qwen3-asr"
        with pytest.raises(DownloadError):
            await ResourceDownloader(client=client).download(source, destination)

    assert not destination.exists()
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


async def test_downloads_selected_huggingface_directory_without_repo_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**kwargs: object) -> str:
        assert kwargs["allow_patterns"] == [
            "yolov8n.mlpackage/Manifest.json",
            "yolov8n.mlpackage/Data/**",
        ]
        local_dir = Path(str(kwargs["local_dir"]))
        package = local_dir / "yolov8n.mlpackage"
        (package / "Data" / "com.apple.CoreML").mkdir(parents=True)
        (package / "Manifest.json").write_text("{}")
        (package / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(
            b"coreml"
        )
        (local_dir / "README.md").write_text("repository card")
        return str(local_dir)

    monkeypatch.setattr(downloader_module, "snapshot_download", fake_snapshot_download)
    destination = tmp_path / "models" / "yolov8n.mlpackage"

    result = await ResourceDownloader().download(
        HuggingFaceSource(
            repo_id="hugging-mac/yolov8-coreml",
            allow_patterns=(
                "yolov8n.mlpackage/Manifest.json",
                "yolov8n.mlpackage/Data/**",
            ),
            strip_prefix="yolov8n.mlpackage",
        ),
        destination,
    )

    assert (destination / "Manifest.json").read_text() == "{}"
    assert (
        destination / "Data" / "com.apple.CoreML" / "model.mlmodel"
    ).read_bytes() == b"coreml"
    assert not (destination / "yolov8n.mlpackage").exists()
    assert not (destination / "README.md").exists()
    assert result.digest is not None
    assert not tuple(tmp_path.rglob("*.partial-*"))


async def test_missing_huggingface_strip_prefix_never_commits_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**kwargs: object) -> str:
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir()
        return str(local_dir)

    monkeypatch.setattr(downloader_module, "snapshot_download", fake_snapshot_download)
    destination = tmp_path / "models" / "missing"

    with pytest.raises(DownloadError):
        await ResourceDownloader().download(
            HuggingFaceSource(
                repo_id="hugging-mac/missing",
                allow_patterns=("bundle/**",),
                strip_prefix="bundle",
            ),
            destination,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.rglob("*.partial-*"))


@pytest.mark.parametrize("prefix", ("/absolute", "../outside", "."))
def test_huggingface_strip_prefix_must_be_safe(prefix: str) -> None:
    with pytest.raises(ValueError):
        HuggingFaceSource(
            repo_id="hugging-mac/model",
            allow_patterns=("bundle/**",),
            strip_prefix=prefix,
        )


def test_huggingface_strip_prefix_rejects_files_and_unscoped_patterns() -> None:
    with pytest.raises(ValueError):
        HuggingFaceSource(
            repo_id="hugging-mac/model",
            filename="model.bin",
            allow_patterns=("bundle/**",),
            strip_prefix="bundle",
        )
    with pytest.raises(ValueError):
        HuggingFaceSource(
            repo_id="hugging-mac/model",
            allow_patterns=("README.md",),
            strip_prefix="bundle",
        )
