"""Silero prebuilt ONNX/Core ML resource lifecycle."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Mapping
from pathlib import Path

from hugging_mac_sdk.errors import (
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size, file_sha256
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
    UrlFileSource,
)

from .config import SileroInstanceConfig


class SileroResourceResolver:
    def __init__(
        self,
        onnx_source: UrlFileSource,
        coreml_source: HuggingFaceSource,
        config: SileroInstanceConfig,
        *,
        manifest: ModelManifest,
        onnx_artifact: ModelArtifact,
        coreml_artifact: ModelArtifact,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._onnx_source = onnx_source
        self._coreml_source = coreml_source
        self._config = config
        self._manifest = manifest
        self._onnx_artifact, self._coreml_artifact = onnx_artifact, coreml_artifact
        self._downloader = downloader or ResourceDownloader()

    async def resolve_onnx(self) -> ResolvedResource:
        path = self._onnx_path()
        if not path.is_file():
            raise ResourceNotFoundError("Silero ONNX model is not downloaded")
        digest = file_sha256(path)
        return ResolvedResource(
            path=path,
            source=self._onnx_source,
            digest=digest,
            size_bytes=path.stat().st_size,
        )

    async def resolve_coreml(self) -> ResolvedResource:
        path = self._coreml_path()
        missing = [
            name for name in self._coreml_artifact.required_files if not (path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Silero Core ML artifact is not downloaded",
                details={"missing": missing},
            )
        return ResolvedResource(
            path=path,
            source=self._coreml_source,
            digest=directory_sha256(path),
            size_bytes=directory_size(path),
        )

    async def download(self, *, overwrite: bool = False) -> None:
        onnx_path = self._onnx_path()
        if overwrite or not onnx_path.exists():
            await self._downloader.download(self._onnx_source, onnx_path, overwrite=overwrite)
        else:
            await self.resolve_onnx()

        coreml_path = self._coreml_path()
        if overwrite or not coreml_path.exists():
            await self._downloader.download(self._coreml_source, coreml_path, overwrite=overwrite)
        await self.resolve_coreml()

    def status(self) -> ModelResourceStatus:
        onnx_path = self._onnx_path()
        onnx_available = onnx_path.is_file()
        coreml_path = self._coreml_path()
        coreml_available = all(
            (coreml_path / name).is_file() for name in self._coreml_artifact.required_files
        )
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="coreml",
                    format="coreml",
                    runtime="coreml",
                    available=coreml_available,
                    size_bytes=directory_size(coreml_path) if coreml_available else None,
                ),
                ModelArtifactStatus(
                    artifact_id="onnx",
                    format="onnx",
                    runtime="onnx",
                    available=onnx_available,
                    size_bytes=onnx_path.stat().st_size if onnx_available else None,
                ),
            ),
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "coreml", "onnx"}:
            raise UnsupportedRuntimeError(f"Silero does not support runtime {runtime}")
        root = self._model_root().expanduser().resolve(strict=False)
        if runtime is None:
            target = root
        elif runtime == "coreml":
            target = self._coreml_path()
        else:
            target = self._onnx_path()
        resolved = target.expanduser().resolve(strict=False)
        if resolved != root and not resolved.is_relative_to(root):
            raise ResourceIntegrityError("Refusing to delete Silero resources outside model root")
        await asyncio.to_thread(self._delete_path, resolved)

    def _onnx_path(self) -> Path:
        if self._config.runtime == "onnx" and self._config.artifact_path is not None:
            return self._config.artifact_path
        return self._onnx_artifact.resolve(self._config.model_home)

    def _coreml_path(self) -> Path:
        if self._config.runtime == "coreml" and self._config.artifact_path is not None:
            path = self._config.artifact_path
        else:
            path = self._coreml_artifact.resolve(self._config.model_home)

        # Models-page snapshot downloads preserve the repository's top-level
        # ``silero_vad.mlmodelc`` directory. Older declarations used that same
        # name as the artifact destination, producing one extra directory level.
        # Accept that already-downloaded layout instead of reporting a false
        # "not downloaded" error or forcing users to download the model again.
        nested = path / "silero_vad.mlmodelc"
        if not self._coreml_complete(path) and self._coreml_complete(nested):
            return nested
        return path

    def _coreml_complete(self, path: Path) -> bool:
        return all((path / name).is_file() for name in self._coreml_artifact.required_files)

    def _model_root(self) -> Path:
        return self._onnx_artifact.storage_path(self._config.model_home).parents[1]

    @staticmethod
    def _delete_path(path: Path) -> None:
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


class SileroResourceProvider:
    def __init__(
        self,
        onnx_source: UrlFileSource,
        coreml_source: HuggingFaceSource,
        manifest: ModelManifest,
        onnx_artifact: ModelArtifact,
        coreml_artifact: ModelArtifact,
    ) -> None:
        self._onnx_source = onnx_source
        self._coreml_source = coreml_source
        self._manifest = manifest
        self._onnx_artifact, self._coreml_artifact = onnx_artifact, coreml_artifact

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        return self._resolver(variant, options).status()

    async def download_source(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        resolver = self._resolver(variant, options)
        await resolver.download(overwrite=overwrite)
        return resolver.status()

    async def convert(
        self,
        variant: str,
        target_format: ArtifactFormat,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        del variant, options, overwrite
        raise UnsupportedRuntimeError(
            f"Silero has prebuilt artifacts and does not support conversion to {target_format}"
        )

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        resolver = self._resolver(variant, options)
        await resolver.delete(runtime=runtime)
        return resolver.status()

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> SileroResourceResolver:
        config = SileroInstanceConfig.model_validate(dict(options or {}) | {"variant": variant})
        return SileroResourceResolver(
            self._onnx_source,
            self._coreml_source,
            config,
            manifest=self._manifest,
            onnx_artifact=self._onnx_artifact,
            coreml_artifact=self._coreml_artifact,
        )
