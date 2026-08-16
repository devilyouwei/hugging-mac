"""Download and resolve the official PyTorch and precompiled Core ML artifacts."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

from hugging_mac_sdk.errors import (
    DownloadError,
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size, file_sha256
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import (
    KOKORO_82M_MODEL_ID,
    KOKORO_82M_REQUIRED_FILES,
    KOKORO_82M_REVISION,
    KOKORO_82M_VARIANT,
    KOKORO_82M_WEIGHT_SHA256,
    Kokoro82mInstanceConfig,
)

_COREML_REQUIRED_FILES = (
    "kokoro_5s.mlmodelc/model.mil",
    "kokoro_5s.mlmodelc/weights/weight.bin",
    "G2PEncoder.mlmodelc/model.mil",
    "G2PDecoder.mlmodelc/model.mil",
    "vocab_index.json",
    "voices/af_heart.json",
)


class Kokoro82mResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        coreml_source: HuggingFaceSource,
        config: Kokoro82mInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._coreml_source = coreml_source
        self._config = config
        # This repository contains several precompiled bundles. Serial Hub
        # materialization avoids a huggingface_hub local-dir race when moving
        # the large LFS/Xet weight into its final nested path.
        self._downloader = downloader or ResourceDownloader(timeout=1800, max_workers=1)

    async def resolve_source(self) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        digest = await asyncio.to_thread(self._validate_snapshot, path)
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=digest,
            size_bytes=directory_size(path),
        )

    async def resolve_coreml(self) -> ResolvedResource:
        path = self._config.artifact_path or self._default_coreml_path()
        digest = await asyncio.to_thread(self._validate_coreml_artifact, path)
        return ResolvedResource(
            path=path,
            source=self._coreml_source,
            digest=digest,
            size_bytes=directory_size(path),
        )

    async def download_source(self, *, overwrite: bool = False) -> tuple[ResolvedResource, ...]:
        """Download both declared runtime artifacts from their pinned repositories."""

        pytorch = await self._download_or_resolve(
            self._source,
            self._config.source_path or self._default_source_path(),
            self.resolve_source,
            overwrite=overwrite,
        )
        coreml = await self._download_or_resolve(
            self._coreml_source,
            self._config.artifact_path or self._default_coreml_path(),
            self.resolve_coreml,
            overwrite=overwrite,
        )
        return pytorch, coreml

    async def _download_or_resolve(
        self,
        source: HuggingFaceSource,
        path: Path,
        resolver: Callable[[], Awaitable[ResolvedResource]],
        *,
        overwrite: bool,
    ) -> ResolvedResource:
        if path.exists() and not overwrite:
            return await resolver()
        try:
            await self._downloader.download(
                source,
                path,
                overwrite=overwrite,
                token=self._config.hf_token,
            )
        except DownloadError:
            # Another request may have atomically published the same pinned
            # artifact while this request was downloading its staging copy.
            if path.exists() and not overwrite:
                return await resolver()
            raise
        return await resolver()

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "pytorch-mps", "coreml"}:
            raise UnsupportedRuntimeError(
                f"Kokoro-82M does not support runtime {runtime}",
                details={"model_id": KOKORO_82M_MODEL_ID, "runtime": runtime},
            )
        root = self._model_root().expanduser().resolve(strict=False)
        if runtime is None:
            target = root
        elif runtime == "pytorch-mps":
            target = self._config.source_path or self._default_source_path()
        else:
            target = self._config.artifact_path or self._default_coreml_path()
        resolved = target.expanduser().resolve(strict=False)
        if resolved != root and not resolved.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Kokoro-82M resources outside the model directory",
                details={"model_id": KOKORO_82M_MODEL_ID},
            )
        await asyncio.to_thread(self._delete_path, resolved)

    def status(self) -> ModelResourceStatus:
        source = self._config.source_path or self._default_source_path()
        coreml = self._config.artifact_path or self._default_coreml_path()
        source_available = self._snapshot_files_exist(source)
        coreml_available = self._coreml_files_exist(coreml)
        return ModelResourceStatus(
            model_id=KOKORO_82M_MODEL_ID,
            revision=KOKORO_82M_REVISION,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="source",
                    format="pytorch",
                    runtime="pytorch-mps",
                    available=source_available,
                    size_bytes=directory_size(source) if source_available else None,
                ),
                ModelArtifactStatus(
                    artifact_id="coreml",
                    format="coreml",
                    runtime="coreml",
                    available=coreml_available,
                    size_bytes=directory_size(coreml) if coreml_available else None,
                ),
            ),
        )

    def _default_source_path(self) -> Path:
        return self._model_root() / "pytorch" / "model"

    def _default_coreml_path(self) -> Path:
        return self._model_root() / "coreml" / "model"

    def _model_root(self) -> Path:
        return (
            self._config.model_home
            / "hexgrad"
            / "kokoro"
            / self._source.revision
            / self._config.variant
        )

    @staticmethod
    def _snapshot_files_exist(path: Path) -> bool:
        return path.is_dir() and all((path / name).is_file() for name in KOKORO_82M_REQUIRED_FILES)

    @staticmethod
    def _coreml_files_exist(path: Path) -> bool:
        return path.is_dir() and all((path / name).is_file() for name in _COREML_REQUIRED_FILES)

    def _validate_snapshot(self, path: Path) -> str:
        if not self._snapshot_files_exist(path):
            raise ResourceNotFoundError(
                "Kokoro-82M PyTorch artifact is not downloaded",
                details={"model_id": KOKORO_82M_MODEL_ID, "artifact_id": "source"},
            )
        digest = file_sha256(path / "kokoro-v1_0.pth")
        if digest != KOKORO_82M_WEIGHT_SHA256:
            raise ResourceIntegrityError(
                "Kokoro-82M weight SHA-256 mismatch",
                details={"expected": KOKORO_82M_WEIGHT_SHA256, "actual": digest},
            )
        return digest

    def _validate_coreml_artifact(self, path: Path) -> str:
        if not self._coreml_files_exist(path):
            raise ResourceNotFoundError(
                "Kokoro-82M Core ML artifact is not downloaded",
                details={"model_id": KOKORO_82M_MODEL_ID, "artifact_id": "coreml"},
            )
        return directory_sha256(path)

    @staticmethod
    def _delete_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


class Kokoro82mResourceProvider:
    def __init__(self, source: HuggingFaceSource, coreml_source: HuggingFaceSource) -> None:
        self._source = source
        self._coreml_source = coreml_source

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
        await resolver.download_source(overwrite=overwrite)
        return resolver.status()

    async def convert(self, *args: object, **kwargs: object) -> ModelResourceStatus:
        raise UnsupportedRuntimeError(
            "Kokoro-82M Core ML is distributed as a downloadable artifact; conversion is disabled"
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
    ) -> Kokoro82mResourceResolver:
        if variant != KOKORO_82M_VARIANT:
            raise ResourceNotFoundError(
                f"Kokoro-82M variant is not registered: {variant}",
                details={"supported_variants": [KOKORO_82M_VARIANT]},
            )
        config = Kokoro82mInstanceConfig.model_validate(dict(options or {}) | {"variant": variant})
        return Kokoro82mResourceResolver(
            self._source,
            self._coreml_source,
            config,
        )
