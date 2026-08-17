"""Source download, conversion, status, and deletion for py-feat RetinaFace."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Mapping
from pathlib import Path

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import (
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size, file_sha256
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import (
    RETINAFACE_CONFIG_SHA256,
    RETINAFACE_MODEL_ID,
    RETINAFACE_REVISION,
    RETINAFACE_WEIGHTS_SHA256,
    RetinaFaceInstanceConfig,
)
from .converter import RetinaFaceConverter


class RetinaFaceResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: RetinaFaceInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
        converter: ModelConverter | None = None,
    ) -> None:
        self._source, self._config = source, config
        self._downloader = downloader or ResourceDownloader()
        self._converter = converter or RetinaFaceConverter()

    @property
    def model_root(self) -> Path:
        return self._config.model_home.joinpath(
            "py-feat", "retinaface", RETINAFACE_REVISION, self._config.variant
        )

    @property
    def source_path(self) -> Path:
        return self._config.source_path or self.model_root / "source" / "model"

    @property
    def coreml_path(self) -> Path:
        return self._config.artifact_path or self.model_root / "coreml" / "retinaface.mlpackage"

    async def resolve_source(self) -> ResolvedResource:
        required = {
            "config.json": RETINAFACE_CONFIG_SHA256,
            "mobilenet0.25_Final.pth": RETINAFACE_WEIGHTS_SHA256,
        }
        missing = [name for name in required if not (self.source_path / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "RetinaFace source is not downloaded", details={"missing": missing}
            )
        for name, expected in required.items():
            actual = file_sha256(self.source_path / name)
            if actual != expected:
                raise ResourceIntegrityError(
                    f"RetinaFace source SHA-256 mismatch: {name}",
                    details={"expected": expected, "actual": actual},
                )
        return ResolvedResource(
            path=self.source_path,
            source=self._source,
            digest=directory_sha256(self.source_path),
            size_bytes=directory_size(self.source_path),
        )

    async def resolve_coreml(self) -> ResolvedResource:
        if not (self.coreml_path / "Manifest.json").is_file():
            raise ResourceNotFoundError("RetinaFace Core ML artifact has not been converted")
        return ResolvedResource(
            path=self.coreml_path,
            source=self._source,
            digest=directory_sha256(self.coreml_path),
            size_bytes=directory_size(self.coreml_path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        if overwrite or not self.source_path.exists():
            await self._downloader.download(
                self._source, self.source_path, overwrite=overwrite, token=self._config.hf_token
            )
        return await self.resolve_source()

    async def convert_coreml(self, *, overwrite: bool = False) -> None:
        if self.coreml_path.exists() and not overwrite:
            await self.resolve_coreml()
            return
        source = await self.resolve_source()
        await self._converter.convert(
            ConversionRequest(
                model_id=RETINAFACE_MODEL_ID,
                model_revision=RETINAFACE_REVISION,
                variant=self._config.variant,
                source=source,
                source_format=ArtifactFormat.PYTORCH,
                target_format=ArtifactFormat.COREML,
                output_path=self.coreml_path,
                overwrite=overwrite,
                options={"input_size": self._config.input_size},
            )
        )

    def status(self) -> ModelResourceStatus:
        source_ok = all(
            (self.source_path / name).is_file()
            for name in ("config.json", "mobilenet0.25_Final.pth")
        )
        coreml_ok = (self.coreml_path / "Manifest.json").is_file()
        return ModelResourceStatus(
            model_id=RETINAFACE_MODEL_ID,
            revision=RETINAFACE_REVISION,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="source",
                    format="pytorch",
                    runtime="pytorch-mps",
                    available=source_ok,
                    size_bytes=directory_size(self.source_path) if source_ok else None,
                ),
                ModelArtifactStatus(
                    artifact_id="coreml-fp16",
                    format="coreml",
                    runtime="coreml",
                    provisioning="convert",
                    available=coreml_ok,
                    size_bytes=directory_size(self.coreml_path) if coreml_ok else None,
                ),
            ),
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        targets = {
            None: (self.model_root,),
            "pytorch-mps": (self.source_path,),
            "coreml": (self.coreml_path,),
        }
        if runtime not in targets:
            raise UnsupportedRuntimeError(f"Unsupported runtime: {runtime}")
        model_root = self._config.model_home.joinpath("py-feat", "retinaface").resolve(strict=False)
        for target in targets[runtime]:
            resolved = target.resolve(strict=False)
            if resolved == model_root or not resolved.is_relative_to(model_root):
                raise ResourceIntegrityError(
                    "Refusing to delete RetinaFace resources outside model root"
                )
            await asyncio.to_thread(shutil.rmtree, resolved, True)


class RetinaFaceResourceProvider:
    def __init__(self, source: HuggingFaceSource) -> None:
        self._source = source

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> RetinaFaceResourceResolver:
        return RetinaFaceResourceResolver(
            self._source,
            RetinaFaceInstanceConfig.model_validate(dict(options or {}) | {"variant": variant}),
        )

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        return self._resolver(variant, options).status()

    async def download_source(
        self, variant: str, options: Mapping[str, object] | None = None, *, overwrite: bool = False
    ) -> ModelResourceStatus:
        resolver = self._resolver(variant, options)
        await resolver.download_source(overwrite=overwrite)
        return resolver.status()

    async def convert(
        self,
        variant: str,
        target_format: ArtifactFormat,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        if target_format is not ArtifactFormat.COREML:
            raise UnsupportedRuntimeError(f"RetinaFace cannot convert to {target_format}")
        resolver = self._resolver(variant, options)
        await resolver.convert_coreml(overwrite=overwrite)
        return resolver.status()

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
