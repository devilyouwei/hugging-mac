"""Resource lifecycle for PP-DocLayoutV3 source and converted artifacts."""

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
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import PPDocLayoutV3InstanceConfig
from .converter import PPDocLayoutV3Converter


class PPDocLayoutV3ResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: PPDocLayoutV3InstanceConfig,
        manifest: ModelManifest,
        source_artifact: ModelArtifact,
        coreml_artifact: ModelArtifact,
        *,
        downloader: ResourceDownloader | None = None,
        converter: ModelConverter | None = None,
    ) -> None:
        self._source, self._config, self._manifest = source, config, manifest
        self._source_artifact, self._coreml_artifact = source_artifact, coreml_artifact
        self._downloader = downloader or ResourceDownloader()
        self._converter = converter or PPDocLayoutV3Converter(manifest.model_id)

    @property
    def source_path(self) -> Path:
        return self._config.source_path or self._source_artifact.resolve(self._config.model_home)

    @property
    def coreml_path(self) -> Path:
        return self._config.artifact_path or self._coreml_artifact.resolve(self._config.model_home)

    async def resolve_source(self) -> ResolvedResource:
        missing = [
            str(name)
            for name in self._source_artifact.required_files
            if not (self.source_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "PP-DocLayoutV3 source is not downloaded", details={"missing": missing}
            )
        return ResolvedResource(
            path=self.source_path,
            source=self._source,
            digest=directory_sha256(self.source_path),
            size_bytes=directory_size(self.source_path),
        )

    async def resolve_coreml(self) -> ResolvedResource:
        missing = [
            str(name)
            for name in self._coreml_artifact.required_files
            if not (self.coreml_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "PP-DocLayoutV3 Core ML artifact has not been converted",
                details={"missing": missing},
            )
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
                model_id=self._manifest.model_id,
                model_revision=self._manifest.revision,
                variant=self._config.variant,
                source=source,
                source_format=ArtifactFormat.SAFETENSORS,
                target_format=ArtifactFormat.COREML,
                output_path=self.coreml_path,
                overwrite=overwrite,
                options={},
            )
        )

    def status(self) -> ModelResourceStatus:
        source_ok = all(
            (self.source_path / name).is_file() for name in self._source_artifact.required_files
        )
        coreml_ok = all(
            (self.coreml_path / name).is_file() for name in self._coreml_artifact.required_files
        )
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="source",
                    format="safetensors",
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
        model_root = self._source_artifact.storage_path(self._config.model_home).parents[2]
        targets = {
            None: (model_root,),
            "pytorch-mps": (self.source_path,),
            "coreml": (self.coreml_path,),
        }
        if runtime not in targets:
            raise UnsupportedRuntimeError(f"Unsupported runtime: {runtime}")
        storage_root = self._config.model_home.expanduser().resolve(strict=False)
        for target in targets[runtime]:
            resolved = target.resolve(strict=False)
            if resolved == storage_root or not resolved.is_relative_to(storage_root):
                raise ResourceIntegrityError(
                    "Refusing to delete PP-DocLayoutV3 resources outside model storage"
                )
            await asyncio.to_thread(shutil.rmtree, resolved, True)


class PPDocLayoutV3ResourceProvider:
    def __init__(
        self,
        source: HuggingFaceSource,
        manifest: ModelManifest,
        source_artifact: ModelArtifact,
        coreml_artifact: ModelArtifact,
    ) -> None:
        self._source, self._manifest = source, manifest
        self._source_artifact, self._coreml_artifact = source_artifact, coreml_artifact

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> PPDocLayoutV3ResourceResolver:
        return PPDocLayoutV3ResourceResolver(
            self._source,
            PPDocLayoutV3InstanceConfig.model_validate(dict(options or {}) | {"variant": variant}),
            self._manifest,
            self._source_artifact,
            self._coreml_artifact,
        )

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

    async def convert(
        self,
        variant: str,
        target_format: ArtifactFormat,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        if target_format is not ArtifactFormat.COREML:
            raise UnsupportedRuntimeError(f"PP-DocLayoutV3 cannot convert to {target_format}")
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
