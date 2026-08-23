"""Audio8-TTS resource resolution, download, status, and deletion."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from hugging_mac_sdk.errors import (
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import Audio8TtsInstanceConfig
from .mlx_resources import Audio8TtsMlxResourceProvider


class Audio8TtsResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Audio8TtsInstanceConfig,
        *,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        tokenizer_artifact: ModelArtifact,
        tokenizer_source: HuggingFaceSource | None = None,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._manifest = manifest
        self._artifact = artifact
        self._tokenizer_artifact = tokenizer_artifact
        self._tokenizer_source = tokenizer_source or source
        self._config = config
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    async def resolve_source(self) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        digest = await asyncio.to_thread(self._validate_snapshot, path)
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=digest,
            size_bytes=directory_size(path),
        )

    @property
    def tokenizer_path(self) -> Path:
        return self._config.tokenizer_path or self._tokenizer_artifact.resolve(
            self._config.model_home
        )

    async def resolve_tokenizer(self) -> ResolvedResource:
        missing = [
            name
            for name in self._tokenizer_artifact.required_files
            if not (self.tokenizer_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Audio8-TTS shared tokenizer is incomplete",
                details={"model_id": self._manifest.model_id, "missing": missing},
            )
        return ResolvedResource(
            path=self.tokenizer_path,
            source=self._tokenizer_source,
            size_bytes=directory_size(self.tokenizer_path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        if overwrite or not path.exists():
            await self._downloader.download(
                self._source,
                path,
                overwrite=overwrite,
                token=self._config.hf_token,
            )
        if overwrite or not self.tokenizer_path.exists():
            await self._downloader.download(
                self._tokenizer_source,
                self.tokenizer_path,
                overwrite=overwrite,
                token=self._config.hf_token,
            )
        await self.resolve_tokenizer()
        return await self.resolve_source()

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, self._config.runtime}:
            raise UnsupportedRuntimeError(
                f"Audio8-TTS does not support runtime {runtime}",
                details={"model_id": self._manifest.model_id, "runtime": runtime},
            )
        root = self._model_root().expanduser().resolve(strict=False)
        target = (
            root if runtime is None else (self._config.source_path or self._default_source_path())
        )
        resolved = target.expanduser().resolve(strict=False)
        if resolved != root and not resolved.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Audio8-TTS resources outside the model directory",
                details={"model_id": self._manifest.model_id},
            )
        await asyncio.to_thread(self._delete_path, resolved)

    def status(self) -> ModelResourceStatus:
        path = self._config.source_path or self._default_source_path()
        available = self._snapshot_files_exist(path)
        tokenizer_available = all(
            (self.tokenizer_path / name).is_file()
            for name in self._tokenizer_artifact.required_files
        )
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._source.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="source",
                    format=ArtifactFormat.SAFETENSORS.value,
                    runtime=self._artifact.runtime,
                    required_shares=self._artifact.required_shares,
                    available=available,
                    size_bytes=directory_size(path) if available else None,
                ),
                ModelArtifactStatus(
                    artifact_id=self._tokenizer_artifact.artifact_id,
                    format=ArtifactFormat.TOKENIZER.value,
                    runtime=None,
                    shared=True,
                    available=tokenizer_available,
                    size_bytes=(
                        directory_size(self.tokenizer_path) if tokenizer_available else None
                    ),
                ),
            ),
        )

    def _default_source_path(self) -> Path:
        return self._artifact.resolve(self._config.model_home)

    def _model_root(self) -> Path:
        return self._artifact.storage_path(self._config.model_home).parents[1]

    def _snapshot_files_exist(self, path: Path) -> bool:
        return path.is_dir() and all(
            (path / name).is_file() for name in self._artifact.required_files
        )

    def _validate_snapshot(self, path: Path) -> str:
        if not path.is_dir():
            raise ResourceNotFoundError(
                "Audio8-TTS source model is not downloaded",
                details={"model_id": self._manifest.model_id, "artifact_id": "source"},
            )
        missing = [name for name in self._artifact.required_files if not (path / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "Audio8-TTS snapshot is incomplete",
                details={"model_id": self._manifest.model_id, "missing": missing},
            )
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "arktts":
            raise ResourceIntegrityError(
                "Audio8-TTS config has an unexpected model_type",
                details={"model_type": config.get("model_type")},
            )
        return directory_sha256(path)

    @staticmethod
    def _delete_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


class Audio8TtsResourceProvider:
    def __init__(
        self,
        manifest: ModelManifest,
        artifacts: Mapping[str, ModelArtifact],
        tokenizer_artifacts: Mapping[str, ModelArtifact],
    ) -> None:
        self._manifest = manifest
        self._artifacts = dict(artifacts)
        self._sources = {
            variant: artifact.source
            for variant, artifact in self._artifacts.items()
            if isinstance(artifact.source, HuggingFaceSource)
        }
        self._tokenizer_artifacts = dict(tokenizer_artifacts)

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
        del options, overwrite
        raise UnsupportedRuntimeError(
            f"Audio8-TTS conversion to {target_format} is not implemented",
            details={"model_id": self._manifest.model_id, "variant": variant},
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
    ) -> Audio8TtsResourceResolver:
        if variant not in self._sources:
            raise ResourceNotFoundError(
                f"Audio8-TTS variant is not registered: {variant}",
                details={"supported_variants": sorted(self._sources)},
            )
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError(
                "Conflicting Audio8-TTS variant values were provided",
                details={"variant": variant, "options_variant": option_variant},
            )
        config = Audio8TtsInstanceConfig.model_validate(normalized | {"variant": variant})
        tokenizer_artifact = self._tokenizer_artifacts[variant]
        tokenizer_source = tokenizer_artifact.source
        if not isinstance(tokenizer_source, HuggingFaceSource):
            raise TypeError(f"Audio8-TTS variant {variant} needs a tokenizer source")
        return Audio8TtsResourceResolver(
            self._sources[variant],
            config,
            manifest=self._manifest,
            artifact=self._artifacts[variant],
            tokenizer_artifact=tokenizer_artifact,
            tokenizer_source=tokenizer_source,
        )


class Audio8TtsCombinedResourceProvider:
    """Expose all downloadable runtime artifacts through one canonical model."""

    def __init__(
        self,
        manifest: ModelManifest,
        pytorch_artifacts: Mapping[str, ModelArtifact],
        tokenizer_artifacts: Mapping[str, ModelArtifact],
        mlx_source: HuggingFaceSource,
        mlx_artifact: ModelArtifact,
    ) -> None:
        self._pytorch = Audio8TtsResourceProvider(manifest, pytorch_artifacts, tokenizer_artifacts)
        assert mlx_artifact.variant is not None
        self._mlx_variant = mlx_artifact.variant
        mlx_tokenizer_artifact = tokenizer_artifacts[self._mlx_variant]
        mlx_tokenizer_source = mlx_tokenizer_artifact.source
        if not isinstance(mlx_tokenizer_source, HuggingFaceSource):
            raise TypeError("Audio8-TTS MLX needs a tokenizer source")
        self._mlx = Audio8TtsMlxResourceProvider(
            mlx_source,
            mlx_tokenizer_source,
            manifest=manifest,
            artifact=mlx_artifact,
            tokenizer_artifact=mlx_tokenizer_artifact,
        )

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        pytorch = await self._pytorch.status(variant, options)
        artifacts = pytorch.artifacts
        if variant == self._mlx_variant:
            mlx = await self._mlx.status(variant, options)
            artifacts += mlx.artifacts
        # Reconstruct the status so ModelResourceStatus recomputes its derived
        # runtime aggregates for both backends. model_copy(update=...) skips
        # validation and would leave only the PyTorch runtime visible.
        return ModelResourceStatus(
            model_id=pytorch.model_id,
            revision=pytorch.revision,
            variant=pytorch.variant,
            artifacts=artifacts,
        )

    async def download_source(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        await self._pytorch.download_source(variant, options, overwrite=overwrite)
        if variant == self._mlx_variant:
            await self._mlx.download_source(variant, options, overwrite=overwrite)
        return await self.status(variant, options)

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
            f"Audio8-TTS conversion to {target_format} is not implemented"
        )

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        if runtime in {None, "pytorch"}:
            await self._pytorch.delete(variant, options, runtime=runtime)
        if variant == self._mlx_variant and runtime in {None, "mlx"}:
            await self._mlx.delete(variant, options, runtime=runtime)
        if variant != self._mlx_variant and runtime == "mlx":
            raise UnsupportedRuntimeError(
                f"Audio8-TTS variant {variant} does not provide an MLX artifact"
            )
        if runtime not in {None, "pytorch", "mlx"}:
            raise UnsupportedRuntimeError(f"Audio8-TTS does not support runtime {runtime}")
        return await self.status(variant, options)
