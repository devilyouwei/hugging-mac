"""Resource management for Qwen3-TTS 0.6B Base runtimes."""

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
from hugging_mac_sdk.resources.hashing import directory_size
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import QWEN3_TTS_COREML_GRAPHS, Qwen3TtsInstanceConfig


class Qwen3TtsCoreMlResourceResolver:
    """Resolve aufklarer's six-graph Core ML pipeline and shared tokenizer."""

    def __init__(
        self,
        source: HuggingFaceSource,
        tokenizer_source: HuggingFaceSource,
        config: Qwen3TtsInstanceConfig,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        tokenizer_artifact: ModelArtifact,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source, self._tokenizer_source, self._config = source, tokenizer_source, config
        self._manifest = manifest
        self._artifact = artifact
        self._tokenizer_artifact = tokenizer_artifact
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    @property
    def path(self) -> Path:
        return self._config.coreml_path or self._artifact.resolve(self._config.model_home)

    @property
    def tokenizer_path(self) -> Path:
        return self._config.tokenizer_path or self._tokenizer_artifact.resolve(
            self._config.model_home
        )

    async def resolve_source(self) -> ResolvedResource:
        missing = [
            name for name in self._artifact.required_files if not (self.path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Qwen3-TTS Core ML bundle is incomplete", details={"missing": missing}
            )
        await asyncio.to_thread(self._validate, self.path)
        return ResolvedResource(
            path=self.path, source=self._source, size_bytes=directory_size(self.path)
        )

    async def resolve_tokenizers(self) -> ResolvedResource:
        missing = [
            name
            for name in self._tokenizer_artifact.required_files
            if not (self.tokenizer_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Qwen3-TTS shared tokenizers are incomplete",
                details={"model_id": self._manifest.model_id, "missing": missing},
            )
        return ResolvedResource(
            path=self.tokenizer_path,
            source=self._tokenizer_source,
            size_bytes=directory_size(self.tokenizer_path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        if overwrite or not self.path.exists():
            await self._downloader.download(
                self._source, self.path, overwrite=overwrite, token=self._config.hf_token
            )
        if overwrite or not self.tokenizer_path.exists():
            await self._downloader.download(
                self._tokenizer_source,
                self.tokenizer_path,
                overwrite=overwrite,
                token=self._config.hf_token,
            )
        await self.resolve_tokenizers()
        return await self.resolve_source()

    def available(self) -> bool:
        return all((self.path / name).is_file() for name in self._artifact.required_files)

    def status(self) -> ModelResourceStatus:
        available = self.available()
        tokenizer_available = all(
            (self.tokenizer_path / name).is_file()
            for name in self._tokenizer_artifact.required_files
        )
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="coreml-w8a16",
                    format="coreml",
                    runtime="coreml",
                    required_shares=self._artifact.required_shares,
                    available=available,
                    size_bytes=directory_size(self.path) if available else None,
                ),
                ModelArtifactStatus(
                    artifact_id="tokenizers",
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

    @staticmethod
    def _validate(path: Path) -> None:
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResourceIntegrityError("Qwen3-TTS Core ML config is invalid") from error
        if config.get("model_type") != "qwen3_tts_coreml":
            raise ResourceIntegrityError(
                "Qwen3-TTS Core ML config has an unsupported model_type",
                details={"model_type": config.get("model_type")},
            )
        if tuple(config.get("models", ())) != QWEN3_TTS_COREML_GRAPHS:
            raise ResourceIntegrityError("Qwen3-TTS Core ML graph declaration is invalid")

    async def delete(self) -> None:
        target = self.path.expanduser().resolve(strict=False)
        root = self._artifact.storage_path(self._config.model_home).parents[2]
        if target != root and not target.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Qwen3-TTS Core ML resources outside model root"
            )
        await asyncio.to_thread(shutil.rmtree, target, True)


class Qwen3TtsResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Qwen3TtsInstanceConfig,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        tokenizer_artifact: ModelArtifact,
        *,
        tokenizer_source: HuggingFaceSource | None = None,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._tokenizer_source = tokenizer_source or source
        self._config = config
        self._manifest = manifest
        self._artifact = artifact
        self._tokenizer_artifact = tokenizer_artifact
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    @property
    def path(self) -> Path:
        return self._config.source_path or self._artifact.resolve(self._config.model_home)

    @property
    def tokenizer_path(self) -> Path:
        return self._config.tokenizer_path or self._tokenizer_artifact.resolve(
            self._config.model_home
        )

    async def resolve_source(self) -> ResolvedResource:
        await asyncio.to_thread(self._validate, self.path)
        return ResolvedResource(
            path=self.path,
            source=self._source,
            size_bytes=directory_size(self.path),
        )

    async def resolve_tokenizers(self) -> ResolvedResource:
        missing = [
            name
            for name in self._tokenizer_artifact.required_files
            if not (self.tokenizer_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Qwen3-TTS shared tokenizers are incomplete",
                details={"model_id": self._manifest.model_id, "missing": missing},
            )
        return ResolvedResource(
            path=self.tokenizer_path,
            source=self._tokenizer_source,
            size_bytes=directory_size(self.tokenizer_path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        if overwrite or not self.path.exists():
            await self._downloader.download(
                self._source,
                self.path,
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
        await self.resolve_tokenizers()
        return await self.resolve_source()

    def status(self) -> ModelResourceStatus:
        available = self._files_exist(self.path)
        tokenizers_available = all(
            (self.tokenizer_path / name).is_file()
            for name in self._tokenizer_artifact.required_files
        )
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="mlx-4bit",
                    format=ArtifactFormat.MLX.value,
                    runtime="mlx",
                    required_shares=self._artifact.required_shares,
                    available=available,
                    size_bytes=directory_size(self.path) if available else None,
                ),
                ModelArtifactStatus(
                    artifact_id="tokenizers",
                    format=ArtifactFormat.TOKENIZER.value,
                    runtime=None,
                    shared=True,
                    available=tokenizers_available,
                    size_bytes=(
                        directory_size(self.tokenizer_path) if tokenizers_available else None
                    ),
                ),
            ),
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "mlx"}:
            raise UnsupportedRuntimeError(f"Qwen3-TTS does not support runtime {runtime}")
        root = self._model_root().expanduser().resolve(strict=False)
        target = (root if runtime is None else self.path).expanduser().resolve(strict=False)
        if target != root and not target.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Qwen3-TTS resources outside the model directory"
            )
        await asyncio.to_thread(shutil.rmtree, target, True)

    def _model_root(self) -> Path:
        return self._artifact.storage_path(self._config.model_home).parents[1]

    def _files_exist(self, path: Path) -> bool:
        return (
            path.is_dir()
            and all((path / name).is_file() for name in self._artifact.required_files)
            and any(path.glob("*.safetensors"))
        )

    def _validate(self, path: Path) -> None:
        missing = [name for name in self._artifact.required_files if not (path / name).is_file()]
        if not any(path.glob("*.safetensors")):
            missing.append(Path("*.safetensors"))
        if missing:
            raise ResourceNotFoundError(
                "Qwen3-TTS MLX snapshot is incomplete",
                details={"model_id": self._manifest.model_id, "missing": missing},
            )
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3_tts":
            raise ResourceIntegrityError(
                "Qwen3-TTS config has an unsupported model_type",
                details={"model_type": config.get("model_type")},
            )


class Qwen3TtsResourceProvider:
    def __init__(
        self,
        source: HuggingFaceSource,
        tokenizer_source: HuggingFaceSource,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        tokenizer_artifact: ModelArtifact,
        coreml_source: HuggingFaceSource | None = None,
        coreml_artifact: ModelArtifact | None = None,
    ) -> None:
        self._source = source
        self._tokenizer_source = tokenizer_source
        self._manifest = manifest
        self._artifact = artifact
        self._tokenizer_artifact = tokenizer_artifact
        self._coreml_source = coreml_source
        self._coreml_artifact = coreml_artifact

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        if self._coreml_source is not None and "runtime" not in dict(options or {}):
            mlx_status = self._resolver(variant, options).status()
            coreml_status = self._resolver(
                variant, dict(options or {}) | {"runtime": "coreml"}
            ).status()
            artifacts = tuple(
                artifact
                for index, artifact in enumerate((*mlx_status.artifacts, *coreml_status.artifacts))
                if artifact.artifact_id != "tokenizers"
                or not any(
                    previous.artifact_id == "tokenizers"
                    for previous in (*mlx_status.artifacts, *coreml_status.artifacts)[:index]
                )
            )
            return ModelResourceStatus(
                model_id=mlx_status.model_id,
                revision=mlx_status.revision,
                variant=mlx_status.variant,
                artifacts=artifacts,
            )
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
        del variant, options, overwrite
        raise UnsupportedRuntimeError(f"Qwen3-TTS conversion to {target_format} is not implemented")

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        normalized = dict(options or {})
        if runtime is not None:
            normalized["runtime"] = runtime
        resolver = self._resolver(variant, normalized)
        if isinstance(resolver, Qwen3TtsCoreMlResourceResolver):
            await resolver.delete()
        else:
            await resolver.delete(runtime=runtime)
        return resolver.status()

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> Qwen3TtsResourceResolver | Qwen3TtsCoreMlResourceResolver:
        if variant not in {item.name for item in self._manifest.variants}:
            raise ResourceNotFoundError(f"Qwen3-TTS variant is not registered: {variant}")
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError("Conflicting Qwen3-TTS variant values were provided")
        config = Qwen3TtsInstanceConfig.model_validate(normalized | {"variant": variant})
        selected_runtime = str(normalized.get("runtime", "mlx"))
        if selected_runtime == "coreml":
            if self._coreml_source is None or self._coreml_artifact is None:
                raise ResourceNotFoundError("Qwen3-TTS Core ML source is not registered")
            return Qwen3TtsCoreMlResourceResolver(
                self._coreml_source,
                self._tokenizer_source,
                config,
                self._manifest,
                self._coreml_artifact,
                self._tokenizer_artifact,
            )
        return Qwen3TtsResourceResolver(
            self._source,
            config,
            self._manifest,
            self._artifact,
            self._tokenizer_artifact,
            tokenizer_source=self._tokenizer_source,
        )
