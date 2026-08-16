"""SenseVoiceSmall resource resolution, download, status, and deletion."""

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
from hugging_mac_sdk.resources.hashing import directory_size, file_sha256
from hugging_mac_sdk.resources.views import merged_directory_view
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import (
    SENSEVOICE_SMALL_MODEL_ID,
    SENSEVOICE_SMALL_REQUIRED_FILES,
    SENSEVOICE_SMALL_REVISION,
    SENSEVOICE_SMALL_TOKENIZER_FILES,
    SENSEVOICE_SMALL_TOKENIZER_SHA256,
    SENSEVOICE_SMALL_VARIANT,
    SENSEVOICE_SMALL_WEIGHT_SHA256,
    SenseVoiceSmallInstanceConfig,
)
from .converter import SenseVoiceSmallConverter


class SenseVoiceSmallResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: SenseVoiceSmallInstanceConfig,
        *,
        tokenizer_source: HuggingFaceSource | None = None,
        downloader: ResourceDownloader | None = None,
        converter: SenseVoiceSmallConverter | None = None,
    ) -> None:
        self._source = source
        self._tokenizer_source = tokenizer_source or source
        self._config = config
        self._downloader = downloader or ResourceDownloader(timeout=1800)
        self._converter = converter or SenseVoiceSmallConverter()

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
        return (
            self._config.tokenizer_path
            or self._config.model_home
            / "funaudiollm"
            / "sensevoice"
            / "shared"
            / "tokenizer"
            / SENSEVOICE_SMALL_TOKENIZER_SHA256
        )

    async def resolve_tokenizer(self) -> ResolvedResource:
        missing = [
            name
            for name in SENSEVOICE_SMALL_TOKENIZER_FILES
            if not (self.tokenizer_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "SenseVoiceSmall shared tokenizer is incomplete",
                details={"model_id": SENSEVOICE_SMALL_MODEL_ID, "missing": missing},
            )
        digest = file_sha256(
            self.tokenizer_path / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        )
        if digest != SENSEVOICE_SMALL_TOKENIZER_SHA256:
            raise ResourceIntegrityError(
                "SenseVoiceSmall tokenizer SHA-256 mismatch",
                details={"expected": SENSEVOICE_SMALL_TOKENIZER_SHA256, "actual": digest},
            )
        return ResolvedResource(
            path=self.tokenizer_path,
            source=self._tokenizer_source,
            digest=digest,
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

    async def resolve_coreml(self) -> ResolvedResource:
        path = self._config.artifact_path or self._default_coreml_path()
        await asyncio.to_thread(self._validate_coreml_artifact, path)
        return ResolvedResource(
            path=path, source=self._source, digest=None, size_bytes=directory_size(path)
        )

    async def convert_coreml(self, *, overwrite: bool = False) -> None:
        path = self._config.artifact_path or self._default_coreml_path()
        if path.exists() and not overwrite:
            await self.resolve_coreml()
            return
        source = await self.resolve_source()
        tokenizer = await self.resolve_tokenizer()
        with merged_directory_view(source.path, (tokenizer.path,)) as conversion_source:
            await self._converter.convert(
                ConversionRequest(
                    model_id=SENSEVOICE_SMALL_MODEL_ID,
                    model_revision=SENSEVOICE_SMALL_REVISION,
                    variant=self._config.variant,
                    source=source.model_copy(update={"path": conversion_source}),
                    source_format=ArtifactFormat.PYTORCH,
                    target_format=ArtifactFormat.COREML,
                    output_path=path,
                    overwrite=overwrite,
                )
            )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "pytorch-mps", "coreml"}:
            raise UnsupportedRuntimeError(
                f"SenseVoiceSmall does not support runtime {runtime}",
                details={"model_id": SENSEVOICE_SMALL_MODEL_ID, "runtime": runtime},
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
                "Refusing to delete SenseVoiceSmall resources outside the model directory",
                details={"model_id": SENSEVOICE_SMALL_MODEL_ID},
            )
        await asyncio.to_thread(self._delete_path, resolved)

    def status(self) -> ModelResourceStatus:
        path = self._config.source_path or self._default_source_path()
        available = self._snapshot_files_exist(path)
        coreml_path = self._config.artifact_path or self._default_coreml_path()
        coreml_available = self._coreml_files_exist(coreml_path)
        tokenizer_available = all(
            (self.tokenizer_path / name).is_file()
            for name in SENSEVOICE_SMALL_TOKENIZER_FILES
        )
        return ModelResourceStatus(
            model_id=SENSEVOICE_SMALL_MODEL_ID,
            revision=SENSEVOICE_SMALL_REVISION,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="source",
                    format=ArtifactFormat.PYTORCH.value,
                    runtime="pytorch-mps",
                    available=available,
                    size_bytes=directory_size(path) if available else None,
                ),
                ModelArtifactStatus(
                    artifact_id="tokenizer",
                    format=ArtifactFormat.TOKENIZER.value,
                    runtime=None,
                    shared=True,
                    available=tokenizer_available,
                    size_bytes=(
                        directory_size(self.tokenizer_path) if tokenizer_available else None
                    ),
                ),
                ModelArtifactStatus(
                    artifact_id="coreml",
                    format=ArtifactFormat.COREML.value,
                    runtime="coreml",
                    provisioning="convert",
                    available=coreml_available,
                    size_bytes=directory_size(coreml_path) if coreml_available else None,
                ),
            ),
        )

    def _default_source_path(self) -> Path:
        return self._model_root() / "pytorch" / "model"

    def _default_coreml_path(self) -> Path:
        return self._model_root() / "coreml" / "sensevoice-small-coreml"

    def _model_root(self) -> Path:
        return (
            self._config.model_home
            / "funaudiollm"
            / "sensevoice"
            / self._source.revision
            / self._config.variant
        )

    @staticmethod
    def _snapshot_files_exist(path: Path) -> bool:
        return path.is_dir() and all(
            (path / name).is_file() for name in SENSEVOICE_SMALL_REQUIRED_FILES
        )

    @staticmethod
    def _coreml_files_exist(path: Path) -> bool:
        return path.is_dir() and all(
            (path / name).exists()
            for name in (
                "sensevoice.mlpackage",
                "am.mvn",
                "conversion.json",
            )
        )

    def _validate_coreml_artifact(self, path: Path) -> None:
        if not self._coreml_files_exist(path):
            raise ResourceNotFoundError(
                "SenseVoiceSmall Core ML artifact has not been converted",
                details={"model_id": SENSEVOICE_SMALL_MODEL_ID, "artifact_id": "coreml"},
            )
        metadata = json.loads((path / "conversion.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "sensevoice-small-coreml":
            raise ResourceIntegrityError(
                "SenseVoiceSmall Core ML artifact metadata is invalid",
                details={"format": metadata.get("format")},
            )

    def _validate_snapshot(self, path: Path) -> str:
        if not path.is_dir():
            raise ResourceNotFoundError(
                "SenseVoiceSmall source model is not downloaded",
                details={
                    "model_id": SENSEVOICE_SMALL_MODEL_ID,
                    "variant": self._config.variant,
                    "artifact_id": "source",
                },
            )
        missing = [name for name in SENSEVOICE_SMALL_REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "SenseVoiceSmall snapshot is incomplete",
                details={"model_id": SENSEVOICE_SMALL_MODEL_ID, "missing": missing},
            )
        digest = file_sha256(path / "model.pt")
        if digest != SENSEVOICE_SMALL_WEIGHT_SHA256:
            raise ResourceIntegrityError(
                "SenseVoiceSmall weight SHA-256 mismatch",
                details={"expected": SENSEVOICE_SMALL_WEIGHT_SHA256, "actual": digest},
            )
        metadata = json.loads((path / "configuration.json").read_text(encoding="utf-8"))
        if metadata.get("framework") != "pytorch":
            raise ResourceIntegrityError(
                "SenseVoiceSmall configuration has an unexpected framework",
                details={"framework": metadata.get("framework")},
            )
        return digest

    @staticmethod
    def _delete_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


class SenseVoiceSmallResourceProvider:
    def __init__(
        self,
        source: HuggingFaceSource,
        tokenizer_source: HuggingFaceSource | None = None,
    ) -> None:
        self._source = source
        self._tokenizer_source = tokenizer_source or source

    async def status(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
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
            raise UnsupportedRuntimeError(
                f"SenseVoiceSmall conversion to {target_format} is not implemented",
                details={
                    "model_id": SENSEVOICE_SMALL_MODEL_ID,
                    "variant": variant,
                    "target_format": target_format,
                },
            )
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

    def _resolver(
        self,
        variant: str,
        options: Mapping[str, object] | None,
    ) -> SenseVoiceSmallResourceResolver:
        if variant != SENSEVOICE_SMALL_VARIANT:
            raise ResourceNotFoundError(
                f"SenseVoiceSmall variant is not registered: {variant}",
                details={
                    "model_id": SENSEVOICE_SMALL_MODEL_ID,
                    "variant": variant,
                    "supported_variants": [SENSEVOICE_SMALL_VARIANT],
                },
            )
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError(
                "Conflicting SenseVoiceSmall variant values were provided",
                details={"variant": variant, "options_variant": option_variant},
            )
        config = SenseVoiceSmallInstanceConfig.model_validate(normalized | {"variant": variant})
        return SenseVoiceSmallResourceResolver(
            self._source, config, tokenizer_source=self._tokenizer_source
        )
