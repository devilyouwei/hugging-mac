"""Resource lifecycle for prebuilt Qwen3-ASR Core ML graphs."""

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
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import (
    QWEN3_ASR_COREML_MODEL_ID,
    QWEN3_ASR_COREML_REVISION,
    Qwen3AsrCoreMlInstanceConfig,
)

_GRAPHS = (
    "encoder.mlmodelc",
    "embedding.mlmodelc",
    "decoder_part1.mlmodelc",
    "decoder_part2.mlmodelc",
)
_REQUIRED = (*tuple(f"{graph}/model.mil" for graph in _GRAPHS), "config.json")
_TOKENIZER_REQUIRED = ("vocab.json", "merges.txt", "tokenizer_config.json")


class Qwen3AsrCoreMlResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        tokenizer_source: HuggingFaceSource,
        config: Qwen3AsrCoreMlInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source, self._tokenizer_source, self._config = source, tokenizer_source, config
        self._downloader = downloader or ResourceDownloader()

    def root(self) -> Path:
        if self._config.artifact_path is not None:
            return self._config.artifact_path
        return (
            self._config.model_home
            / "qwen"
            / "qwen3-asr"
            / QWEN3_ASR_COREML_REVISION
            / self._config.variant
            / "coreml"
            / "model"
        )

    async def resolve(self) -> ResolvedResource:
        root = self.root()
        missing = [name for name in _REQUIRED if not (root / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "Qwen3-ASR Core ML artifact is not downloaded", details={"missing": missing}
            )
        return ResolvedResource(
            path=root,
            source=self._source,
            digest=directory_sha256(root),
            size_bytes=directory_size(root),
        )

    def tokenizer_root(self) -> Path:
        if self._config.tokenizer_path is not None:
            return self._config.tokenizer_path
        return (
            self._config.model_home
            / "qwen"
            / "qwen3-asr"
            / "shared"
            / "tokenizer"
            / "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
        )

    async def resolve_tokenizer(self) -> ResolvedResource:
        root = self.tokenizer_root()
        missing = [name for name in _TOKENIZER_REQUIRED if not (root / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "Qwen3-ASR tokenizer is not downloaded", details={"missing": missing}
            )
        return ResolvedResource(
            path=root,
            source=self._tokenizer_source,
            digest=directory_sha256(root),
            size_bytes=directory_size(root),
        )

    async def download(self, *, overwrite: bool = False) -> None:
        if overwrite or not self.root().exists():
            await self._downloader.download(self._source, self.root(), overwrite=overwrite)
        await self.resolve()
        if overwrite or not self.tokenizer_root().exists():
            await self._downloader.download(
                self._tokenizer_source, self.tokenizer_root(), overwrite=overwrite
            )
        await self.resolve_tokenizer()

    def status(self) -> ModelResourceStatus:
        root = self.root()
        available = all((root / name).is_file() for name in _REQUIRED)
        tokenizer_root = self.tokenizer_root()
        tokenizer_available = all((tokenizer_root / name).is_file() for name in _TOKENIZER_REQUIRED)
        return ModelResourceStatus(
            model_id=QWEN3_ASR_COREML_MODEL_ID,
            revision=QWEN3_ASR_COREML_REVISION,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="coreml-int8",
                    format="coreml",
                    runtime="coreml",
                    available=available,
                    size_bytes=directory_size(root) if available else None,
                ),
                ModelArtifactStatus(
                    artifact_id="tokenizer",
                    format="tokenizer",
                    runtime=None,
                    shared=True,
                    available=tokenizer_available,
                    size_bytes=(directory_size(tokenizer_root) if tokenizer_available else None),
                ),
            ),
        )

    async def delete(self) -> None:
        root = self.root().expanduser().resolve(strict=False)
        model_root = (
            (self._config.model_home / "qwen" / "qwen3-asr").expanduser().resolve(strict=False)
        )
        if root != model_root and not root.is_relative_to(model_root):
            raise ResourceIntegrityError(
                "Refusing to delete Qwen3-ASR resources outside model root"
            )
        await asyncio.to_thread(shutil.rmtree, root, True)


class Qwen3AsrCoreMlResourceProvider:
    def __init__(self, source: HuggingFaceSource, tokenizer_source: HuggingFaceSource) -> None:
        self._source, self._tokenizer_source = source, tokenizer_source

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> Qwen3AsrCoreMlResourceResolver:
        return Qwen3AsrCoreMlResourceResolver(
            self._source,
            self._tokenizer_source,
            Qwen3AsrCoreMlInstanceConfig.model_validate(dict(options or {}) | {"variant": variant}),
        )

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        return self._resolver(variant, options).status()

    async def download_source(
        self, variant: str, options: Mapping[str, object] | None = None, *, overwrite: bool = False
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
            f"Qwen3-ASR ships prebuilt Core ML graphs and cannot convert to {target_format}"
        )

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        if runtime not in {None, "coreml"}:
            raise UnsupportedRuntimeError(f"Unsupported runtime: {runtime}")
        resolver = self._resolver(variant, options)
        await resolver.delete()
        return resolver.status()
