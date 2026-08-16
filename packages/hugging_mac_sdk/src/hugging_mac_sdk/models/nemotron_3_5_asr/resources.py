"""Resource lifecycle for prebuilt Nemotron 3.5 ASR Core ML bundles."""

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
    NEMOTRON_3_5_ASR_MODEL_ID,
    NEMOTRON_3_5_ASR_REVISION,
    NemotronCoreMlInstanceConfig,
    variant_source_path,
)

_GRAPHS = ("preprocessor.mlmodelc", "encoder.mlmodelc", "decoder_joint.mlmodelc")
_REQUIRED = (
    *tuple(
        path
        for name in _GRAPHS
        for path in (f"{name}/model.mil", f"{name}/coremldata.bin", f"{name}/weights/weight.bin")
    ),
    "metadata.json",
    "tokenizer.json",
)


class NemotronCoreMlResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: NemotronCoreMlInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source, self._config = source, config
        self._downloader = downloader or ResourceDownloader()

    def root(self) -> Path:
        if self._config.artifact_path is not None:
            return self._config.artifact_path
        return (
            self._config.model_home
            / "nvidia"
            / "nemotron-3.5-asr-streaming-0.6b"
            / NEMOTRON_3_5_ASR_REVISION
            / self._config.variant
            / "coreml"
            / "model"
        )

    def bundle_root(self) -> Path:
        root = self.root()
        nested = root / variant_source_path(self._config.variant)
        return nested if nested.exists() else root

    async def resolve(self) -> ResolvedResource:
        root = self.bundle_root()
        missing = [name for name in _REQUIRED if not (root / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "Nemotron Core ML artifact is not downloaded", details={"missing": missing}
            )
        return ResolvedResource(
            path=root,
            source=self._source,
            digest=directory_sha256(root),
            size_bytes=directory_size(root),
        )

    def status(self) -> ModelResourceStatus:
        root = self.bundle_root()
        available = all((root / name).is_file() for name in _REQUIRED)
        return ModelResourceStatus(
            model_id=NEMOTRON_3_5_ASR_MODEL_ID,
            revision=NEMOTRON_3_5_ASR_REVISION,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="coreml-mixed",
                    format="coreml",
                    runtime="coreml",
                    available=available,
                    size_bytes=directory_size(root) if available else None,
                ),
            ),
        )

    async def download(self, *, overwrite: bool = False) -> None:
        if overwrite or not self.root().exists():
            await self._downloader.download(self._source, self.root(), overwrite=overwrite)
        await self.resolve()

    async def delete(self) -> None:
        root = self.root().expanduser().resolve(strict=False)
        model_root = (
            (self._config.model_home / "nvidia" / "nemotron-3.5-asr-streaming-0.6b")
            .expanduser()
            .resolve(strict=False)
        )
        if root != model_root and not root.is_relative_to(model_root):
            raise ResourceIntegrityError("Refusing to delete Nemotron resources outside model root")
        await asyncio.to_thread(shutil.rmtree, root, True)


class NemotronCoreMlResourceProvider:
    def __init__(self, sources: Mapping[str, HuggingFaceSource]) -> None:
        self._sources = dict(sources)

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> NemotronCoreMlResourceResolver:
        return NemotronCoreMlResourceResolver(
            self._sources[variant],
            NemotronCoreMlInstanceConfig.model_validate(dict(options or {}) | {"variant": variant}),
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
            f"Nemotron ships prebuilt Core ML graphs and cannot convert to {target_format}"
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
