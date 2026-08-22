"""Inspect installed model artifacts without creating inference instances."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import ResourceNotFoundError, UnsupportedCapabilityError
from hugging_mac_sdk.resources.hashing import directory_size
from hugging_mac_sdk.schemas.model_structure import ModelComponentStructure, ModelStructure


class ModelInspectionService:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    async def structure(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        variant: str | None = None,
        runtime: str,
        artifact_id: str,
        model_home: Path | None = None,
    ) -> ModelStructure:
        definition = self._registry.get(model_id, revision)
        selected_variant = definition.manifest.get_variant(variant).name
        artifact = definition.get_artifact(runtime, artifact_id, selected_variant)
        path = artifact.resolve(model_home or self._registry.storage_root)
        if not path.exists():
            raise ResourceNotFoundError(
                "Model artifact is not installed",
                details={"artifact_id": artifact_id, "path": str(path)},
            )

        inspector = definition.artifact_inspectors.get(runtime)
        if inspector is not None:
            components = await asyncio.to_thread(inspector, path)
        else:
            components = await asyncio.to_thread(
                _inspect_artifact,
                path,
                runtime,
                artifact.format.value,
            )
        size_bytes = directory_size(path) if path.is_dir() else path.stat().st_size
        return ModelStructure(
            model_id=model_id,
            revision=definition.manifest.revision,
            variant=selected_variant,
            runtime=runtime,
            artifact_id=artifact_id,
            format=artifact.format.value,
            size_bytes=size_bytes,
            components=components,
        )


def _inspect_artifact(
    path: Path, runtime: str, artifact_format: str
) -> tuple[ModelComponentStructure, ...]:
    if runtime == "coreml" or artifact_format == "coreml":
        from hugging_mac_sdk.runtime.coreml import inspect_coreml_artifact

        return inspect_coreml_artifact(path)
    if runtime == "onnx" or artifact_format == "onnx":
        from hugging_mac_sdk.runtime.onnx import inspect_onnx_artifact

        return inspect_onnx_artifact(path)
    if runtime == "mlx" or artifact_format in {"mlx", "safetensors"}:
        from hugging_mac_sdk.runtime.mlx import inspect_mlx_artifact

        return inspect_mlx_artifact(path)
    if runtime in {"pytorch", "pytorch-mps"} or artifact_format in {
        "pytorch",
        "torchscript",
    }:
        from hugging_mac_sdk.runtime.torch import inspect_torch_artifact

        return inspect_torch_artifact(path)
    raise UnsupportedCapabilityError(
        f"Model structure inspection is not supported for runtime {runtime}",
        details={"runtime": runtime, "format": artifact_format},
    )
