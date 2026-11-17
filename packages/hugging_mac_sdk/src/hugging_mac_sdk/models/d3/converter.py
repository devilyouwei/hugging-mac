"""Export original D3 weights to a scoped Core AI artifact with shared dependencies."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any

from hugging_mac_sdk.converters import ModelConverter, convert_pytorch_to_coreai
from hugging_mac_sdk.converters.coreai import _publish
from hugging_mac_sdk.core.config import ModelPackageConfig
from hugging_mac_sdk.core.resources import artifact_available
from hugging_mac_sdk.errors import ResourceNotFoundError, UnsupportedRuntimeError
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult

from .utils.execution import run_blocking
from .utils.types import CoreAILayout


class D3CoreAIConverter(ModelConverter):
    def __init__(self, package: ModelPackageConfig, layout: CoreAILayout) -> None:
        self._package, self._layout = package, layout

    @property
    def converter_id(self) -> str:
        return "d3.coreai"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        return (
            request.model_id == self._package.manifest.model_id
            and request.variant in {v.name for v in self._package.manifest.variants}
            and request.source_format is ArtifactFormat.SAFETENSORS
            and request.target_format is ArtifactFormat.COREAI
            and request.source.path.is_dir()
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if not self.supports(request):
            raise UnsupportedRuntimeError("Unsupported D3 Core AI conversion")
        if request.options:
            raise ValueError(f"Unknown D3 conversion options: {sorted(request.options)}")
        cancelled = Event()
        # Mark cancellation before draining the worker, so staged files cannot publish.
        task = asyncio.create_task(run_blocking(self._convert, request, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    def _convert(self, request: ConversionRequest, cancelled: Event) -> ConversionResult:
        source = request.source.path.resolve()
        output = request.output_path.expanduser().absolute()
        resolved = output.resolve()
        if source == resolved or source.is_relative_to(resolved) or resolved.is_relative_to(source):
            raise ValueError("D3 conversion source and output must not overlap")
        if output.exists() and not request.overwrite:
            raise FileExistsError(output)
        target = self._package.get_artifact(
            self._layout.target_artifact, variant=request.variant, runtime="coreai"
        )
        dependencies = self._package.get_artifacts(
            variant=request.variant, runtime="pytorch", include_shared=True
        )
        if any(not artifact_available(artifact, source) for artifact in dependencies):
            raise ResourceNotFoundError("D3 conversion source is incomplete")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".d3-coreai-", dir=output.parent) as temporary:
            staged = Path(temporary) / "artifact"
            staged.mkdir()
            self._build(source, staged)
            # Only variant-owned configuration is copied; shares stay in canonical storage.
            for name in target.required_files:
                if name.parts[0] not in {
                    self._layout.decision_asset.parts[0],
                    self._layout.vision_asset.parts[0],
                }:
                    (staged / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, staged / name)
            for license_name in ("NOTICE", "LICENSE"):
                if (source / license_name).is_file():
                    shutil.copyfile(source / license_name, staged / license_name)
            if not artifact_available(target, staged):
                raise ResourceNotFoundError("D3 conversion produced incomplete native assets")
            digest, size = directory_sha256(staged), directory_size(staged)
            if cancelled.is_set():
                raise RuntimeError("D3 conversion cancelled before publication")
            _publish(staged, output, overwrite=request.overwrite)
        return ConversionResult(
            path=output,
            format=ArtifactFormat.COREAI,
            digest=digest,
            size_bytes=size,
            converter_id=self.converter_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            variant=request.variant,
            source_digest=request.source.digest,
        )

    def _build(self, source: Path, output: Path) -> None:
        import gc

        import torch
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        from .utils.coreai_export import prepare_decision
        from .utils.coreai_vision import prepare_vision

        decision = prepare_decision(
            source, self._layout.export_sequence_length, self._layout.max_sequence_length
        )
        convert_pytorch_to_coreai(
            decision.module,
            output / self._layout.decision_asset,
            example_inputs=decision.args,
            dynamic_shapes=decision.dynamic_shapes,
            input_names=decision.input_names,
            output_names=[decision.output_name],
            optimize=False,
        )
        del decision
        gc.collect()
        config = AutoConfig.from_pretrained(source, local_files_only=True).vision_config
        config._attn_implementation = "sdpa"
        visual: Any = Qwen3_5VisionModel(config)
        visual = visual.float().eval().requires_grad_(False)
        weights = {}
        for file in source.glob("*.safetensors"):
            with safe_open(file, framework="pt") as handle:
                for key in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
                    if key.startswith("visual."):
                        weights[key.removeprefix("visual.")] = handle.get_tensor(key)
        visual.load_state_dict(weights, strict=True)
        del weights
        vision = prepare_vision(visual)
        with torch.no_grad():
            convert_pytorch_to_coreai(
                vision.module,
                output / self._layout.vision_asset,
                example_inputs=vision.args,
                dynamic_shapes=vision.dynamic_shapes,
                input_names=vision.input_names,
                output_names=[vision.output_name],
                optimize=False,
            )
