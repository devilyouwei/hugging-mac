"""Architecture-aware Audio8 fast AR and codec decoder export to Core AI."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from hugging_mac_sdk.converters import ModelConverter, convert_pytorch_to_coreai
from hugging_mac_sdk.converters.coreai import _publish
from hugging_mac_sdk.core.config import ModelPackageConfig
from hugging_mac_sdk.core.resources import artifact_available
from hugging_mac_sdk.errors import ResourceNotFoundError, UnsupportedRuntimeError
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult

from .utils.coreai_export import codec_decoder_module, fast_step_module
from .utils.types import CoreAILayout


class Audio8TtsCoreAIConverter(ModelConverter):
    def __init__(self, package: ModelPackageConfig, layout: CoreAILayout) -> None:
        self._package = package
        self._layout = layout
        self._targets = {
            a.variant: a for a in package.artifacts if a.format is ArtifactFormat.COREAI
        }

    @property
    def converter_id(self) -> str:
        return "audio8.tts.coreai"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        return (
            request.model_id == self._package.manifest.model_id
            and request.variant in self._targets
            and request.source_format is ArtifactFormat.SAFETENSORS
            and request.target_format is ArtifactFormat.COREAI
            and request.source.path.is_dir()
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if not self.supports(request):
            raise UnsupportedRuntimeError("Unsupported Audio8 TTS Core AI conversion")
        if request.options:
            raise ValueError(
                f"Unknown Audio8 Core AI conversion options: {sorted(request.options)}"
            )
        cancelled = Event()
        task = asyncio.create_task(asyncio.to_thread(self._convert, request, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # Native conversion is not interruptible; wait for staging cleanup.
            with contextlib.suppress(Exception):
                await task
            raise

    def _convert(self, request: ConversionRequest, cancelled: Event) -> ConversionResult:
        source = request.source.path.resolve()
        output = request.output_path.expanduser().absolute()
        if source == output.resolve() or source.is_relative_to(output.resolve()):
            raise ValueError("Core AI output must not replace the source model or its parent")
        if output.exists() and not request.overwrite:
            raise FileExistsError(output)
        declaration = self._package.get_artifact(
            self._layout.source_artifact, variant=request.variant, runtime="pytorch"
        )
        if not artifact_available(declaration, source):
            raise ResourceNotFoundError("Audio8 conversion source is incomplete")
        target = self._targets[request.variant]
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".audio8-coreai-", dir=output.parent) as temporary:
            staged = Path(temporary) / "artifact"
            staged.mkdir()
            self._build(source, staged)
            retained = staged / self._layout.pytorch_directory
            retained.mkdir(exist_ok=True)
            for name in declaration.required_files:
                destination = retained / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, destination)
            if not artifact_available(target, staged):
                raise ResourceNotFoundError("Audio8 Core AI conversion produced incomplete files")
            retained = staged / self._layout.pytorch_directory
            for name in declaration.required_files:
                if not (retained / name).is_file():
                    raise ResourceNotFoundError(f"Missing retained PyTorch file: {name}")
            digest, size = directory_sha256(staged), directory_size(staged)
            if cancelled.is_set():
                raise RuntimeError("Audio8 Core AI conversion cancelled before publication")
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
            options={"stages": self._layout.stages},
        )

    def _build(self, source: Path, output: Path) -> None:
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            importlib.import_module("coreai_torch")
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "Audio8 Core AI conversion requires the isolated conversion environment; "
                "see the Audio8 SDK documentation",
                cause=error,
            ) from error
        from .torch import (
            _install_falcon_h1_cache_compatibility,
            _restore_falcon_h1_buffers,
            _restore_rope_buffers,
        )

        _install_falcon_h1_cache_compatibility(transformers)
        model = transformers.AutoModel.from_pretrained(
            source, trust_remote_code=True, local_files_only=True, dtype=torch.float32
        ).eval()
        _restore_rope_buffers(torch, model)
        _restore_falcon_h1_buffers(transformers, model)
        config = model.config
        if config.slow_backbone != "falcon_h1":
            raise UnsupportedRuntimeError("This converter requires the Falcon-H1 Audio8 variant")
        shape = (
            config.n_fast_layer,
            1,
            config.fast_n_local_heads,
            config.num_codebooks,
            config.fast_head_dim,
        )
        convert_pytorch_to_coreai(
            fast_step_module(model),
            output / self._layout.fast_graph,
            example_inputs=(
                torch.zeros(1, 1, config.fast_dim),
                torch.tensor([0], dtype=torch.int64),
                torch.zeros(shape),
                torch.zeros(shape),
            ),
            input_names=["hidden", "position", "keys", "values"],
            output_names=["logits", "new_keys", "new_values"],
        )
        codec = model.load_codec(device="cpu", dtype=torch.float32)
        convert_pytorch_to_coreai(
            codec_decoder_module(codec),
            output / self._layout.codec_graph,
            example_inputs=(torch.zeros(1, config.num_codebooks, 8, dtype=torch.int32),),
            dynamic_shapes={
                "codes": {2: torch.export.Dim("frames", min=1, max=self._layout.codec_max_frames)}
            },
            input_names=["codes"],
            output_names=["waveform"],
        )
