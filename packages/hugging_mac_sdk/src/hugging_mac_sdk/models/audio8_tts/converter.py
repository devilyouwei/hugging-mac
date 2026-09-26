"""Architecture-aware export of every Audio8 neural stage to Core AI."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
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

from .utils.coreai_export import codec_decoder_module, codec_encoder_module, native_fast_step_module
from .utils.coreai_slow import slow_step_module
from .utils.types import CoreAILayout, CoreAINativeMetadata


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
            if not artifact_available(target, staged):
                raise ResourceNotFoundError("Audio8 Core AI conversion produced incomplete files")
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
        model.requires_grad_(False)
        slow = slow_step_module(model, config.max_seq_len)
        fast = native_fast_step_module(model)
        slow_names = ("keys", "values", "conv", "ssm")
        fast_names = ("keys", "values")
        metadata = CoreAINativeMetadata(
            sample_rate=config.codec_sample_rate,
            frame_length=config.codec_frame_size,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            semantic_begin_id=config.semantic_begin_id,
            max_seq_len=config.max_seq_len,
            ras_window_size=config.ras_window_size,
            ras_top_p=config.ras_top_p,
            ras_temperature=config.ras_temperature,
            slow_states={name: tuple(getattr(slow, name).shape) for name in slow_names},
            fast_states={name: tuple(getattr(fast, name).shape) for name in fast_names},
        )
        (output / self._layout.config_file).write_text(
            json.dumps(metadata.model_dump(), indent=2) + "\n"
        )
        convert_pytorch_to_coreai(
            slow,
            output / self._layout.slow_graph,
            example_inputs=(
                torch.zeros(1, config.num_codebooks + 1, 1, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            ),
            input_names=["ids", "position"],
            output_names=["logits", "hidden"],
            state_names=slow_names,
        )
        convert_pytorch_to_coreai(
            fast,
            output / self._layout.fast_graph,
            example_inputs=(
                torch.zeros(1, 1, config.fast_dim),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            ),
            input_names=["hidden", "token", "position"],
            output_names=["logits"],
            state_names=fast_names,
        )
        codec = model.load_codec(device="cpu", dtype=torch.float32).requires_grad_(False)
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
        convert_pytorch_to_coreai(
            codec_encoder_module(codec),
            output / self._layout.encoder_graph,
            example_inputs=(torch.zeros(1, 1, config.codec_frame_size * 8),),
            dynamic_shapes={
                "audio": {
                    2: config.codec_frame_size
                    * torch.export.Dim("frames", min=1, max=self._layout.codec_max_frames)
                }
            },
            input_names=["audio"],
            output_names=["codes"],
        )
