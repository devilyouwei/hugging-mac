"""Audio8-ASR Core ML audio-tower converter."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import ResourceIntegrityError, UnsupportedRuntimeError
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)

_COPIED_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
)


class Audio8AsrConverter(ModelConverter):
    """Convert the audio tower to Core ML and package the cached Qwen2 decoder."""

    def __init__(self, model_id: str, variant: str) -> None:
        self._model_id = model_id
        self._variant = variant

    @property
    def converter_id(self) -> str:
        return "audio8.audio8-asr-0.1b"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        return (
            request.model_id == self._model_id
            and request.variant == self._variant
            and request.source_format is ArtifactFormat.SAFETENSORS
            and request.target_format is ArtifactFormat.COREML
            and request.source.path.is_dir()
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if not self.supports(request):
            raise UnsupportedRuntimeError(
                "Audio8-ASR converter only supports base safetensors -> Core ML"
            )
        return await asyncio.to_thread(self._convert_sync, request)

    def _convert_sync(self, request: ConversionRequest) -> ConversionResult:
        source = request.source.path.expanduser().resolve()
        output = request.output_path.expanduser().resolve()
        staging = output.parent / f".{output.name}.partial-{uuid4().hex}"
        if output.exists() and not request.overwrite:
            raise FileExistsError(f"Core ML artifact already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        try:
            self._build_artifact(
                source,
                staging,
                quantize_weights=bool(request.options.get("quantize_weights", True)),
            )
            digest = directory_sha256(staging)
            size_bytes = directory_size(staging)
            if output.exists():
                if output.is_dir():
                    shutil.rmtree(output)
                else:
                    output.unlink()
            os.replace(staging, output)
            return ConversionResult(
                path=output,
                format=ArtifactFormat.COREML,
                digest=digest,
                size_bytes=size_bytes,
                converter_id=self.converter_id,
                model_id=request.model_id,
                model_revision=request.model_revision,
                variant=request.variant,
                source_digest=request.source.digest,
                options={
                    "quantize_weights": bool(request.options.get("quantize_weights", True)),
                    "audio_buckets": [500, 1000, 3000],
                    "decoder_runtime": "pytorch-mps",
                    "minimum_deployment_target": "macOS15",
                },
            )
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(staging)
            raise

    def _build_artifact(
        self,
        source: Path,
        staging: Path,
        *,
        quantize_weights: bool,
    ) -> None:
        torch = importlib.import_module("torch")
        coremltools = importlib.import_module("coremltools")
        np = importlib.import_module("numpy")
        safetensors = importlib.import_module("safetensors.torch")
        from .utils.coreml import AUDIO_BUCKET_FRAMES, AUDIO_BUCKET_TOKENS, CoreMlAudioTower
        from .utils.modeling import load_audio8_asr_model

        model = load_audio8_asr_model(
            source,
            device="cpu",
            dtype=torch.float32,
        )
        function_root = staging / "_functions"
        function_root.mkdir()
        function_packages: list[tuple[Path, str]] = []
        try:
            for bucket_frames in AUDIO_BUCKET_FRAMES:
                token_count = AUDIO_BUCKET_TOKENS[bucket_frames]
                wrapper = CoreMlAudioTower(model, bucket_frames).eval()
                audio_example = torch.zeros(
                    (1, 128, bucket_frames),
                    dtype=torch.float32,
                )
                mask_example = torch.zeros(
                    (1, 1, token_count, token_count),
                    dtype=torch.float32,
                )
                traced = torch.jit.trace(
                    wrapper,
                    (audio_example, mask_example),
                    strict=False,
                )
                converted = coremltools.convert(
                    traced,
                    convert_to="mlprogram",
                    inputs=[
                        coremltools.TensorType(
                            name="audios",
                            shape=audio_example.shape,
                            dtype=np.float32,
                        ),
                        coremltools.TensorType(
                            name="attn_mask",
                            shape=mask_example.shape,
                            dtype=np.float32,
                        ),
                    ],
                    outputs=[
                        coremltools.TensorType(
                            name="hidden",
                            dtype=np.float16,
                        )
                    ],
                    compute_precision=coremltools.precision.FLOAT16,
                    minimum_deployment_target=coremltools.target.macOS15,
                )
                if quantize_weights:
                    optimization = coremltools.optimize.coreml.OptimizationConfig(
                        global_config=coremltools.optimize.coreml.OpLinearQuantizerConfig(
                            mode="linear_symmetric",
                            dtype="int8",
                        )
                    )
                    converted = coremltools.optimize.coreml.linear_quantize_weights(
                        converted,
                        optimization,
                    )
                package = function_root / f"tower_{bucket_frames}.mlpackage"
                converted.save(str(package))
                function_packages.append((package, f"tower_{bucket_frames // 100}s"))

            descriptor = coremltools.utils.MultiFunctionDescriptor()
            for package, function_name in function_packages:
                descriptor.add_function(
                    str(package),
                    "main",
                    function_name,
                )
            descriptor.default_function_name = "tower_30s"
            coremltools.utils.save_multifunction(
                descriptor,
                str(staging / "audio_tower.mlpackage"),
            )
            self._extract_runtime_weights(
                source,
                staging,
                safetensors=safetensors,
            )
            for filename in _COPIED_CONFIG_FILES:
                source_file = source / filename
                if not source_file.is_file():
                    raise ResourceIntegrityError(
                        f"Audio8-ASR conversion source is missing {filename}"
                    )
                shutil.copy2(source_file, staging / filename)
            (staging / "conversion.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "format": "audio8-asr-coreml-hybrid",
                        "audio_tower": "Core ML",
                        "decoder": "PyTorch MPS",
                        "functions": [name for _, name in function_packages],
                        "quantize_weights": quantize_weights,
                        "minimum_deployment_target": "macOS15",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        finally:
            del model
            gc.collect()
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(function_root)

    @staticmethod
    def _extract_runtime_weights(
        source: Path,
        destination: Path,
        *,
        safetensors: Any,
    ) -> None:
        language_weights: dict[str, Any] = {}
        projector_weights: dict[str, Any] = {}
        with safetensors.safe_open(
            source / "model.safetensors",
            framework="pt",
            device="cpu",
        ) as checkpoint:
            for name in checkpoint.keys():  # noqa: SIM118 - safe_open is not iterable
                if name.startswith("language_model."):
                    language_weights[name.removeprefix("language_model.")] = checkpoint.get_tensor(
                        name
                    )
                elif name.startswith("audio_projector."):
                    projector_weights[name.removeprefix("audio_projector.")] = (
                        checkpoint.get_tensor(name)
                    )
        if not language_weights or not projector_weights:
            raise ResourceIntegrityError(
                "Audio8-ASR checkpoint is missing decoder or projector weights"
            )
        safetensors.save_file(
            language_weights,
            destination / "language_model.safetensors",
        )
        safetensors.save_file(
            projector_weights,
            destination / "projector.safetensors",
        )
