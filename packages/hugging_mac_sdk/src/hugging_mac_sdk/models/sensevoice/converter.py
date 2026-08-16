"""SenseVoiceSmall SANM/CTC Core ML converter."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import ResourceIntegrityError, UnsupportedRuntimeError
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult

from .config import SENSEVOICE_SMALL_MODEL_ID

FEATURE_BUCKETS = (100, 250, 500)
_RUNTIME_FILES = ("am.mvn",)


class SenseVoiceSmallConverter(ModelConverter):
    @property
    def converter_id(self) -> str:
        return "funaudiollm.sensevoice-small"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        return (
            request.model_id == SENSEVOICE_SMALL_MODEL_ID
            and request.variant == "small"
            and request.source_format is ArtifactFormat.PYTORCH
            and request.target_format is ArtifactFormat.COREML
            and request.source.path.is_dir()
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if not self.supports(request):
            raise UnsupportedRuntimeError(
                "SenseVoiceSmall converter only supports small PyTorch -> Core ML"
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
        quantize = bool(request.options.get("quantize_weights", True))
        try:
            self._build_artifact(source, staging, quantize_weights=quantize)
            digest = directory_sha256(staging)
            size_bytes = directory_size(staging)
            if output.exists():
                shutil.rmtree(output) if output.is_dir() else output.unlink()
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
                    "quantize_weights": quantize,
                    "feature_buckets": list(FEATURE_BUCKETS),
                    "minimum_deployment_target": "macOS15",
                },
            )
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(staging)
            raise

    def _build_artifact(self, source: Path, staging: Path, *, quantize_weights: bool) -> None:
        torch = importlib.import_module("torch")
        ct = importlib.import_module("coremltools")
        np = importlib.import_module("numpy")
        sentencepiece = importlib.import_module("sentencepiece")
        yaml = importlib.import_module("yaml")
        from .utils.modeling import SenseVoiceCoreMlWrapper, load_sensevoice_model

        tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(source / "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
        config = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
        model = load_sensevoice_model(
            source,
            vocabulary_size=int(tokenizer.get_piece_size()),
            config=config,
            device="cpu",
            dtype=torch.float32,
        )
        function_root = staging / "_functions"
        function_root.mkdir()
        packages: list[tuple[Path, str]] = []
        for frames in FEATURE_BUCKETS:
            wrapper = SenseVoiceCoreMlWrapper(model).eval()
            features = torch.zeros((1, frames, 560), dtype=torch.float32)
            lengths = torch.tensor([frames], dtype=torch.int32)
            language = torch.tensor([0], dtype=torch.int32)
            style = torch.tensor([14], dtype=torch.int32)
            traced = torch.jit.trace(wrapper, (features, lengths, language, style), strict=False)
            converted = ct.convert(
                traced,
                convert_to="mlprogram",
                inputs=[
                    ct.TensorType(name="features", shape=features.shape, dtype=np.float32),
                    ct.TensorType(name="lengths", shape=lengths.shape, dtype=np.int32),
                    ct.TensorType(name="language_id", shape=language.shape, dtype=np.int32),
                    ct.TensorType(name="style_id", shape=style.shape, dtype=np.int32),
                ],
                outputs=[ct.TensorType(name="logits", dtype=np.float16)],
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS15,
            )
            if quantize_weights:
                optimization = ct.optimize.coreml.OptimizationConfig(
                    global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                        mode="linear_symmetric", dtype="int8"
                    )
                )
                converted = ct.optimize.coreml.linear_quantize_weights(converted, optimization)
            package = function_root / f"encoder_{frames}.mlpackage"
            converted.save(str(package))
            packages.append((package, f"encoder_{frames}"))
        descriptor = ct.utils.MultiFunctionDescriptor()
        for package, name in packages:
            descriptor.add_function(str(package), "main", name)
        descriptor.default_function_name = "encoder_500"
        ct.utils.save_multifunction(descriptor, str(staging / "sensevoice.mlpackage"))
        shutil.rmtree(function_root)
        for filename in _RUNTIME_FILES:
            if not (source / filename).is_file():
                raise ResourceIntegrityError(f"SenseVoiceSmall source is missing {filename}")
            shutil.copy2(source / filename, staging / filename)
        (staging / "conversion.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": "sensevoice-small-coreml",
                    "functions": [name for _, name in packages],
                    "quantize_weights": quantize_weights,
                    "minimum_deployment_target": "macOS15",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
