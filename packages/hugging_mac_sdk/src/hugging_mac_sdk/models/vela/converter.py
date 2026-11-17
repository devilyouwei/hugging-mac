"""Export original Vela weights to a scoped Core AI artifact with shared dependencies."""

from __future__ import annotations

import asyncio
import json
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

from .utils.execution import run_blocking
from .utils.types import CoreAILayout


class VelaCoreAIConverter(ModelConverter):
    def __init__(self, package: ModelPackageConfig, layout: CoreAILayout) -> None:
        self._package, self._layout = package, layout

    @property
    def converter_id(self) -> str:
        return "vela.coreai"

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
            raise UnsupportedRuntimeError("Unsupported Vela Core AI conversion")
        if request.options:
            raise ValueError(f"Unknown Vela conversion options: {sorted(request.options)}")
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
            raise ValueError("Vela conversion source and output must not overlap")
        if output.exists() and not request.overwrite:
            raise FileExistsError(output)
        target = self._package.get_artifact(
            self._layout.target_artifact, variant=request.variant, runtime="coreai"
        )
        dependencies = self._package.get_artifacts(
            variant=request.variant, runtime="pytorch", include_shared=True
        )
        if any(not artifact_available(artifact, source) for artifact in dependencies):
            raise ResourceNotFoundError("Vela conversion source is incomplete")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".vela-coreai-", dir=output.parent) as temporary:
            staged = Path(temporary) / "artifact"
            staged.mkdir()
            self._build(source, staged)
            for license_name in ("NOTICE", "LICENSE"):
                if (source / license_name).is_file():
                    shutil.copyfile(source / license_name, staged / license_name)
            if not (staged / "NOTICE").exists():
                shutil.copyfile(Path(__file__).with_name("NOTICE"), staged / "NOTICE")
            if not artifact_available(target, staged):
                raise ResourceNotFoundError("Vela conversion produced incomplete native assets")
            digest, size = directory_sha256(staged), directory_size(staged)
            if cancelled.is_set():
                raise RuntimeError("Vela conversion cancelled before publication")
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
        from .utils.coreai_export import prepare_decision

        graph = prepare_decision(source, self._layout)
        convert_pytorch_to_coreai(
            graph.module,
            output / self._layout.decision_asset,
            example_inputs=graph.args,
            dynamic_shapes=graph.dynamic_shapes,
            input_names=graph.input_names,
            output_names=[graph.output_name],
            optimize=False,
        )
        config = json.loads((source / "config.json").read_text())
        decision = {key: config[key] for key in ("bos_token_id", "eos_token_id", "pad_token_id")}
        decision["marker_ids"] = config["decision"]["marker_ids"]
        (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_scope": "decision-only",
                    "function": "main",
                    "output": graph.output_name,
                    "sequence_length": self._layout.max_sequence_length,
                    "dynamic_sequence_length": True,
                    "option_count": "dynamic",
                },
                indent=2,
            )
            + "\n"
        )
