"""Explicit PyTorch export to Apple's Core AI asset format."""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult


class CoreAIExportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_names: tuple[str, ...] | None = None
    output_names: tuple[str, ...] | None = None
    state_names: tuple[str, ...] | None = None
    function_name: str = Field(default="main", min_length=1)
    optimize: bool = True


def convert_pytorch_to_coreai(
    model: Any,
    output_path: Path,
    *,
    example_inputs: tuple[Any, ...] | None = None,
    example_kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Any = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    state_names: Sequence[str] | None = None,
    function_name: str = "main",
    optimize: bool = True,
    overwrite: bool = False,
) -> Path:
    """Convert an eval-mode nn.Module or ExportedProgram, publishing only on success.

    Modules require example inputs. Checkpoint reconstruction belongs to the caller;
    this helper never unpickles a checkpoint or changes the caller's training mode.
    """
    output = Path(output_path).expanduser().absolute()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if not function_name:
        raise ValueError("function_name must not be empty")
    try:
        torch = importlib.import_module("torch")
        coreai_torch = importlib.import_module("coreai_torch")
    except ImportError as error:
        raise UnsupportedRuntimeError(
            "Core AI conversion requires coreai-torch (PyTorch >=2.8); "
            "install it in a separate conversion environment",
            cause=error,
        ) from error
    if isinstance(model, torch.export.ExportedProgram):
        if example_inputs is not None or example_kwargs is not None or dynamic_shapes is not None:
            raise ValueError("Example inputs and dynamic shapes apply only to nn.Module")
        exported = model
    elif isinstance(model, torch.nn.Module):
        if any(module.training for module in model.modules()):
            raise ValueError("Call model.eval() before Core AI conversion")
        if example_inputs is None:
            raise ValueError("nn.Module conversion requires example_inputs")
        exported = torch.export.export(
            model, args=example_inputs, kwargs=example_kwargs, dynamic_shapes=dynamic_shapes
        )
    else:
        raise TypeError("Expected a torch.nn.Module or torch.export.ExportedProgram")
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    program = (
        coreai_torch.TorchConverter()
        .add_exported_program(
            exported,
            input_names=input_names,
            output_names=output_names,
            state_names=state_names,
            entrypoint_name=function_name,
        )
        .to_coreai()
    )
    if optimize:
        program.optimize()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}.convert-", dir=output.parent) as temporary:
        staged = Path(temporary) / "model.aimodel"
        program.save_asset(staged)
        if not staged.is_dir() or not any(staged.iterdir()):
            raise RuntimeError("Core AI converter did not produce a non-empty asset")
        _publish(staged, output, overwrite=overwrite)
    return output


def _publish(staged: Path, output: Path, *, overwrite: bool) -> None:
    backup = output.with_name(f".{output.name}.backup-{uuid4().hex}")
    had_output = output.exists() or output.is_symlink()
    if had_output:
        if not overwrite:
            raise FileExistsError(output)
        os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException:
        if had_output:
            os.replace(backup, output)
        raise
    if had_output:
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()


class PyTorchCoreAIConverter(ModelConverter):
    """Convert saved torch.export .pt2 programs through ConversionService."""

    @property
    def converter_id(self) -> str:
        return "pytorch.coreai"

    def supports(self, request: ConversionRequest) -> bool:
        return (
            request.source_format is ArtifactFormat.PYTORCH
            and request.target_format is ArtifactFormat.COREAI
            and request.source.path.suffix == ".pt2"
            and request.source.path.is_file()
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if not self.supports(request):
            raise UnsupportedRuntimeError("Core AI conversion requires a PyTorch .pt2 export")
        options = CoreAIExportOptions.model_validate(request.options)
        return await asyncio.to_thread(self._convert, request, options)

    def _convert(
        self, request: ConversionRequest, options: CoreAIExportOptions
    ) -> ConversionResult:
        if request.output_path.exists() and not request.overwrite:
            raise FileExistsError(request.output_path)
        if request.source.path.resolve() == request.output_path.resolve():
            raise ValueError("Conversion output must differ from the source")
        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "Core AI conversion requires PyTorch", cause=error
            ) from error
        model = torch.export.load(request.source.path)
        output = convert_pytorch_to_coreai(
            model, request.output_path, overwrite=request.overwrite, **options.model_dump()
        )
        return ConversionResult(
            path=output,
            format=ArtifactFormat.COREAI,
            digest=directory_sha256(output),
            size_bytes=directory_size(output),
            converter_id=self.converter_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            variant=request.variant,
            source_digest=request.source.digest,
            options=options.model_dump(mode="json"),
        )
