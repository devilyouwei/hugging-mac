"""Standalone transactional Core AI export for the conversion package."""

from __future__ import annotations

import importlib
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4


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
        raise RuntimeError(
            "Core AI conversion requires coreai-torch (PyTorch >=2.8); "
            "install it in a separate conversion environment",
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
