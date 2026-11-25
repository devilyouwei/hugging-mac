"""Core ML graph export and execution; no model-specific facts."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .contracts import PreparedGraph
from .coreml_ops import register_lowerings


def export(graph: PreparedGraph, output: Path) -> Path:
    torch = importlib.import_module("torch")
    # MIL's Torch frontend cannot materialize BF16 constants through NumPy.
    # Promote the export graph; the pipeline retains the original-dtype baseline.
    tensors = (*graph.module.parameters(), *graph.module.buffers())
    if any(tensor.dtype == torch.bfloat16 for tensor in tensors):
        print("Core ML: promoting BFloat16 graph weights to Float32", flush=True)
        graph.module.float()
    return export_mlprogram(
        graph.module,
        graph.args,
        graph.input_names,
        (graph.output_name,),
        output / "model.mlpackage",
    )


def export_mlprogram(
    module: Any,
    args: tuple[Any, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    path: Path,
    *,
    deployment_target: str = "macOS15",
    check_trace: bool = True,
) -> Path:
    torch = importlib.import_module("torch")
    ct = importlib.import_module("coremltools")
    register_lowerings()
    with torch.no_grad():
        traced = torch.jit.trace(module, args, strict=True, check_trace=check_trace)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, deployment_target),
        inputs=[
            ct.TensorType(name=name, shape=tuple(arg.shape), dtype=arg.numpy().dtype)
            for name, arg in zip(input_names, args, strict=True)
        ],
        outputs=[ct.TensorType(name=name) for name in output_names],
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )
    converted.save(str(path))
    return path


def predict(artifact: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    ct = importlib.import_module("coremltools")
    return dict(
        ct.models.MLModel(str(artifact), compute_units=ct.ComputeUnit.CPU_ONLY).predict(inputs)
    )
