"""PyTorch provider with automatic Apple MPS selection."""

from __future__ import annotations

import asyncio
import gc
import importlib
import importlib.util
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.base import RuntimeBackend
from hugging_mac_sdk.schemas.model_structure import (
    ModelComponentStructure,
    ModelLayerStructure,
    TensorStructure,
)

ModelLoader = Callable[[Path, Any], Any]
OutputAdapter = Callable[[Any], Mapping[str, Any]]


class TorchSession:
    def __init__(self, torch: Any, model: Any, device: str, output_adapter: OutputAdapter) -> None:
        self._torch = torch
        self._model = model
        self._device = device
        self._output_adapter = output_adapter

    @property
    def device(self) -> str:
        return self._device

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._model is None:
            raise RuntimeError("PyTorch session is closed")
        tensor = inputs.get("input")
        if tensor is None:
            raise ValueError("PyTorch session expects an 'input' tensor")
        if hasattr(tensor, "to"):
            tensor = tensor.to(self._device)
        with self._torch.inference_mode():
            output = self._model(tensor)
        return self._output_adapter(output)

    async def close(self) -> None:
        had_model = self._model is not None
        self._model = None
        if not had_model:
            return
        gc.collect()
        if self._device == "mps" and hasattr(self._torch, "mps"):
            empty_cache = getattr(self._torch.mps, "empty_cache", None)
            if empty_cache is not None:
                empty_cache()


class TorchProvider(RuntimeBackend):
    @property
    def name(self) -> str:
        return "pytorch"

    def is_available(self) -> bool:
        return importlib.util.find_spec("torch") is not None

    def available_devices(self) -> tuple[str, ...]:
        if not self.is_available():
            return ()
        torch = importlib.import_module("torch")
        devices: list[str] = []
        if bool(torch.backends.mps.is_available()):
            devices.append("mps")
        devices.append("cpu")
        return tuple(devices)

    def resolve_device(self, requested: str | None, *, allow_cpu_fallback: bool) -> str:
        requested = requested or "mps"
        available = self.available_devices()
        if requested in available:
            return requested
        if allow_cpu_fallback and "cpu" in available:
            return "cpu"
        raise UnsupportedRuntimeError(
            f"PyTorch device is not available: {requested}",
            details={"available": available},
        )

    async def create_session(
        self,
        artifact: Path,
        *,
        device: str | None,
        options: Mapping[str, Any],
    ) -> TorchSession:
        if not self.is_available():
            raise UnsupportedRuntimeError("PyTorch is not installed; install the yolo extra")
        torch = importlib.import_module("torch")
        loader = options.get("model_loader")
        if not callable(loader):
            raise TypeError("TorchProvider requires a callable 'model_loader' option")
        output_adapter = options.get("output_adapter", _default_output_adapter)
        if not callable(output_adapter):
            raise TypeError("'output_adapter' must be callable")
        resolved = self.resolve_device(
            device,
            allow_cpu_fallback=bool(options.get("allow_cpu_fallback", True)),
        )
        model = await asyncio.to_thread(loader, artifact, torch)
        model = model.eval().to(resolved)
        return TorchSession(torch, model, resolved, output_adapter)


def _default_output_adapter(output: Any) -> Mapping[str, Any]:
    return {"output": output}


def inspect_torch_artifact(path: Path) -> tuple[ModelComponentStructure, ...]:
    if not importlib.util.find_spec("torch"):
        raise UnsupportedRuntimeError("PyTorch structure inspection requires torch")
    torch = importlib.import_module("torch")
    candidates = (
        (path,)
        if path.suffix in {".pt", ".pth", ".torchscript"}
        else tuple(
            candidate
            for candidate in sorted(
                (*path.rglob("*.pt"), *path.rglob("*.pth"), *path.rglob("*.torchscript"))
            )
            if "voices" not in candidate.relative_to(path).parts
        )
    )
    if not candidates:
        raise UnsupportedRuntimeError("No TorchScript artifact was found")
    components: list[ModelComponentStructure] = []
    for candidate in candidates:
        try:
            model = torch.jit.load(str(candidate), map_location="cpu")
        except Exception as script_error:
            try:
                checkpoint = _safe_torch_load(torch, candidate)
                components.append(_inspect_state_dict(checkpoint, candidate.name))
                continue
            except Exception as checkpoint_error:
                raise UnsupportedRuntimeError(
                    "This PyTorch checkpoint needs a model-specific structure inspector",
                    details={"path": str(candidate)},
                    cause=checkpoint_error,
                ) from script_error
        graph = model.inlined_graph
        operators: dict[str, int] = {}
        for node in graph.nodes():
            kind = str(node.kind())
            operators[kind] = operators.get(kind, 0) + 1
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
        components.append(
            ModelComponentStructure(
                name=candidate.name,
                model_type="torchscript",
                inputs=tuple(_torch_value(value) for value in graph.inputs()),
                outputs=tuple(_torch_value(value) for value in graph.outputs()),
                node_count=sum(operators.values()),
                parameter_count=parameter_count,
                operator_counts=dict(
                    sorted(operators.items(), key=lambda item: (-item[1], item[0]))
                ),
            )
        )
    return tuple(components)


def inspect_torch_module(model: Any, name: str) -> ModelComponentStructure:
    """Summarize an already loaded module without running a forward pass."""

    operators: dict[str, int] = {}
    layers: list[ModelLayerStructure] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        kind = type(module).__name__
        operators[kind] = operators.get(kind, 0) + 1
        direct_parameters = tuple(module.parameters(recurse=False))
        layers.append(
            ModelLayerStructure(
                name=module_name,
                layer_type=kind,
                parameter_count=sum(int(parameter.numel()) for parameter in direct_parameters),
            )
        )
    parameters = tuple(model.parameters())
    return ModelComponentStructure(
        name=name,
        model_type="pytorch-module",
        layers=tuple(layers),
        parameter_count=sum(int(parameter.numel()) for parameter in parameters),
        operator_counts=dict(sorted(operators.items(), key=lambda item: (-item[1], item[0]))),
        metadata={"module_count": len(layers), "tensor_count": len(parameters)},
    )


def inspect_torch_checkpoint(
    path: Path, loader: ModelLoader
) -> tuple[ModelComponentStructure, ...]:
    """Inspect a checkpoint through a model package's constrained loader."""

    if not importlib.util.find_spec("torch"):
        raise UnsupportedRuntimeError("PyTorch structure inspection requires torch")
    torch = importlib.import_module("torch")
    model = loader(path, torch)
    return (inspect_torch_module(model, path.name),)


def _safe_torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=True)


def _inspect_state_dict(checkpoint: Any, name: str) -> ModelComponentStructure:
    state = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, Mapping):
                state = nested
                break
    tensors = list(_named_tensors(state))
    if not tensors:
        raise TypeError("Checkpoint state dictionary contains no tensors")
    return ModelComponentStructure(
        name=name,
        model_type="pytorch-state-dict",
        outputs=tuple(
            TensorStructure(
                name=tensor_name,
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=tuple(int(value) for value in tensor.shape),
            )
            for tensor_name, tensor in tensors
        ),
        parameter_count=sum(int(tensor.numel()) for _, tensor in tensors),
        metadata={"tensor_count": len(tensors)},
    )


def _named_tensors(value: Any, prefix: str = "") -> tuple[tuple[str, Any], ...]:
    if hasattr(value, "numel") and hasattr(value, "shape"):
        return ((prefix or "tensor", value),)
    if not isinstance(value, Mapping):
        return ()
    tensors: list[tuple[str, Any]] = []
    for key, nested in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        tensors.extend(_named_tensors(nested, name))
    return tuple(tensors)


def _torch_value(value: Any) -> TensorStructure:
    value_type = value.type()
    sizes = getattr(value_type, "sizes", lambda: None)()
    shape = tuple("?" if item is None else int(item) for item in sizes or ())
    dtype = getattr(value_type, "dtype", lambda: None)()
    return TensorStructure(name=str(value.debugName()), dtype=str(dtype or value_type), shape=shape)
