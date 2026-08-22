"""ONNX Runtime provider with Core ML EP preference on macOS."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.base import RuntimeBackend
from hugging_mac_sdk.schemas.model_structure import ModelComponentStructure, TensorStructure


class OnnxRuntimeSession:
    def __init__(self, session: Any, providers: tuple[str, ...]) -> None:
        self._session = session
        self.providers = providers

    @property
    def device(self) -> str:
        return self.providers[0] if self.providers else "unknown"

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._session is None:
            raise RuntimeError("ONNX Runtime session is closed")
        output_names = [item.name for item in self._session.get_outputs()]
        values = self._session.run(output_names, dict(inputs))
        return dict(zip(output_names, values, strict=True))

    async def close(self) -> None:
        self._session = None


class OnnxRuntimeProvider(RuntimeBackend):
    @property
    def name(self) -> str:
        return "onnx"

    def is_available(self) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None

    def available_devices(self) -> tuple[str, ...]:
        if not self.is_available():
            return ()
        ort = importlib.import_module("onnxruntime")
        return tuple(str(item) for item in ort.get_available_providers())

    def select_execution_providers(
        self, requested: str | Sequence[str] | None = None
    ) -> tuple[str, ...]:
        available = self.available_devices()
        if requested is None or requested == "auto":
            preference: tuple[str, ...] = (
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            )
        elif isinstance(requested, str):
            aliases = {
                "coreml": "CoreMLExecutionProvider",
                "cpu": "CPUExecutionProvider",
            }
            preference = (aliases.get(requested.lower(), requested), "CPUExecutionProvider")
        else:
            preference = tuple(requested)
        selected = tuple(dict.fromkeys(item for item in preference if item in available))
        if not selected:
            raise UnsupportedRuntimeError(
                "No requested ONNX Runtime execution provider is available",
                details={"requested": tuple(preference), "available": available},
            )
        return selected

    async def create_session(
        self,
        artifact: Path,
        *,
        device: str | None,
        options: Mapping[str, Any],
    ) -> OnnxRuntimeSession:
        if not self.is_available():
            raise UnsupportedRuntimeError("ONNX Runtime is not installed")
        ort = importlib.import_module("onnxruntime")
        requested = options.get("execution_providers", device or "auto")
        providers = self.select_execution_providers(requested)
        session_options = options.get("session_options")
        session = await asyncio.to_thread(
            ort.InferenceSession,
            str(artifact),
            sess_options=session_options,
            providers=list(providers),
        )
        return OnnxRuntimeSession(session, providers)


def inspect_onnx_artifact(path: Path) -> tuple[ModelComponentStructure, ...]:
    if not importlib.util.find_spec("onnx"):
        raise UnsupportedRuntimeError("ONNX structure inspection requires the onnx package")
    onnx = importlib.import_module("onnx")
    candidates = (path,) if path.suffix == ".onnx" else tuple(sorted(path.rglob("*.onnx")))
    if not candidates:
        raise UnsupportedRuntimeError("No .onnx model was found in this artifact")
    components: list[ModelComponentStructure] = []
    for candidate in candidates:
        model = onnx.load(str(candidate), load_external_data=False)
        graph = model.graph
        initializers = {item.name: item for item in graph.initializer}
        operators: dict[str, int] = {}
        for node in graph.node:
            operators[node.op_type] = operators.get(node.op_type, 0) + 1
        parameter_count = sum(
            _shape_size(tuple(int(value) for value in item.dims)) for item in initializers.values()
        )
        components.append(
            ModelComponentStructure(
                name=candidate.name,
                model_type="onnx",
                inputs=tuple(
                    _onnx_tensor(item, onnx)
                    for item in graph.input
                    if item.name not in initializers
                ),
                outputs=tuple(_onnx_tensor(item, onnx) for item in graph.output),
                node_count=len(graph.node),
                parameter_count=parameter_count,
                operator_counts=dict(
                    sorted(operators.items(), key=lambda item: (-item[1], item[0]))
                ),
                metadata={"ir_version": int(model.ir_version)},
            )
        )
    return tuple(components)


def _onnx_tensor(value: Any, onnx: Any) -> TensorStructure:
    tensor = value.type.tensor_type
    shape = tuple(
        int(dim.dim_value) if dim.HasField("dim_value") else (dim.dim_param or "?")
        for dim in tensor.shape.dim
    )
    return TensorStructure(
        name=value.name,
        dtype=str(onnx.TensorProto.DataType.Name(tensor.elem_type)).lower(),
        shape=shape,
    )


def _shape_size(shape: tuple[int, ...]) -> int:
    total = 1
    for value in shape:
        total *= value
    return total
