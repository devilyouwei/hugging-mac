"""Core ML provider and session.

This module owns Core ML model loading and Apple compute-unit selection. It
intentionally knows nothing about image preprocessing or model output schemas.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import re
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Executor
from functools import partial
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.base import RuntimeBackend
from hugging_mac_sdk.schemas.model_structure import ModelComponentStructure, TensorStructure

_COMPUTE_UNIT_ALIASES = {
    "auto": "ALL",
    "all": "ALL",
    "cpu": "CPU_ONLY",
    "cpu-only": "CPU_ONLY",
    "cpu_and_gpu": "CPU_AND_GPU",
    "cpu-and-gpu": "CPU_AND_GPU",
    "gpu": "CPU_AND_GPU",
    "cpu_and_ne": "CPU_AND_NE",
    "cpu-and-neural-engine": "CPU_AND_NE",
    "ane": "CPU_AND_NE",
    "npu": "CPU_AND_NE",
}


class CoreMLSession:
    def __init__(self, model: Any, device: str, *, retained_predictions: int = 8) -> None:
        self._model = model
        self._device = device
        get_spec = getattr(model, "get_spec", None)
        self._input_names = (
            frozenset(item.name for item in get_spec().description.input)
            if get_spec is not None
            else frozenset()
        )
        # coremltools may expose output arrays backed by native MLMultiArray
        # storage. Retain a small history of both feature providers until Core
        # ML's internal asynchronous work has fully quiesced.
        self._retained_predictions: deque[tuple[dict[str, Any], Mapping[str, Any]]] = deque(
            maxlen=retained_predictions
        )

    @property
    def device(self) -> str:
        return self._device

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._model is None:
            raise RuntimeError("Core ML session is closed")
        selected = (
            {name: value for name, value in inputs.items() if name in self._input_names}
            if self._input_names
            else dict(inputs)
        )
        raw = self._model.predict(selected)
        self._retained_predictions.append((selected, raw))
        return _copy_outputs(raw)

    def make_state(self) -> Any:
        if self._model is None:
            raise RuntimeError("Core ML session is closed")
        maker = getattr(self._model, "make_state", None)
        if maker is None:
            raise UnsupportedRuntimeError("This Core ML model does not expose MLState")
        return maker()

    def run_with_state(self, inputs: Mapping[str, Any], state: Any) -> Mapping[str, Any]:
        if self._model is None:
            raise RuntimeError("Core ML session is closed")
        selected = (
            {name: value for name, value in inputs.items() if name in self._input_names}
            if self._input_names
            else dict(inputs)
        )
        raw = self._model.predict(selected, state=state)
        self._retained_predictions.append((selected, raw))
        return _copy_outputs(raw)

    async def close(self) -> None:
        self._model = None
        self._retained_predictions.clear()


class CoreMLProvider(RuntimeBackend):
    @property
    def name(self) -> str:
        return "coreml"

    def is_available(self) -> bool:
        return importlib.util.find_spec("coremltools") is not None

    def available_devices(self) -> tuple[str, ...]:
        return ("all", "cpu-and-neural-engine", "cpu-and-gpu", "cpu-only")

    async def create_session(
        self,
        artifact: Path,
        *,
        device: str | None,
        options: Mapping[str, Any],
        executor: Executor | None = None,
    ) -> CoreMLSession:
        if not self.is_available():
            raise UnsupportedRuntimeError(
                "Core ML requires coremltools on Apple Silicon; install the yolo extra"
            )
        requested = (device or str(options.get("compute_units", "all"))).lower()
        try:
            unit_name = _COMPUTE_UNIT_ALIASES[requested]
        except KeyError as error:
            raise UnsupportedRuntimeError(
                f"Unsupported Core ML compute units: {requested}",
                details={"available": self.available_devices()},
            ) from error
        coremltools = importlib.import_module("coremltools")
        compute_units = getattr(coremltools.ComputeUnit, unit_name)
        function_name = options.get("function_name")
        model_type = (
            coremltools.models.CompiledMLModel
            if artifact.suffix == ".mlmodelc"
            else coremltools.models.MLModel
        )
        model = await asyncio.get_running_loop().run_in_executor(
            executor,
            partial(
                model_type,
                str(artifact),
                compute_units=compute_units,
                function_name=str(function_name) if function_name is not None else None,
            ),
        )
        retained_predictions = int(options.get("retained_predictions", 8))
        if retained_predictions < 0 or retained_predictions > 8:
            raise ValueError("retained_predictions must be between 0 and 8")
        return CoreMLSession(model, requested, retained_predictions=retained_predictions)


def _copy_outputs(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Detach ndarray-like results from Core ML-owned native storage."""
    copied: dict[str, Any] = {}
    for name, value in outputs.items():
        copy = getattr(value, "copy", None)
        copied[name] = copy() if callable(copy) else value
    return copied


def inspect_coreml_artifact(path: Path) -> tuple[ModelComponentStructure, ...]:
    """Return normalized structure for Core ML models contained in an artifact."""

    candidates = _coreml_candidates(path)
    if not candidates:
        raise UnsupportedRuntimeError(
            "No inspectable .mlpackage or .mlmodel was found in this artifact",
            details={"path": str(path)},
        )
    components: list[ModelComponentStructure] = []
    coremltools: Any | None = None
    for candidate in candidates:
        if candidate.suffix == ".mlmodelc":
            components.append(_inspect_compiled_coreml(candidate))
            continue
        if not importlib.util.find_spec("coremltools"):
            raise UnsupportedRuntimeError("Core ML inspection requires coremltools")
        coremltools = coremltools or importlib.import_module("coremltools")
        model = coremltools.models.MLModel(str(candidate), skip_model_load=True)
        spec = model.get_spec()
        description = spec.description
        operators = _coreml_operator_counts(spec)
        components.append(
            ModelComponentStructure(
                name=candidate.name,
                model_type=spec.WhichOneof("Type") or "coreml",
                inputs=tuple(_coreml_feature(item) for item in description.input),
                outputs=tuple(_coreml_feature(item) for item in description.output),
                node_count=sum(operators.values()) or None,
                operator_counts=operators,
                metadata={
                    "specification_version": int(spec.specificationVersion),
                    "functions": len(getattr(description, "functions", ())),
                },
            )
        )
    return tuple(components)


def _coreml_candidates(path: Path) -> tuple[Path, ...]:
    if path.suffix in {".mlpackage", ".mlmodel", ".mlmodelc"}:
        return (path,)
    if not path.is_dir():
        return ()
    packages = sorted((*path.rglob("*.mlpackage"), *path.rglob("*.mlmodelc")))
    package_roots = tuple(item.resolve() for item in packages)
    models = [
        item
        for item in sorted(path.rglob("*.mlmodel"))
        if not any(item.resolve().is_relative_to(root) for root in package_roots)
    ]
    return tuple([*packages, *models])


def _inspect_compiled_coreml(path: Path) -> ModelComponentStructure:
    mil_path = path / "model.mil"
    if not mil_path.is_file():
        return ModelComponentStructure(name=path.name, model_type="compiled-coreml")
    source = mil_path.read_text(encoding="utf-8", errors="replace")
    tensors: dict[str, TensorStructure] = {}
    tensor_pattern = re.compile(
        r"tensor<(?P<dtype>[^,>]+),\s*\[(?P<shape>[^\]]*)\]>\s+(?P<name>[A-Za-z_]\w*)"
    )
    for match in tensor_pattern.finditer(source):
        tensors[match.group("name")] = TensorStructure(
            name=match.group("name"),
            dtype=match.group("dtype"),
            shape=tuple(_mil_dimension(item) for item in match.group("shape").split(",")),
        )
    function = re.search(r"func\s+\w+<[^>]*>\((?P<inputs>.*?)\)\s*\{", source, re.DOTALL)
    input_names = (
        tuple(match.group("name") for match in tensor_pattern.finditer(function.group("inputs")))
        if function
        else ()
    )
    output_match = re.search(r"\}\s*->\s*\((?P<outputs>[^)]*)\)", source)
    output_names = (
        tuple(item.strip() for item in output_match.group("outputs").split(",") if item.strip())
        if output_match
        else ()
    )
    operators: dict[str, int] = {}
    operation_pattern = re.compile(
        r"^\s*(?:tensor<[^>]+>|[A-Za-z_]\w*)\s+[A-Za-z_]\w*\s*=\s*([A-Za-z_]\w*)\(",
        re.MULTILINE,
    )
    for operator in operation_pattern.findall(source):
        operators[operator] = operators.get(operator, 0) + 1
    parameter_count = sum(
        _shape_size(tuple(value for value in tensor.shape if isinstance(value, int)))
        for name, tensor in tensors.items()
        if re.search(rf"\b{re.escape(name)}\b\s*=\s*const\([^\n]*BLOBFILE", source)
    )
    return ModelComponentStructure(
        name=path.name,
        model_type="compiled-coreml",
        inputs=tuple(tensors[name] for name in input_names if name in tensors),
        outputs=tuple(tensors.get(name, TensorStructure(name=name)) for name in output_names),
        node_count=sum(operators.values()) or None,
        parameter_count=parameter_count or None,
        operator_counts=dict(sorted(operators.items(), key=lambda item: (-item[1], item[0]))),
    )


def _mil_dimension(value: str) -> int | str:
    value = value.strip()
    if not value:
        return "?"
    try:
        return int(value)
    except ValueError:
        return value


def _shape_size(shape: tuple[int, ...]) -> int:
    total = 1
    for value in shape:
        total *= value
    return total


def _coreml_feature(feature: Any) -> TensorStructure:
    feature_type = feature.type
    kind = feature_type.WhichOneof("Type") or "unknown"
    if kind == "multiArrayType":
        tensor = feature_type.multiArrayType
        shape = tuple(int(value) for value in tensor.shape)
        return TensorStructure(
            name=feature.name,
            dtype=_protobuf_enum_name(tensor, "dataType", int(tensor.dataType)).lower(),
            shape=shape,
        )
    if kind == "imageType":
        image = feature_type.imageType
        shape = tuple(value for value in (int(image.height), int(image.width)) if value)
        return TensorStructure(name=feature.name, dtype="image", shape=shape)
    return TensorStructure(name=feature.name, dtype=kind)


def _protobuf_enum_name(message: Any, field_name: str, value: int) -> str:
    field = message.DESCRIPTOR.fields_by_name[field_name]
    descriptor = field.enum_type.values_by_number.get(value)
    return descriptor.name if descriptor is not None else str(value)


def _coreml_operator_counts(spec: Any) -> dict[str, int]:
    counts: dict[str, int] = {}

    def add(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    model_type = spec.WhichOneof("Type")
    if model_type in {"neuralNetwork", "neuralNetworkClassifier", "neuralNetworkRegressor"}:
        network = getattr(spec, model_type)
        for layer in network.layers:
            add(layer.WhichOneof("layer") or "unknown")
    elif model_type == "mlProgram":
        for function in spec.mlProgram.functions.values():
            for block in function.block_specializations.values():
                for operation in block.operations:
                    add(str(operation.type))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
