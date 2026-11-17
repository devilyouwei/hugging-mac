"""Small, explicit MIL lowerings for operations missing from the pinned converter."""

from __future__ import annotations

import importlib
from typing import Any


def register_lowerings() -> None:
    frontend = importlib.import_module("coremltools.converters.mil.frontend.torch")
    registry = importlib.import_module(
        "coremltools.converters.mil.frontend.torch.torch_op_registry"
    )._TORCH_OPS_REGISTRY
    if registry.get_func("linalg_solve_triangular") is not None:
        return
    mb = importlib.import_module("coremltools.converters.mil").Builder
    get_inputs = importlib.import_module(
        "coremltools.converters.mil.frontend.torch.ops"
    )._get_inputs

    def linalg_solve_triangular(context: Any, node: Any) -> None:
        matrix, rhs, upper, left, unit = get_inputs(context, node, expected=5)
        if bool(upper.val) or not bool(left.val) or not bool(unit.val):
            raise NotImplementedError("MIL lowering supports lower, left, unit-triangular solves")
        width = matrix.shape[-1]
        if not isinstance(width, int) or matrix.shape[-2] != width:
            raise NotImplementedError("Triangular solve requires a static square matrix")

        def rows(value: Any, start: int, stop: int, columns: int | None = None) -> Any:
            rank = len(value.shape)
            begin, end = [0] * rank, [0] * rank
            begin[-2], end[-2] = start, stop
            end[-1] = columns or 0
            return mb.slice_by_index(
                x=value,
                begin=begin,
                end=end,
                begin_mask=[True] * (rank - 2) + [False, True],
                end_mask=[True] * (rank - 2) + [False, columns is None],
            )

        # Forward substitution preserves the diagonal contract without matrix inversion.
        solution = rows(rhs, 0, 1)
        for index in range(1, width):
            previous = mb.matmul(x=rows(matrix, index, index + 1, index), y=solution)
            current = mb.sub(x=rows(rhs, index, index + 1), y=previous)
            solution = mb.concat(values=[solution, current], axis=-2)
        context.add(mb.identity(x=solution, name=node.name))

    frontend.register_torch_op(linalg_solve_triangular)
