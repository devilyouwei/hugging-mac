"""Runtime-neutral model structure returned by artifact inspectors."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TensorStructure(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str | None = None
    shape: tuple[int | str, ...] = ()


class ModelLayerStructure(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    layer_type: str
    parameter_count: int = 0


class ModelComponentStructure(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    model_type: str
    inputs: tuple[TensorStructure, ...] = ()
    outputs: tuple[TensorStructure, ...] = ()
    layers: tuple[ModelLayerStructure, ...] = ()
    node_count: int | None = None
    parameter_count: int | None = None
    operator_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ModelStructure(BaseModel):
    """Normalized structure for one installed model artifact."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    variant: str
    runtime: str
    artifact_id: str
    format: str
    size_bytes: int | None = None
    components: tuple[ModelComponentStructure, ...]

    @property
    def total_nodes(self) -> int | None:
        counts = [item.node_count for item in self.components if item.node_count is not None]
        return sum(counts) if counts else None

    @property
    def total_parameters(self) -> int | None:
        counts = [
            item.parameter_count for item in self.components if item.parameter_count is not None
        ]
        return sum(counts) if counts else None
