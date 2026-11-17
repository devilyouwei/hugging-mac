"""Private declarative layout for the Vela Core AI bundle."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CoreAILayout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_artifact: str
    target_artifact: str
    export_sequence_length: int = Field(ge=2)
    max_sequence_length: int = Field(ge=2)
    decision_asset: Path

    @field_validator("decision_asset")
    @classmethod
    def validate_relative_path(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts or not value.parts:
            raise ValueError("Core AI assets must use safe relative paths")
        return value


@dataclass(frozen=True, slots=True)
class ExportGraph:
    module: Any
    args: tuple[Any, ...]
    input_names: tuple[str, ...]
    output_name: str
    dynamic_shapes: Any
