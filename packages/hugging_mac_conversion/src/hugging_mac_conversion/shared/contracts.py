"""Small ports between model adapters and shared conversion orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import ModelConfig, VariantConfig
from .evaluation.contracts import EvaluationDefinition

Runtime = Literal["coreml", "coreai"]


@dataclass(frozen=True)
class PreparedGraph:
    module: Any
    args: tuple[Any, ...]
    input_names: tuple[str, ...]
    output_name: str
    dynamic_shapes: Any = None


@dataclass(frozen=True)
class ModelDefinition:
    config: ModelConfig
    prepare: Callable[[VariantConfig, Path], PreparedGraph]
    evaluation: EvaluationDefinition | None = None

    def snapshot(self, variant: str) -> dict[str, Any]:
        """Only the selected variant belongs to a build's reproducible configuration."""
        return {
            "model": self.config.name,
            "variant": variant,
            "settings": self.config.variant(variant).model_dump(mode="json"),
            "required_capabilities": list(self.config.required_capabilities),
            "bundle_validator": self.config.bundle_validator,
        }
