"""Typed evaluation configuration and model-adapter ports."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ..config import FrozenConfig, Name, VariantConfig


class EvaluationCase(FrozenConfig):
    id: Name
    capability: str
    payload: dict[str, Any]
    expected: str | None = None
    variants: tuple[str, ...] = ()


class EvaluationConfig(FrozenConfig):
    scope: Literal["diagnostic"] = "diagnostic"
    cases: tuple[str, ...] = Field(min_length=1)
    atol: float = Field(default=0.02, ge=0, allow_inf_nan=False)
    rtol: float = Field(default=0.02, ge=0, allow_inf_nan=False)
    require_decision_agreement: bool = True

    @model_validator(mode="after")
    def safe_case_paths(self) -> EvaluationConfig:
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in self.cases):
            raise ValueError("Case files must be relative to the model evaluation directory")
        return self


@dataclass(frozen=True)
class EvaluationSample:
    case: EvaluationCase
    inputs: dict[str, Any]
    reference: Any
    labels: tuple[str, ...]
    output_name: str
    temperature: float = 1.0
    reference_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class EvaluationDefinition:
    config_path: Path
    prepare: Callable[[VariantConfig, Path, tuple[EvaluationCase, ...]], list[EvaluationSample]]

    def load(self, variant: str) -> tuple[EvaluationConfig, tuple[EvaluationCase, ...]]:
        config = EvaluationConfig.model_validate(yaml.safe_load(self.config_path.read_text()))
        cases = []
        root = self.config_path.parent.resolve()
        for name in config.cases:
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Evaluation case file escapes its model directory")
            for item in json.loads(path.read_text()):
                case = EvaluationCase.model_validate(item)
                if not case.variants or variant in case.variants:
                    cases.append(case)
        ids = [case.id for case in cases]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Evaluation requires non-empty cases with unique IDs")
        return config, tuple(cases)
