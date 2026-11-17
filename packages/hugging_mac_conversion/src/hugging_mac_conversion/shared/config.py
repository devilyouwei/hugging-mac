"""Validated declarative model/variant configuration, without framework imports."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProbeConfig(FrozenConfig):
    description: str
    sequence_length: int = Field(gt=0)
    dynamic_sequence_length: int | None = Field(default=None, gt=0)
    inputs: tuple[str, ...] = Field(min_length=1)
    output: str
    download_patterns: tuple[str, ...]
    ignore_patterns: tuple[str, ...] = ()
    asset_patterns: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()


class VariantConfig(FrozenConfig):
    source: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    target: str = Field(pattern=r"^hugging-mac/[\w.-]+$")
    probe: ProbeConfig


class ModelConfig(FrozenConfig):
    name: Name
    default_variant: Name
    sdk_validator: str = Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
    required_capabilities: tuple[str, ...] = Field(min_length=1)
    variants: dict[Name, VariantConfig]

    @model_validator(mode="after")
    def check_default(self) -> ModelConfig:
        if self.default_variant not in self.variants:
            raise ValueError("default_variant must name a declared variant")
        return self

    def variant(self, name: str | None = None) -> VariantConfig:
        selected = name or self.default_variant
        if selected not in self.variants:
            raise ValueError(f"Unknown variant {selected!r} for {self.name}")
        return self.variants[selected]


def load_config(path: Path) -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(path.read_text()))
