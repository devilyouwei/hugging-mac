from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hugging_mac_sdk.errors import ResourceNotFoundError


class ModelVariantSpec(BaseModel):
    """One interchangeable weight/configuration variant of a model architecture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
    display_name: Annotated[str, Field(min_length=1)]
    description: str = ""
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(min_length=1)]
    devices: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    quantizations: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()


class ModelManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "1"
    model_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._/-]*$")]
    revision: Annotated[str, Field(min_length=1)] = "main"
    display_name: Annotated[str, Field(min_length=1)]
    description: str = ""
    tags: frozenset[str] = frozenset()
    family: Annotated[str, Field(min_length=1)]
    capabilities: frozenset[str]
    runtimes: tuple[RuntimeSpec, ...]
    license: str | None = None
    source_url: str | None = None
    default_runtime: str | None = None
    variants: tuple[ModelVariantSpec, ...] = (
        ModelVariantSpec(name="default", display_name="Default"),
    )
    default_variant: str = "default"

    @model_validator(mode="after")
    def validate_manifest(self) -> ModelManifest:
        if not self.capabilities:
            raise ValueError("At least one capability is required")
        if not self.runtimes:
            raise ValueError("At least one runtime is required")
        runtime_names = [runtime.name for runtime in self.runtimes]
        if len(runtime_names) != len(set(runtime_names)):
            raise ValueError("Runtime names must be unique")
        if self.default_runtime is not None and self.default_runtime not in runtime_names:
            raise ValueError("default_runtime must be declared in runtimes")
        if not self.variants:
            raise ValueError("At least one variant is required")
        variant_names = [variant.name for variant in self.variants]
        if len(variant_names) != len(set(variant_names)):
            raise ValueError("Variant names must be unique")
        if self.default_variant not in variant_names:
            raise ValueError("default_variant must be declared in variants")
        return self

    def get_variant(self, variant: str | None = None) -> ModelVariantSpec:
        selected = variant or self.default_variant
        for spec in self.variants:
            if spec.name == selected:
                return spec
        raise ResourceNotFoundError(
            f"Variant {selected} is not declared for {self.model_id}",
            details={
                "model_id": self.model_id,
                "variant": selected,
                "supported_variants": [item.name for item in self.variants],
            },
        )
