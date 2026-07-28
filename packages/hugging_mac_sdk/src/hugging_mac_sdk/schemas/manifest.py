from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hugging_mac_sdk.schemas.resources import ResourceSource


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
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1"
    model_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._/-]*$")]
    revision: Annotated[str, Field(min_length=1)] = "main"
    display_name: Annotated[str, Field(min_length=1)]
    description: str = ""
    tags: frozenset[str] = frozenset()
    family: Annotated[str, Field(min_length=1)]
    capabilities: frozenset[str]
    runtimes: tuple[RuntimeSpec, ...]
    resources: tuple[ResourceSource, ...] = ()
    license: str | None = None
    source_url: str | None = None
    default_runtime: str | None = None

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
        return self
