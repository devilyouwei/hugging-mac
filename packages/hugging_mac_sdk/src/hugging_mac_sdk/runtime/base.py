"""Runtime adapter boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from hugging_mac_sdk.core.instance import BaseModelInstance
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import ResolvedResource


class RuntimeAdapter(ABC):
    """Create model instances without exposing backend objects to callers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def create_instance(
        self,
        manifest: ModelManifest,
        resources: tuple[ResolvedResource, ...],
        options: Mapping[str, Any],
    ) -> BaseModelInstance: ...
