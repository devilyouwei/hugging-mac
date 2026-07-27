"""Converter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hugging_mac_sdk.schemas.conversion import ConversionRequest, ConversionResult


class ModelConverter(ABC):
    """Convert one artifact format to another without owning model registration."""

    @property
    @abstractmethod
    def converter_id(self) -> str: ...

    @property
    def priority(self) -> int:
        """Higher values win during automatic generic selection."""

        return 0

    @abstractmethod
    def supports(self, request: ConversionRequest) -> bool: ...

    @abstractmethod
    async def convert(self, request: ConversionRequest) -> ConversionResult: ...
