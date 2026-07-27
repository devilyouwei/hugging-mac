"""Converter discovery and model-specific override selection."""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import RegistrationConflictError, UnsupportedRuntimeError
from hugging_mac_sdk.schemas.conversion import ConversionRequest, ConversionResult

if TYPE_CHECKING:
    from hugging_mac_sdk.core.registry import ModelDefinition


class ConverterRegistry:
    def __init__(self) -> None:
        self._converters: dict[str, ModelConverter] = {}
        self._lock = RLock()

    def register(self, converter: ModelConverter, *, replace: bool = False) -> None:
        with self._lock:
            if converter.converter_id in self._converters and not replace:
                raise RegistrationConflictError(
                    f"Converter already registered: {converter.converter_id}"
                )
            self._converters[converter.converter_id] = converter

    def get(self, converter_id: str) -> ModelConverter:
        with self._lock:
            try:
                return self._converters[converter_id]
            except KeyError as error:
                raise UnsupportedRuntimeError(
                    f"Converter is not registered: {converter_id}"
                ) from error

    def resolve(
        self,
        request: ConversionRequest,
        *,
        preferred_ids: tuple[str, ...] = (),
    ) -> ModelConverter:
        with self._lock:
            converters = tuple(self._converters.values())

        for converter_id in preferred_ids:
            converter = self.get(converter_id)
            if converter.supports(request):
                return converter

        candidates = [converter for converter in converters if converter.supports(request)]
        if not candidates:
            raise UnsupportedRuntimeError(
                f"No converter supports {request.source_format} -> {request.target_format}",
                details={
                    "source_format": request.source_format,
                    "target_format": request.target_format,
                    "model_id": request.model_id,
                },
            )
        return max(candidates, key=lambda converter: (converter.priority, converter.converter_id))


class ConversionService:
    def __init__(self, registry: ConverterRegistry) -> None:
        self._registry = registry

    async def convert(
        self,
        request: ConversionRequest,
        *,
        definition: ModelDefinition | None = None,
        converter_id: str | None = None,
    ) -> ConversionResult:
        if converter_id is not None:
            converter = self._registry.get(converter_id)
            if not converter.supports(request):
                raise UnsupportedRuntimeError(
                    f"Converter {converter_id} does not support this request"
                )
        else:
            preferred = definition.converter_ids if definition is not None else ()
            converter = self._registry.resolve(request, preferred_ids=preferred)
        return await converter.convert(request)
