"""Explicit composition root; shared infrastructure never imports model packages."""

from .models.d3.definition import DEFINITION as D3
from .models.vela.definition import DEFINITION as VELA
from .shared.contracts import ModelDefinition

MODELS = {definition.config.name: definition for definition in (VELA, D3)}


def get_model(name: str) -> ModelDefinition:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model: {name}") from None
