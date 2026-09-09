"""Document Parser application."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blueprint import DocumentParserBlueprint


def create_blueprint() -> "DocumentParserBlueprint":
    # Keep persistence imports independent from FastAPI dependency imports.
    from .blueprint import create_blueprint as factory

    return factory()


__all__ = ["create_blueprint"]
