"""Compose the d3 configuration and its architecture adapter."""

from pathlib import Path

from ...shared.config import load_config
from ...shared.contracts import ModelDefinition
from .converter import prepare
from .evaluation.runner import EVALUATION

DEFINITION = ModelDefinition(
    load_config(Path(__file__).with_name("model.yaml")), prepare, EVALUATION
)
