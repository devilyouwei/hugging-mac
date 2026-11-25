"""Pinned inputs for the Zipformer publication recipe."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    upstream_checkpoint_revision: str
    upstream_checkpoint_sha256: str
    license: str
    revision: str
    archive: str
    archive_sha256: str
    graph_suffix: str
    encoder_chunk_size: int
    features_per_chunk: int
    features_with_lookahead: int
    source_files: dict[str, str]


class Definition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    default_variant: str
    variants: dict[str, Variant]


DEFINITION = Definition.model_validate(
    yaml.safe_load(Path(__file__).with_name("model.yaml").read_text())
)
VARIANT = DEFINITION.variants[DEFINITION.default_variant]
PARTS = ("encoder", "decoder", "joiner")
