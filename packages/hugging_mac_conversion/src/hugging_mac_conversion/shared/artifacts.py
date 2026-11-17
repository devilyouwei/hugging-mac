"""Project-local temporary bundles and their calculated file inventory."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for root in (current, *current.parents):
        if (root / "pyproject.toml").is_file() and (root / "packages/hugging_mac_sdk").is_dir():
            return root
    raise ValueError("Run inside the repository or supply an explicit storage root")


def default_output_root(start: Path | None = None) -> Path:
    return project_root(start) / "tmp/conversions"


def default_source_root(start: Path | None = None) -> Path:
    return project_root(start) / "tmp/originals"


def default_evaluation_root(start: Path | None = None) -> Path:
    return project_root(start) / "tmp/evaluations"


def source_directory(root: Path, model: str, variant: str, revision: str) -> Path:
    import re

    if any(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", n) is None for n in (model, variant)):
        raise ValueError("Invalid model/variant path")
    if re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
        raise ValueError("Source storage requires an immutable commit hash")
    return root / model / variant / revision


def create_run(output: Path, model: str, variant: str, runtime: str | None) -> Path:
    # Even direct Python callers cannot turn identities into arbitrary paths.
    import re

    parts = (model, variant, runtime) if runtime is not None else (model, variant)
    if any(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", n) is None for n in parts):
        raise ValueError("Invalid artifact path component")
    run = output.joinpath(*parts) / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    run.mkdir(parents=True, exist_ok=False)
    return run


def write_report(run: Path, report: dict[str, Any]) -> None:
    staged = run / ".build.json.tmp"
    staged.write_text(json.dumps(report, indent=2) + "\n")
    staged.replace(run / "build.json")


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Release bundles must not contain symlinks")
        if path.is_file() and path != root / "build.json":
            with path.open("rb") as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    return result
