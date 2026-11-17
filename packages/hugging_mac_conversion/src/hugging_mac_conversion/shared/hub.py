"""Hub transport and asset packaging, driven exclusively by variant configuration."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .config import VariantConfig


def resolve_revision(config: VariantConfig, revision: str) -> str:
    from huggingface_hub import HfApi

    resolved = HfApi().model_info(config.source, revision=revision, timeout=30).sha
    if not resolved:
        raise ValueError("Hub did not resolve an immutable source commit")
    return resolved


def download_source(
    config: VariantConfig,
    revision: str,
    destination: Path,
    *,
    probe: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        config.source,
        revision=revision,
        local_dir=destination,
        allow_patterns=list(config.probe.download_patterns) if probe else None,
        ignore_patterns=list(config.probe.ignore_patterns) if probe else None,
        max_workers=4,
    )
    record_source(config, revision, destination, "probe" if probe else "full")
    return destination


def download_probe(config: VariantConfig, revision: str, destination: Path) -> Path:
    return download_source(config, revision, destination, probe=True)


def local_source(config: VariantConfig, revision: str, destination: Path) -> Path:
    """Use an explicitly pinned local snapshot without network or manifest mutation."""
    metadata = json.loads((destination / ".source.json").read_text())
    if (metadata["repo_id"], metadata["revision"]) != (config.source, revision):
        raise ValueError("Local source repository or revision does not match")
    if metadata["files"] != source_inventory(destination):
        raise ValueError("Local source files changed since download")
    return destination


def source_inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".cache" in relative.parts or "__pycache__" in relative.parts:
            continue
        if relative.name in {".source.json", ".source.json.tmp"}:
            continue
        if path.is_symlink():
            raise ValueError("Project originals must be real files, not external cache links")
        if path.is_file():
            with path.open("rb") as stream:
                result[relative.as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def record_source(config: VariantConfig, revision: str, root: Path, selection: str) -> None:
    """Record only completed downloads; migrated partial caches never imply full models."""
    metadata = root / ".source.json"
    previous = json.loads(metadata.read_text()) if metadata.exists() else {}
    if previous and (previous["repo_id"], previous["revision"]) != (config.source, revision):
        raise ValueError("Source directory belongs to a different repository or revision")
    selections = sorted(set(previous.get("selections", [])) | {selection})
    report = {
        "repo_id": config.source,
        "revision": revision,
        "selections": selections,
        "files": source_inventory(root),
    }
    staged = root / ".source.json.tmp"
    staged.write_text(json.dumps(report, indent=2) + "\n")
    staged.replace(metadata)


def package_assets(config: VariantConfig, source: Path, run: Path) -> None:
    assets = run / "assets"
    assets.mkdir()
    for pattern in config.probe.asset_patterns:
        for path in source.glob(pattern):
            if path.is_file():
                destination = assets / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)


def upload(config: VariantConfig, run: Path, runtime: str, revision: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(config.target, repo_type="model", exist_ok=True)
    return str(
        api.upload_folder(
            repo_id=config.target,
            repo_type="model",
            folder_path=run,
            path_in_repo=runtime,
            commit_message=f"Update {runtime} from {revision}",
        )
    )
