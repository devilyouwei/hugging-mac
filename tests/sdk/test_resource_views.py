from pathlib import Path

import pytest
from hugging_mac_sdk.core.resources import artifact_available
from hugging_mac_sdk.resources.views import merged_directory_view
from hugging_mac_sdk.schemas.artifact import ArtifactKind, ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_merged_directory_view_uses_non_copying_links(tmp_path: Path) -> None:
    model = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    model.mkdir()
    tokenizer.mkdir()
    (model / "config.json").write_text("model", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("tokenizer", encoding="utf-8")

    with merged_directory_view(model, (tokenizer,)) as view:
        assert (view / "config.json").is_symlink()
        assert (view / "tokenizer.json").is_symlink()
        assert (view / "tokenizer.json").read_text(encoding="utf-8") == "tokenizer"

    assert not view.exists()


def test_shared_overlay_shadows_legacy_file_without_mutating_it(tmp_path: Path) -> None:
    model = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    model.mkdir()
    tokenizer.mkdir()
    (model / "tokenizer.json").write_text("legacy", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("shared", encoding="utf-8")

    with merged_directory_view(model, (tokenizer,)) as view:
        assert (view / "tokenizer.json").read_text(encoding="utf-8") == "shared"

    assert (model / "tokenizer.json").read_text(encoding="utf-8") == "legacy"


def test_merged_directory_view_resolves_relative_model_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "model" / "config.json").write_text("model", encoding="utf-8")
    (tmp_path / "tokenizer" / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with merged_directory_view(Path("model"), (Path("tokenizer"),)) as view:
        assert (view / "config.json").read_text(encoding="utf-8") == "model"
        assert (view / "tokenizer.json").read_text(encoding="utf-8") == "tokenizer"


def test_shared_snapshot_is_unavailable_until_declared_files_exist(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    artifact = ModelArtifact(
        artifact_id="tokenizer",
        runtime="mlx",
        format=ArtifactFormat.TOKENIZER,
        path=Path("tokenizer"),
        kind=ArtifactKind.DIRECTORY,
        shared=True,
        source=HuggingFaceSource(
            repo_id="example/model",
            allow_patterns=("tokenizer.json", "speech_tokenizer/*.safetensors"),
        ),
    )

    assert not artifact_available(artifact, tokenizer)
    (tokenizer / "tokenizer.json").touch()
    (tokenizer / "speech_tokenizer").mkdir()
    (tokenizer / "speech_tokenizer" / "model.safetensors").touch()
    assert artifact_available(artifact, tokenizer)
