"""Both source revision lookup and download honor the same endpoint."""

from types import SimpleNamespace

from hugging_mac_conversion.registry import get_model
from hugging_mac_conversion.shared import hub


def test_endpoint_from_dotenv_after_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    (tmp_path / ".env").write_text("HF_ENDPOINT=https://dotenv.example/\n")
    calls = []

    def api(**kwargs):
        calls.append(kwargs["endpoint"])
        return SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(sha="commit"))

    def download(*args, **kwargs):
        calls.append(kwargs["endpoint"])

    monkeypatch.setattr("huggingface_hub.HfApi", api)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    config = get_model("vela").config.variant("3-fast")
    assert hub.resolve_revision(config, "main") == "commit"
    hub.download_source(config, "commit", tmp_path / "source")
    assert calls == ["https://dotenv.example"] * 2
    monkeypatch.setenv("HF_ENDPOINT", "https://process.example")
    hub.resolve_revision(config, "main")
    assert calls[-1] == "https://process.example"
