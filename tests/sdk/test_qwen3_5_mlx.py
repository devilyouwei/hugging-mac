from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from hugging_mac_sdk.models.qwen3_5 import QWEN3_5_DEFINITION, QWEN3_5_MANIFEST
from hugging_mac_sdk.models.qwen3_5.config import Qwen35MlxInstanceConfig
from hugging_mac_sdk.models.qwen3_5.mlx import MlxQwen35Engine
from hugging_mac_sdk.models.qwen3_5.resources import Qwen35MlxResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource


def test_2b_optiq_artifact_owns_source_and_variant_metadata() -> None:
    variant = QWEN3_5_MANIFEST.get_variant("2b")

    assert QWEN3_5_DEFINITION.supported_runtimes == ("mlx",)
    assert tuple(item.name for item in QWEN3_5_MANIFEST.variants) == ("9b", "4b", "2b")
    source = QWEN3_5_DEFINITION.get_artifact("mlx", "mlx-optiq-4bit", "2b").source
    assert isinstance(source, HuggingFaceSource)
    assert source.repo_id == "mlx-community/Qwen3.5-2B-OptiQ-4bit"
    assert source.revision == "main"
    assert variant.metadata == {
        "parameter_count_billions": 2,
        "quantization_bits": 4,
        "quantization_group_size": 64,
        "disk_size_bytes": 1_400_000_000,
        "multimodal": True,
        "quantization_method": "optiq",
        "mixed_precision_8bit_layers": 56,
        "mixed_precision_4bit_layers": 130,
    }


def test_2b_optiq_snapshot_requires_the_visual_sidecar(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    artifact = QWEN3_5_DEFINITION.get_artifact("mlx", "mlx-optiq-4bit", "2b")
    tokenizer = QWEN3_5_DEFINITION.shared_artifacts[0]
    for filename in artifact.required_files:
        path = model / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model-data")

    resolver = Qwen35MlxResourceResolver(
        HuggingFaceSource(
            repo_id="mlx-community/Qwen3.5-2B-OptiQ-4bit",
            revision="main",
        ),
        Qwen35MlxInstanceConfig(source_path=model, variant="2b"),
        manifest=QWEN3_5_MANIFEST,
        artifact=artifact,
        tokenizer_artifact=tokenizer,
    )

    assert resolver.status().artifacts[0].available
    (model / "optiq" / "optiq_vision.safetensors").unlink()
    assert not resolver.status().artifacts[0].available


async def test_lazy_mlx_load_materializes_before_merged_view_closes(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    (artifact / "optiq").mkdir(parents=True)
    tokenizer.mkdir()
    (artifact / "config.json").write_text("{}")
    (artifact / "optiq" / "optiq_vision.safetensors").touch()
    (tokenizer / "tokenizer.json").write_text("{}")

    observed: dict[str, object] = {}

    class FakeModel:
        def __init__(self, view: Path) -> None:
            self.view = view

        @staticmethod
        def sanitize(weights):
            return weights

        @staticmethod
        def load_weights(weights, *, strict: bool) -> None:
            assert weights == [("vision_tower.weight", "vision")]
            assert not strict

        def parameters(self):
            return {"weight": self.view / "config.json"}

    mlx_vlm = ModuleType("mlx_vlm")

    def fake_load(path: str, *, lazy: bool, strict: bool):
        view = Path(path)
        assert (view / "config.json").is_file()
        assert (view / "tokenizer.json").is_file()
        assert lazy and not strict
        observed["view"] = view
        return FakeModel(view), object()

    mlx_vlm.load = fake_load  # type: ignore[attr-defined]
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_core.load = lambda _path: {"vision_tower.weight": "vision"}  # type: ignore[attr-defined]

    def fake_eval(parameters) -> None:
        observed["evaluated_while_present"] = parameters["weight"].is_file()

    mlx_core.eval = fake_eval  # type: ignore[attr-defined]
    mlx.core = mlx_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)

    resources = type("Resources", (), {"tokenizer_path": tokenizer})()
    engine = MlxQwen35Engine(
        Qwen35MlxInstanceConfig(source_path=artifact, tokenizer_path=tokenizer, variant="2b"),
        resources,  # type: ignore[arg-type]
    )
    await engine.load(artifact)

    assert observed["evaluated_while_present"] is True
    assert not Path(observed["view"]).exists()  # type: ignore[arg-type]
