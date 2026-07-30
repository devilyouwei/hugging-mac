from __future__ import annotations

from pathlib import Path

MODEL_ROOT = (
    Path(__file__).parents[2]
    / "packages"
    / "hugging_mac_sdk"
    / "src"
    / "hugging_mac_sdk"
    / "models"
)
MODEL_PACKAGES = ("yolov8", "yolov8_pose", "yolov8_seg")
ASR_MODEL_PACKAGES = ("audio8_asr",)
REQUIRED_INTEGRATION_FILES = {
    "__init__.py",
    "config.py",
    "definition.py",
    "instance.py",
    "model.yaml",
}
YOLO_RUNTIME_FILES = {"coreml.py", "onnx.py", "torch.py"}
OLD_ROOT_IMPLEMENTATION_FILES = {
    "assets.py",
    "catalog.py",
    "checkpoint.py",
    "types.py",
    "util.py",
}


def test_yolov8_model_packs_follow_the_integration_layout() -> None:
    for package_name in MODEL_PACKAGES:
        package = MODEL_ROOT / package_name
        root_files = {path.name for path in package.iterdir() if path.is_file()}

        assert root_files >= REQUIRED_INTEGRATION_FILES | YOLO_RUNTIME_FILES
        assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
        assert (package / "utils" / "__init__.py").is_file()


def test_model_instances_compose_runtime_engines() -> None:
    for package_name in MODEL_PACKAGES:
        package = MODEL_ROOT / package_name
        instance_source = (package / "instance.py").read_text()

        assert "Engine(Protocol)" in instance_source
        assert "self._engine" in instance_source
        assert "from .torch" not in instance_source
        assert "from .coreml" not in instance_source
        assert "from .onnx" not in instance_source


def test_model_packs_do_not_import_other_model_packs() -> None:
    all_packages = MODEL_PACKAGES + ASR_MODEL_PACKAGES
    for package_name in all_packages:
        package = MODEL_ROOT / package_name
        other_packages = set(all_packages) - {package_name}
        source = "\n".join(path.read_text() for path in package.rglob("*.py"))
        for other_package in other_packages:
            assert f"hugging_mac_sdk.models.{other_package}" not in source
            assert f"from ..{other_package}" not in source


def test_audio8_asr_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "audio8_asr"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {
        "converter.py",
        "coreml.py",
        "resources.py",
        "torch.py",
    }
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
    assert (package / "utils" / "__init__.py").is_file()
