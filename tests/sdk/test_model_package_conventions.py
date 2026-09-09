from __future__ import annotations

import ast
from pathlib import Path

import yaml

MODEL_ROOT = (
    Path(__file__).parents[2]
    / "packages"
    / "hugging_mac_sdk"
    / "src"
    / "hugging_mac_sdk"
    / "models"
)
REPOSITORY_ROOT = Path(__file__).parents[2]
WEB_BACKEND_ROOT = REPOSITORY_ROOT / "apps" / "web" / "backend" / "src"
MODEL_PACKAGES = ("yolov8", "yolov8_pose", "yolov8_seg")
ASR_MODEL_PACKAGES = ("audio8_asr", "sensevoice", "qwen3_asr", "nemotron_3_5_asr")
TTS_MODEL_PACKAGES = (
    "audio8_tts",
    "kokoro",
    "moss_tts_nano",
    "qwen3_tts",
)
LLM_MODEL_PACKAGES = ("qwen3_5",)
OCR_MODEL_PACKAGES = ("glm_ocr", "got_ocr2", "unlimited_ocr")
DOCUMENT_LAYOUT_MODEL_PACKAGES = ("pp_doclayout_v3",)
VAD_MODEL_PACKAGES = ("silero",)
SPEECH_ENHANCEMENT_MODEL_PACKAGES = ("deepfilternet3",)
REQUIRED_INTEGRATION_FILES = {
    "__init__.py",
    "config.py",
    "definition.py",
    "instance.py",
    "model.yaml",
    "readme.md",
}
YOLO_RUNTIME_FILES = {"coreml.py", "onnx.py", "torch.py"}
OLD_ROOT_IMPLEMENTATION_FILES = {
    "assets.py",
    "catalog.py",
    "checkpoint.py",
    "types.py",
    "util.py",
}
DERIVED_PACKAGE_SUFFIXES = (
    "_4bit",
    "_8bit",
    "_bf16",
    "_coreml",
    "_fp16",
    "_fp32",
    "_int8",
    "_mlx",
    "_onnx",
)


def test_model_package_names_do_not_encode_runtime_or_artifact_details() -> None:
    package_names = {
        path.name
        for path in MODEL_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(("_", "."))
    }

    for package_name in package_names:
        assert not package_name.endswith(DERIVED_PACKAGE_SUFFIXES)


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
    all_packages = (
        MODEL_PACKAGES
        + ASR_MODEL_PACKAGES
        + TTS_MODEL_PACKAGES
        + LLM_MODEL_PACKAGES
        + OCR_MODEL_PACKAGES
        + DOCUMENT_LAYOUT_MODEL_PACKAGES
        + VAD_MODEL_PACKAGES
        + SPEECH_ENHANCEMENT_MODEL_PACKAGES
    )
    for package_name in all_packages:
        package = MODEL_ROOT / package_name
        other_packages = set(all_packages) - {package_name}
        source = "\n".join(path.read_text() for path in package.rglob("*.py"))
        for other_package in other_packages:
            assert f"hugging_mac_sdk.models.{other_package}" not in source
            assert f"from ..{other_package}" not in source


def test_instance_configs_do_not_redeclare_common_manifest_facts() -> None:
    forbidden_suffixes = (
        "_MODEL_ID",
        "_REPO_ID",
        "_REVISION",
        "_VARIANT",
        "_VARIANTS",
        "_FILENAME",
        "_FILENAMES",
        "_SHA256",
        "_REQUIRED_FILES",
        "_RELEASE",
    )
    for config_path in MODEL_ROOT.glob("*/config.py"):
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
        names = {
            target.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
            if isinstance(target, ast.Name)
        }
        forbidden = {name for name in names if name.endswith(forbidden_suffixes)}
        assert not forbidden, f"{config_path}: move common facts to model.yaml: {forbidden}"
        source = config_path.read_text(encoding="utf-8")
        assert "variant: Literal[" not in source


def test_model_yaml_uses_unpinned_required_file_manifests() -> None:
    forbidden_keys = {"revision", "expected_sha256", "file_sha256", "sha256"}

    def mappings(value: object):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from mappings(child)
        elif isinstance(value, list):
            for child in value:
                yield from mappings(child)

    for yaml_path in MODEL_ROOT.glob("*/model.yaml"):
        text = yaml_path.read_text(encoding="utf-8")
        package = yaml.safe_load(text)
        all_mappings = tuple(mappings(package))
        manifest = package["manifest"]
        manifest_keys = tuple(manifest)
        lines = text.splitlines()
        expected_identity_keys = (
            "schema_version",
            "model_id",
            "display_name",
            "description",
            "tags",
            "family",
            "capabilities",
            "source_url",
            "license",
        )
        license_line = next(
            index for index, line in enumerate(lines) if line.startswith("  license:")
        )
        default_runtime_line = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("  default_runtime:")
        )
        artifacts_line = lines.index("artifacts:")

        assert manifest_keys[: len(expected_identity_keys)] == expected_identity_keys
        assert manifest_keys.index("license") + 1 == manifest_keys.index("runtimes"), (
            f"{yaml_path}: license must end the manifest identity section"
        )
        assert lines[license_line + 1] == "", (
            f"{yaml_path}: license must end the first visual section"
        )
        assert lines[license_line + 2].startswith("  runtimes:"), (
            f"{yaml_path}: exactly one blank line must follow license"
        )
        assert manifest_keys.index("runtimes") + 1 == manifest_keys.index(
            "default_runtime"
        ), f"{yaml_path}: default_runtime must immediately follow runtimes"
        assert manifest_keys.index("default_runtime") + 1 == manifest_keys.index(
            "variants"
        ), f"{yaml_path}: variants must immediately follow default_runtime"
        assert lines[default_runtime_line + 1].startswith("  variants:"), (
            f"{yaml_path}: variants must have no blank line after default_runtime"
        )
        assert manifest_keys.index("variants") + 1 == manifest_keys.index(
            "default_variant"
        ), f"{yaml_path}: default_variant must immediately follow variants"
        assert "\n\nartifacts:\n" in text, (
            f"{yaml_path}: manifest and artifacts must be separated by a blank line"
        )
        assert lines[artifacts_line - 1] == "" and lines[artifacts_line - 2] != "", (
            f"{yaml_path}: exactly one blank line must precede artifacts"
        )
        assert "" not in lines[artifacts_line + 1 :], (
            f"{yaml_path}: artifact entries must not contain blank lines"
        )

        for runtime in manifest["runtimes"]:
            expected_runtime_keys = (
                "name",
                "devices",
                "dtypes",
                *(('quantizations',) if "quantizations" in runtime else ()),
                "platforms",
                "architectures",
                "required_modules",
            )
            assert tuple(runtime) == expected_runtime_keys, (
                f"{yaml_path}: runtime fields are not in canonical order"
            )
        for variant in manifest["variants"]:
            expected_variant_keys = (
                "name",
                "display_name",
                *(("description",) if "description" in variant else ()),
                *(("metadata",) if "metadata" in variant else ()),
            )
            assert tuple(variant) == expected_variant_keys, (
                f"{yaml_path}: variant fields are not in canonical order"
            )

        assert "resources" not in manifest, (
            f"{yaml_path}: sources belong to artifacts, not the manifest"
        )
        for variant in manifest["variants"]:
            assert "resources" not in variant, (
                f"{yaml_path}: sources belong to artifacts, not variants"
            )
            assert "shared_artifacts" not in variant, (
                f"{yaml_path}: shared dependencies belong to artifacts.required_shares"
            )

        shared_ids = {
            artifact["artifact_id"]
            for artifact in package["artifacts"]
            if artifact.get("shared") is True
        }
        referenced_shared_ids = {
            artifact_id
            for artifact in package["artifacts"]
            if artifact.get("shared") is not True
            for artifact_id in artifact.get("required_shares", [])
        }
        assert referenced_shared_ids == shared_ids, (
            f"{yaml_path}: scoped artifacts must reference every declared shared artifact"
        )

        for item in all_mappings:
            assert not forbidden_keys & item.keys(), (
                f"{yaml_path}: revisions and hashes are forbidden"
            )
            if item.get("kind") == "huggingface":
                assert item.get("filename") or item.get("allow_patterns"), (
                    f"{yaml_path}: Hugging Face sources must select their download files"
                )

        for artifact in package["artifacts"]:
            keys = tuple(artifact)
            if artifact.get("shared") is True:
                expected_artifact_keys = (
                    "artifact_id",
                    "shared",
                    "format",
                    "kind",
                    "path",
                    "required_files",
                    *(("source",) if "source" in artifact else ()),
                    *(("metadata",) if "metadata" in artifact else ()),
                )
            else:
                expected_artifact_keys = (
                    "artifact_id",
                    *(("convert",) if artifact.get("convert") is True else ()),
                    "variant",
                    "runtime",
                    "format",
                    "kind",
                    "path",
                    *(("required_files",) if "required_files" in artifact else ()),
                    *(("required_shares",) if "required_shares" in artifact else ()),
                    *(("source",) if "source" in artifact else ()),
                    *(("metadata",) if "metadata" in artifact else ()),
                )
            assert keys == expected_artifact_keys, (
                f"{yaml_path}: artifact fields are not in canonical order: {keys}"
            )
            assert keys.index("kind") < keys.index("path"), (
                f"{yaml_path}: artifact kind must precede path"
            )
            if artifact.get("shared") is True:
                assert keys.index("shared") == 1, (
                    f"{yaml_path}: shared must immediately follow artifact_id"
                )
            if artifact.get("convert") is True:
                assert keys.index("convert") == 1, (
                    f"{yaml_path}: convert must immediately follow artifact_id"
                )
            if artifact.get("shared") is True:
                assert "variant" not in artifact
                assert "runtime" not in artifact
                assert not artifact.get("required_shares")
            else:
                assert artifact.get("variant")
                assert artifact.get("runtime")
            if artifact.get("required_shares"):
                assert keys.index("required_shares") == keys.index("required_files") + 1, (
                    f"{yaml_path}: required_shares must stay next to required_files"
                )
            if artifact["kind"] == "directory":
                assert artifact.get("required_files"), (
                    f"{yaml_path}: directory artifacts need complete required_files"
                )

        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("required_files:", "required_modules:", "required_shares:")):
                assert "[" in stripped
            if stripped.startswith("allow_patterns:"):
                assert "[" not in stripped


def test_web_apps_do_not_import_model_private_configs() -> None:
    for path in WEB_BACKEND_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        private_imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("hugging_mac_sdk.models.")
            and node.module.endswith(".config")
        ]
        assert not private_imports, (
            f"{path} must select models through its app manifest or registry"
        )


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


def test_sensevoice_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "sensevoice"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {
        "resources.py",
        "torch.py",
    }
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
    assert (package / "utils" / "__init__.py").is_file()


def test_audio8_tts_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "audio8_tts"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {
        "mlx.py",
        "resources.py",
        "torch.py",
    }
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
    assert (package / "utils" / "__init__.py").is_file()


def test_kokoro_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "kokoro"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {
        "resources.py",
        "coreml.py",
        "torch.py",
    }
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
    assert (package / "utils" / "__init__.py").is_file()


def test_moss_tts_nano_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "moss_tts_nano"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {"resources.py", "mlx.py"}
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
    assert (package / "utils" / "__init__.py").is_file()


def test_qwen3_tts_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "qwen3_tts"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {"resources.py", "mlx.py"}
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
    assert (package / "utils" / "__init__.py").is_file()


def test_qwen3_5_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "qwen3_5"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {"resources.py", "mlx.py"}
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
    assert (package / "utils" / "__init__.py").is_file()


def test_unlimited_ocr_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "unlimited_ocr"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {"resources.py", "mlx.py"}
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES


def test_pp_doclayout_v3_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "pp_doclayout_v3"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {
        "converter.py",
        "coreml.py",
        "resources.py",
        "torch.py",
    }
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
    assert (package / "utils" / "__init__.py").is_file()


def test_silero_follows_the_integration_layout() -> None:
    package = MODEL_ROOT / "silero"
    root_files = {path.name for path in package.iterdir() if path.is_file()}

    assert root_files >= REQUIRED_INTEGRATION_FILES | {
        "coreml.py",
        "resources.py",
        "onnx.py",
    }
    assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
    assert (package / "utils" / "__init__.py").is_file()


def test_prebuilt_audio_coreml_models_follow_the_integration_layout() -> None:
    for package_name in ("deepfilternet3", "qwen3_asr", "nemotron_3_5_asr"):
        package = MODEL_ROOT / package_name
        root_files = {path.name for path in package.iterdir() if path.is_file()}

        assert root_files >= REQUIRED_INTEGRATION_FILES | {"coreml.py", "resources.py"}
        assert not root_files & OLD_ROOT_IMPLEMENTATION_FILES
