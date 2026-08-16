from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from hugging_mac_sdk import (
    ArtifactFormat,
    ArtifactKind,
    ManifestError,
    ModelArtifact,
    ModelDefinition,
    ModelLoadError,
    ModelManifest,
    ModelRegistry,
    ModelVariantSpec,
    RuntimeBackend,
    RuntimePolicy,
    RuntimeRegistry,
    RuntimeSession,
    RuntimeSpec,
    UnsupportedRuntimeError,
    UrlFileSource,
    load_model_config,
)
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.core.manager import InstanceManager, ReusePolicy
from hugging_mac_sdk.runtime.coreml import CoreMLSession


class DummyInstance(BaseModelInstance):
    def __init__(self, options: dict[str, object]) -> None:
        super().__init__(
            model_id="example/multi-runtime",
            revision="main",
            variant=str(options["variant"]),
            runtime=str(options["runtime"]),
            device=str(options["device"]) if "device" in options else None,
        )

    async def _load(self) -> None:
        return None

    async def _unload(self) -> None:
        return None


class FailingInstance(DummyInstance):
    async def _load(self) -> None:
        raise RuntimeError("backend failed")


class DummyBackend(RuntimeBackend):
    @property
    def name(self) -> str:
        return "dummy"

    def is_available(self) -> bool:
        return True

    async def create_session(
        self,
        artifact: Path,
        *,
        device: str | None,
        options: Mapping[str, object],
    ) -> RuntimeSession:
        raise NotImplementedError


class MultifunctionCoreMlModel:
    def __init__(self) -> None:
        self.received: dict[str, object] | None = None

    def get_spec(self) -> object:
        class Description:
            input: tuple[object, ...] = ()

        class Spec:
            description = Description()

        return Spec()

    def predict(self, inputs: dict[str, object]) -> dict[str, object]:
        self.received = inputs
        return {"hidden": "output"}


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_id="example/multi-runtime",
        display_name="Multi Runtime",
        family="test",
        capabilities=frozenset({"test"}),
        runtimes=(
            RuntimeSpec(name="alpha", devices=("cpu",)),
            RuntimeSpec(name="beta", devices=("gpu", "cpu")),
        ),
        default_runtime="alpha",
    )


def test_runtime_policy_normalizes_mps_alias_and_keeps_configured_order() -> None:
    manifest = ModelManifest(
        model_id="example/apple-runtimes",
        display_name="Apple Runtimes",
        family="test",
        capabilities=frozenset({"test"}),
        runtimes=tuple(
            RuntimeSpec(name=name, required_modules=("sys",))
            for name in ("onnx", "pytorch-mps", "mlx", "coreml")
        ),
    )

    policy = RuntimePolicy(("coreml", "mlx", "mps", "onnx"))

    assert policy.candidates(manifest) == (
        "coreml",
        "mlx",
        "pytorch-mps",
        "onnx",
    )


def _definition() -> ModelDefinition:
    return ModelDefinition(
        manifest=_manifest(),
        runtime_factories={
            "alpha": DummyInstance,
            "beta": DummyInstance,
        },
        artifacts=(
            ModelArtifact(
                artifact_id="tokenizer",
                runtime="alpha",
                format=ArtifactFormat.TOKENIZER,
                path=Path("example/shared/tokenizer"),
                kind=ArtifactKind.DIRECTORY,
                source=UrlFileSource(url="https://models.example/tokenizer.zip"),
                shared=True,
            ),
            ModelArtifact(
                artifact_id="alpha-model",
                runtime="alpha",
                format=ArtifactFormat.ONNX,
                path=Path("example/alpha/model.onnx"),
            ),
            ModelArtifact(
                artifact_id="beta-model",
                runtime="beta",
                format=ArtifactFormat.COREML,
                path=Path("example/beta/model.mlpackage"),
                kind=ArtifactKind.DIRECTORY,
            ),
        ),
    )


def test_registry_reports_only_bound_runtimes_and_resolves_artifacts(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(storage_root=tmp_path)
    registry.register(_definition())

    assert registry.list_models() == (registry.get("example/multi-runtime"),)
    assert registry.supported_runtimes("example/multi-runtime") == ("alpha", "beta")
    shared = registry.get("example/multi-runtime").shared_artifacts
    assert tuple(item.artifact_id for item in shared) == ("tokenizer",)
    assert tuple(
        item.artifact_id
        for item in registry.get_artifacts("example/multi-runtime", "alpha")
    ) == ("alpha-model",)
    assert registry.get_artifact("example/multi-runtime", "beta").artifact_id == "beta-model"
    assert registry.resolve_artifact_path(
        "example/multi-runtime",
        "alpha",
    ) == tmp_path / "example/alpha/model.onnx"


def test_coreml_session_passes_inputs_for_multifunction_packages() -> None:
    model = MultifunctionCoreMlModel()
    session = CoreMLSession(model, "cpu-and-neural-engine")

    output = session.run({"audios": "features", "attn_mask": "mask"})

    assert model.received == {"audios": "features", "attn_mask": "mask"}
    assert output == {"hidden": "output"}


def test_coreml_session_detaches_array_outputs_from_native_storage() -> None:
    class ArrayModel:
        def __init__(self) -> None:
            self.output = np.asarray([1.0, 2.0], dtype=np.float32)

        def predict(self, inputs: dict[str, object]) -> dict[str, object]:
            del inputs
            return {"hidden": self.output}

    model = ArrayModel()
    session = CoreMLSession(model, "cpu-only")

    output = session.run({})
    model.output[0] = 99.0

    assert np.asarray(output["hidden"]).tolist() == [1.0, 2.0]


def test_definition_rejects_declared_runtime_without_implementation() -> None:
    with pytest.raises(ManifestError):
        ModelDefinition(
            manifest=_manifest(),
            runtime_factories={"alpha": DummyInstance},
        )


async def test_manager_auto_device_info_and_runtime_switch() -> None:
    registry = ModelRegistry()
    registry.register(_definition())
    manager = InstanceManager(registry, RuntimePolicy(("beta", "alpha")))

    beta = await manager.load(
        "example/multi-runtime",
        runtime="auto",
        device="gpu",
    )
    assert beta.state is ModelState.READY
    assert beta.info().runtime == "beta"
    assert beta.info().device == "gpu"

    alpha = await manager.switch_runtime(
        str(beta.instance_id),
        "alpha",
        device="cpu",
    )
    assert beta.info().state is ModelState.UNLOADED
    assert alpha.state is ModelState.READY
    assert alpha.info().runtime == "alpha"
    assert alpha.info().device == "cpu"
    assert alpha.info().variant == "default"

    await manager.unload_all()
    assert alpha.info().state is ModelState.UNLOADED


async def test_manager_keeps_variants_distinct_and_switches_safely() -> None:
    manifest = ModelManifest(
        model_id="example/variants",
        display_name="Variant Model",
        family="test",
        capabilities=frozenset({"test"}),
        runtimes=(RuntimeSpec(name="alpha"),),
        variants=(
            ModelVariantSpec(name="n", display_name="Nano"),
            ModelVariantSpec(name="s", display_name="Small"),
            ModelVariantSpec(name="m", display_name="Medium"),
        ),
        default_variant="n",
    )
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            manifest=manifest,
            runtime_factories={"alpha": DummyInstance},
        )
    )
    manager = InstanceManager(registry)

    nano = await manager.create(
        "example/variants",
        variant="n",
        reuse=ReusePolicy.SHARED,
    )
    nano_again = await manager.create(
        "example/variants",
        variant="n",
        reuse=ReusePolicy.SHARED,
    )
    small = await manager.create(
        "example/variants",
        variant="s",
        reuse=ReusePolicy.SHARED,
    )

    assert nano is nano_again
    assert nano is not small
    assert nano.info().variant == "n"
    assert small.info().variant == "s"

    medium = await manager.switch_variant(str(small.instance_id), "m")
    assert small.info().state is ModelState.UNLOADED
    assert medium.info().variant == "m"
    assert (await manager.snapshot(str(medium.instance_id))).variant == "m"


async def test_failed_runtime_switch_keeps_original_instance_ready() -> None:
    manifest = ModelManifest(
        model_id="example/multi-runtime",
        display_name="Safe Switch",
        family="test",
        capabilities=frozenset({"test"}),
        runtimes=(RuntimeSpec(name="alpha"), RuntimeSpec(name="broken")),
    )
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            manifest=manifest,
            runtime_factories={
                "alpha": DummyInstance,
                "broken": FailingInstance,
            },
        )
    )
    manager = InstanceManager(registry)
    original = await manager.load("example/multi-runtime", runtime="alpha")

    with pytest.raises(ModelLoadError):
        await manager.switch_runtime(str(original.instance_id), "broken")

    assert original.info().state is ModelState.READY
    assert await manager.get(str(original.instance_id)) is original


async def test_manager_reports_load_and_unload_metrics() -> None:
    registry = ModelRegistry()
    registry.register(_definition())
    manager = InstanceManager(registry)

    instance = await manager.load("example/multi-runtime", runtime="alpha")
    snapshot = await manager.snapshot(str(instance.instance_id))

    assert snapshot.load_metrics is not None
    assert snapshot.load_metrics.operation == "load"
    assert snapshot.load_metrics.duration_ms >= 0
    assert snapshot.load_metrics.process_rss_before_bytes is not None
    assert snapshot.load_metrics.process_rss_after_bytes is not None

    result = await manager.unload_with_metrics(str(instance.instance_id))

    assert result is not None
    assert result.state is ModelState.UNLOADED
    assert result.metrics.operation == "unload"
    assert result.metrics.duration_ms >= 0
    assert result.metrics.memory_released_bytes is not None


async def test_manager_rejects_unload_while_instance_is_retained() -> None:
    registry = ModelRegistry()
    registry.register(_definition())
    manager = InstanceManager(registry)
    instance = await manager.load("example/multi-runtime", runtime="alpha")
    instance_id = str(instance.instance_id)
    await manager.retain(instance_id)

    with pytest.raises(UnsupportedRuntimeError, match="retained"):
        await manager.unload_with_metrics(instance_id)

    assert await manager.get(instance_id) is instance
    await manager.release(instance_id)
    assert await manager.unload(instance_id)


def test_yaml_model_config_is_data_only(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
manifest:
  model_id: example/yaml
  display_name: YAML Model
  family: test
  capabilities: [embedding]
  runtimes:
    - name: onnx
      devices: [cpu]
artifacts:
  - artifact_id: model
    runtime: onnx
    format: onnx
    path: example/yaml/model.onnx
""".strip(),
        encoding="utf-8",
    )

    config = load_model_config(config_path)

    assert config.manifest.model_id == "example/yaml"
    assert config.artifacts[0].path == Path("example/yaml/model.onnx")


def test_runtime_backend_registry_is_independent_from_model_registry() -> None:
    runtimes = RuntimeRegistry()
    backend = DummyBackend()

    runtimes.register(backend)

    assert runtimes.get("dummy") is backend
    assert runtimes.list() == (backend,)
