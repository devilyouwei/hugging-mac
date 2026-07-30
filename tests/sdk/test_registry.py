from __future__ import annotations

import pytest
from hugging_mac_sdk.core.instance import BaseModelInstance
from hugging_mac_sdk.core.manager import InstanceManager, ReusePolicy
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.core.runtime_policy import RuntimePolicy
from hugging_mac_sdk.errors import (
    RegistrationConflictError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.schemas.manifest import ModelManifest, RuntimeSpec


class DummyInstance(BaseModelInstance):
    async def _load(self) -> None:
        return None

    async def _unload(self) -> None:
        return None


class MissingResourceInstance(DummyInstance):
    async def _resolve(self) -> None:
        raise ResourceNotFoundError("preferred runtime artifact is missing")


def manifest(
    model_id: str = "example/model",
    *,
    capabilities: frozenset[str] = frozenset({"object-detection"}),
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=model_id,
        family="test",
        capabilities=capabilities,
        runtimes=(RuntimeSpec(name="pytorch-mps"),),
    )


def test_registry_filters_and_rejects_conflicts() -> None:
    registry = ModelRegistry()
    definition = ModelDefinition(manifest(), lambda _: DummyInstance())
    registry.register(definition)

    assert registry.get("example/model") is definition
    assert registry.list(capability="object-detection") == (definition,)
    assert registry.list(capability="chat") == ()
    assert registry.list(runtime="pytorch-mps") == (definition,)

    with pytest.raises(RegistrationConflictError):
        registry.register(definition)
    with pytest.raises(UnsupportedRuntimeError):
        registry.create_instance("example/model", runtime="mlx")


def test_registry_resolves_single_revision_without_main_alias() -> None:
    registry = ModelRegistry()
    pinned = manifest().model_copy(update={"revision": "pinned-revision"})
    definition = ModelDefinition(pinned, lambda _: DummyInstance())
    registry.register(definition)

    assert registry.get("example/model") is definition
    assert registry.create_instance("example/model") is not None


def test_definition_rejects_conflicting_runtime_arguments() -> None:
    definition = ModelDefinition(manifest(), lambda _: DummyInstance())

    with pytest.raises(UnsupportedRuntimeError):
        definition.create(
            {"runtime": "pytorch-mps"},
            runtime="coreml",
        )


async def test_instance_manager_applies_reuse_policy() -> None:
    registry = ModelRegistry()
    registry.register(ModelDefinition(manifest(), lambda _: DummyInstance()))
    manager = InstanceManager(registry)

    shared_a = await manager.create(
        "example/model",
        runtime="pytorch-mps",
        options={"dtype": "float16"},
        reuse=ReusePolicy.SHARED,
    )
    shared_b = await manager.create(
        "example/model",
        runtime="pytorch-mps",
        options={"dtype": "float16"},
        reuse=ReusePolicy.SHARED,
    )
    dedicated = await manager.create(
        "example/model",
        runtime="pytorch-mps",
        options={"dtype": "float16"},
        reuse=ReusePolicy.DEDICATED,
    )

    assert shared_a is shared_b
    assert dedicated is not shared_a
    assert await manager.unload(str(shared_a.instance_id))
    assert not await manager.unload(str(shared_a.instance_id))


async def test_instance_manager_auto_falls_back_to_runtime_with_local_artifact() -> None:
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            ModelManifest(
                model_id="example/fallback-model",
                display_name="Fallback Model",
                family="test",
                capabilities=frozenset({"object-detection"}),
                runtimes=(
                    RuntimeSpec(name="preferred"),
                    RuntimeSpec(name="fallback"),
                ),
            ),
            runtime_factories={
                "preferred": lambda _: MissingResourceInstance(),
                "fallback": lambda _: DummyInstance(),
            },
        )
    )
    manager = InstanceManager(
        registry,
        runtime_policy=RuntimePolicy(("preferred", "fallback")),
    )

    instance = await manager.load("example/fallback-model", runtime="auto")
    snapshot = await manager.snapshot(str(instance.instance_id))

    assert instance.state.value == "ready"
    assert snapshot.runtime == "fallback"
    assert len(await manager.list()) == 1
