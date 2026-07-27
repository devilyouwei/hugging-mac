from __future__ import annotations

from hugging_mac_sdk import ModelSdk, ModelState, ReusePolicy, RuntimePolicy
from hugging_mac_sdk.core.instance import BaseModelInstance
from hugging_mac_sdk.core.registry import ModelDefinition
from hugging_mac_sdk.schemas.manifest import ModelManifest, RuntimeSpec


class DummyInstance(BaseModelInstance):
    def __init__(self) -> None:
        super().__init__()
        self.unload_calls = 0

    async def _load(self) -> None:
        return None

    async def _unload(self) -> None:
        self.unload_calls += 1


def _sdk() -> ModelSdk:
    sdk = ModelSdk(runtime_policy=RuntimePolicy(("test-runtime",)))
    sdk.registry.register(
        ModelDefinition(
            manifest=ModelManifest(
                model_id="example/composable-model",
                display_name="Composable Model",
                description="SDK facade test model",
                tags=frozenset({"test"}),
                family="test",
                capabilities=frozenset({"chat"}),
                runtimes=(RuntimeSpec(name="test-runtime"),),
                default_runtime="test-runtime",
            ),
            factory=lambda _: DummyInstance(),
        )
    )
    return sdk


async def test_acquire_reuses_shared_instances_and_catalog_reports_details() -> None:
    sdk = _sdk()

    first = await sdk.acquire("example/composable-model")
    second = await sdk.acquire("example/composable-model")

    assert first.instance is second.instance
    assert first.instance.state is ModelState.READY

    model = (await sdk.catalog.snapshot()).models[0]
    assert model.name == "Composable Model"
    assert model.default_runtime == "test-runtime"
    assert model.instantiated
    assert model.instance_count == 1
    assert model.ready_count == 1
    assert model.instances[0].reference_count == 2

    await first.close()
    assert (await sdk.catalog.snapshot()).models[0].instances[0].reference_count == 1
    await second.close()
    assert (await sdk.catalog.snapshot()).models[0].instances[0].reference_count == 0


async def test_dedicated_handle_unloads_instance_on_close() -> None:
    sdk = _sdk()

    handle = await sdk.acquire(
        "example/composable-model",
        reuse=ReusePolicy.DEDICATED,
    )
    instance = handle.instance
    await handle.close()
    await handle.close()

    assert instance.state is ModelState.UNLOADED
    assert isinstance(instance, DummyInstance)
    assert instance.unload_calls == 1
    assert (await sdk.catalog.snapshot()).models[0].instance_count == 0
