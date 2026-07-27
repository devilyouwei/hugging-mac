from __future__ import annotations

import asyncio
from typing import Protocol

import pytest
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import ModelLoadError, UnsupportedCapabilityError


class Detection(Protocol):
    async def detect(self, request: str) -> str: ...


class Detector:
    async def detect(self, request: str) -> str:
        return request.upper()


class DummyInstance(BaseModelInstance):
    def __init__(self, *, fail_load: bool = False) -> None:
        super().__init__()
        self.load_calls = 0
        self.unload_calls = 0
        self.fail_load = fail_load
        self.detector = Detector()
        self.register_capability(Detection, self.detector)

    async def _load(self) -> None:
        self.load_calls += 1
        await asyncio.sleep(0)
        if self.fail_load:
            raise ValueError("runtime failed")

    async def _unload(self) -> None:
        self.unload_calls += 1


async def test_lifecycle_is_idempotent_and_capabilities_are_composed() -> None:
    instance = DummyInstance()

    await asyncio.gather(instance.load(), instance.load())

    assert instance.state is ModelState.READY
    assert instance.load_calls == 1
    assert instance.supports(Detection)
    assert await instance.require(Detection).detect("cat") == "CAT"

    await instance.unload()
    await instance.unload()
    assert instance.state is ModelState.UNLOADED
    assert instance.unload_calls == 1


async def test_missing_capability_has_stable_error() -> None:
    class Chat(Protocol):
        async def chat(self, request: str) -> str: ...

    instance = DummyInstance()

    with pytest.raises(UnsupportedCapabilityError) as caught:
        instance.require(Chat)

    assert caught.value.code == "unsupported_capability"


async def test_failed_load_cleans_up_and_reports_failure() -> None:
    instance = DummyInstance(fail_load=True)

    with pytest.raises(ModelLoadError):
        await instance.load()

    assert instance.state is ModelState.FAILED
    assert instance.unload_calls == 1
    health = await instance.health()
    assert health.status == "unhealthy"
