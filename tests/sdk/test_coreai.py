from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from hugging_mac_sdk import (
    ArtifactFormat,
    ConversionRequest,
    ConversionService,
    ConverterRegistry,
    CoreAIProvider,
    PyTorchCoreAIConverter,
    ResolvedResource,
    RuntimePolicy,
    RuntimeSpec,
    UrlFileSource,
    convert_pytorch_to_coreai,
    create_default_runtime_registry,
)
from hugging_mac_sdk.converters.coreai import _publish
from hugging_mac_sdk.errors import UnsupportedRuntimeError


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(closed=0, loop=None, options=None, function=None)

    class Array:
        def __init__(self, value: object) -> None:
            self.value = np.asarray(value)

        def numpy(self) -> np.ndarray:
            return self.value

    state.output = np.array([2.0, 4.0], dtype=np.float32)

    runtime_state = state

    async def infer(
        inputs: dict[str, Array], state: dict[str, Array] | None = None
    ) -> dict[str, Array]:
        assert asyncio.get_running_loop() is runtime_state.loop
        assert inputs["x"].numpy().tolist() == [1.0, 2.0]
        if state is not None:
            state["counter"].value += 1
            return {"y": state["counter"]}
        return {"y": Array(runtime_state.output)}

    def load_function(name: str) -> object:
        state.function = name
        if name == "missing":
            raise ValueError("missing function")
        return infer

    @asynccontextmanager
    async def executable(options: object):
        opened_loop = state.loop = asyncio.get_running_loop()
        state.options = options
        try:
            yield SimpleNamespace(load_function=load_function)
        finally:
            assert asyncio.get_running_loop() is opened_loop
            state.closed += 1

    asset = SimpleNamespace(executable=executable)
    monkeypatch.setitem(
        sys.modules,
        "coreai.authoring",
        SimpleNamespace(AIModelAsset=SimpleNamespace(load=Mock(return_value=asset))),
    )
    monkeypatch.setitem(
        sys.modules,
        "coreai.runtime",
        SimpleNamespace(
            NDArray=Array,
            SpecializationOptions=SimpleNamespace(
                is_supported=lambda: True, cpu_only=lambda: "cpu-only"
            ),
        ),
    )
    monkeypatch.setattr("hugging_mac_sdk.runtime.coreai.coreai_available", lambda: True)
    return state


async def test_runtime_lifetime_outputs_and_named_function(tmp_path: Path, native) -> None:
    session = await CoreAIProvider().create_session(
        tmp_path, device="cpu", options={"function_name": "encode"}
    )
    result = session.run({"x": np.array([1.0, 2.0])})
    native.output[:] = 99
    assert result["y"].tolist() == [2.0, 4.0]
    assert native.options == "cpu-only"
    assert native.function == "encode"
    await session.close()
    await session.close()
    assert native.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.run({})


async def test_failed_function_load_releases_context(tmp_path: Path, native) -> None:
    with pytest.raises(ValueError, match="missing function"):
        await CoreAIProvider().create_session(
            tmp_path, device=None, options={"function_name": "missing"}
        )
    assert native.closed == 1


async def test_invalid_runtime_configuration(tmp_path: Path, native) -> None:
    with pytest.raises(UnsupportedRuntimeError, match="device"):
        await CoreAIProvider().create_session(tmp_path, device="cuda", options={})
    with pytest.raises(ValueError, match="Unknown"):
        await CoreAIProvider().create_session(tmp_path, device=None, options={"typo": True})
    assert native.closed == 0


def test_registry_and_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(create_default_runtime_registry().get("coreai"), CoreAIProvider)
    monkeypatch.setattr("hugging_mac_sdk.runtime.coreai.coreai_available", lambda: False)
    assert not RuntimePolicy().availability(RuntimeSpec(name="coreai")).available


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    program = Mock()

    def save(path: Path) -> None:
        path.mkdir()
        (path / "program.bin").write_bytes(b"converted")

    program.save_asset.side_effect = save
    converter = Mock()
    converter.add_exported_program.return_value = converter
    converter.to_coreai.return_value = program
    module = SimpleNamespace(
        TorchConverter=Mock(return_value=converter), get_decomp_table=lambda: {}
    )
    monkeypatch.setitem(sys.modules, "coreai_torch", module)
    return SimpleNamespace(program=program, converter=converter)


def test_module_export_and_failed_overwrite(tmp_path: Path, exporter) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2).eval()
    output = tmp_path / "model.aimodel"
    convert_pytorch_to_coreai(
        model,
        output,
        example_inputs=(torch.ones(1, 2),),
        input_names=["x"],
        output_names=["y"],
        function_name="encode",
    )
    assert (output / "program.bin").read_bytes() == b"converted"
    assert exporter.converter.add_exported_program.call_args.kwargs["entrypoint_name"] == "encode"
    with pytest.raises(FileExistsError):
        convert_pytorch_to_coreai(model, output)
    exporter.program.save_asset.side_effect = RuntimeError("export failed")
    with pytest.raises(RuntimeError, match="export failed"):
        convert_pytorch_to_coreai(model, output, example_inputs=(torch.ones(1, 2),), overwrite=True)
    assert (output / "program.bin").read_bytes() == b"converted"
    assert not list(tmp_path.glob(".*.convert-*"))


async def test_pt2_conversion_service(tmp_path: Path, exporter) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "model.pt2"
    torch.export.save(
        torch.export.export(torch.nn.Linear(2, 2).eval(), (torch.ones(1, 2),)), source
    )
    request = ConversionRequest(
        source=ResolvedResource(
            path=source,
            size_bytes=source.stat().st_size,
            source=UrlFileSource(url="https://example.com/model.pt2"),
        ),
        source_format=ArtifactFormat.PYTORCH,
        target_format=ArtifactFormat.COREAI,
        output_path=tmp_path / "asset",
        options={"optimize": False},
    )
    registry = ConverterRegistry()
    registry.register(PyTorchCoreAIConverter())
    result = await ConversionService(registry).convert(request)
    assert result.format is ArtifactFormat.COREAI
    assert len(result.digest) == 64
    assert result.size_bytes == len(b"converted")
    exporter.program.optimize.assert_not_called()
    assert not PyTorchCoreAIConverter().supports(
        request.model_copy(update={"source_format": ArtifactFormat.TORCHSCRIPT})
    )


def test_publish_restores_old_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    output = tmp_path / "model.aimodel"
    output.mkdir()
    (output / "old").write_text("keep")
    staged = tmp_path / "staged"
    staged.mkdir()
    replace = os.replace

    def fail_staging(src, dest):
        if src == staged:
            raise OSError("publish failed")
        replace(src, dest)

    monkeypatch.setattr("hugging_mac_sdk.converters.coreai.os.replace", fail_staging)
    with pytest.raises(OSError, match="publish failed"):
        _publish(staged, output, overwrite=True)
    assert (output / "old").read_text() == "keep"


@pytest.mark.smoke
@pytest.mark.parametrize("asset_name", ["linear.aimodel", "asset"])
@pytest.mark.parametrize("device", [None, "cpu"])
async def test_real_coreai_roundtrip(tmp_path: Path, asset_name: str, device: str | None) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("coreai_torch")
    if not CoreAIProvider().is_available():
        pytest.skip("Core AI runtime unavailable")
    model = torch.nn.Linear(2, 3).eval()
    sample = torch.ones(1, 2)
    path = convert_pytorch_to_coreai(
        model,
        tmp_path / asset_name,
        example_inputs=(sample,),
        input_names=["x"],
        output_names=["y"],
    )
    session = await CoreAIProvider().create_session(path, device=device, options={})
    try:
        output = await asyncio.to_thread(session.run, {"x": sample.numpy()})
        np.testing.assert_allclose(
            output["y"], model(sample).detach().numpy(), rtol=1e-4, atol=1e-4
        )
    finally:
        await session.close()


async def test_cancelled_load_releases_model_and_view(tmp_path: Path, native, monkeypatch) -> None:
    from threading import Event

    from hugging_mac_sdk.runtime.coreai import CoreAISession

    entered = Event()
    proceed = Event()
    original = CoreAISession._open
    sessions = []

    def delayed_open(self, artifact, function_name):
        sessions.append(self)
        entered.set()
        assert proceed.wait(timeout=5)
        original(self, artifact, function_name)

    monkeypatch.setattr(CoreAISession, "_open", delayed_open)
    task = asyncio.create_task(CoreAIProvider().create_session(tmp_path, device=None, options={}))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert native.closed == 1
    assert sessions[0]._view is None
    assert sessions[0]._runner is None


async def test_native_state_persists_resets_and_is_session_local(tmp_path, native):
    first = await CoreAIProvider().create_session(tmp_path, device="auto", options={})
    second = await CoreAIProvider().create_session(tmp_path, device="auto", options={})
    # Each session uses its own event loop; keep this fake's assertion in sync.
    inputs = {"x": np.array([1.0, 2.0])}
    try:
        native.loop = first._runner.get_loop()
        original = np.zeros(1, np.float32)
        first.reset_state({"counter": original})
        assert first.run(inputs)["y"].tolist() == [1]
        assert first.run(inputs)["y"].tolist() == [2]
        assert original.tolist() == [0]
        first.reset_state({"counter": original})
        assert first.run(inputs)["y"].tolist() == [1]
        native.loop = second._runner.get_loop()
        second.reset_state({"counter": original})
        assert second.run(inputs)["y"].tolist() == [1]
    finally:
        await first.close()
        await second.close()
    assert first._state == second._state == {}
    with pytest.raises(RuntimeError, match="closed"):
        first.reset_state({})
