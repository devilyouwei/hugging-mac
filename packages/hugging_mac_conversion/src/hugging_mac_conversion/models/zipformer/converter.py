"""Export all three native batch-one Core ML ML Programs from pinned FP32 weights."""

from __future__ import annotations

import hashlib
import importlib
import shutil
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from urllib.request import urlopen

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import Tensor, nn

from ...shared import artifacts, coreml
from ...shared.huggingface import download_endpoint
from .definition import DEFINITION, PARTS, VARIANT


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(source: Path) -> None:
    for name, digest in VARIANT.source_files.items():
        path = source / name
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"Pinned upstream file is missing or changed: {name}")


def prepare_sources(root: Path, *, offline: bool) -> tuple[Path, Path]:
    source = artifacts.source_directory(
        root, DEFINITION.name, DEFINITION.default_variant, VARIANT.revision
    )
    if not all((source / name).is_file() for name in VARIANT.source_files) and not offline:
        snapshot_download(
            VARIANT.source,
            revision=VARIANT.revision,
            endpoint=download_endpoint(),
            allow_patterns=list(VARIANT.source_files),
            local_dir=source,
        )
    verify_source(source)

    url_name = VARIANT.archive.rsplit("/", 1)[-1]
    archive = source / url_name
    if not archive.exists() and not offline:
        partial = archive.with_suffix(".partial")
        try:
            with urlopen(VARIANT.archive, timeout=120) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if sha256(partial) != VARIANT.archive_sha256:
                raise ValueError("Downloaded ONNX archive digest differs from the pinned digest")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    if not archive.is_file() or sha256(archive) != VARIANT.archive_sha256:
        raise ValueError("ONNX archive is missing or does not match its pinned digest")

    # Always read the reference graphs and fixtures from the verified archive.
    prefix = url_name.removesuffix(".tar.bz2")
    reference = source / prefix
    names = [f"{part}-{VARIANT.graph_suffix}" for part in PARTS] + ["tokens.txt"]
    with tarfile.open(archive) as tar:
        members = [tar.getmember(f"{prefix}/{name}") for name in names]
        members.extend(
            member
            for member in tar.getmembers()
            if member.name.startswith(f"{prefix}/test_wavs/") and member.isfile()
        )
        if not all(member.isfile() for member in members):
            raise ValueError("Expected handoff fixtures and plain ONNX graph files")
        tar.extractall(source, members=members, filter="data")
    return source, reference


@contextmanager
def source_package(source: Path) -> Iterator[str]:
    name = "_hugging_mac_zipformer_source"
    if name in sys.modules:
        raise RuntimeError("A Zipformer export is already loading upstream code")
    package = ModuleType(name)
    package.__path__ = [str(source)]
    sys.modules[name] = package
    try:
        yield name
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def downsample_fixed(module: Any, features: Tensor) -> Tensor:
    length, batch, channels = map(int, features.shape)
    stride = module.downsample
    if length % stride:
        raise ValueError("Static Zipformer export requires an evenly divisible downsampling size")
    weights = module.bias.softmax(0).reshape(stride, 1, 1)
    return cast(
        Tensor, (features.reshape(length // stride, stride, batch, channels) * weights).sum(1)
    )


class Encoder(nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: Tensor, *states: Tensor) -> tuple[Tensor, ...]:
        lengths = torch.full((features.shape[0],), features.shape[1], dtype=torch.int32)
        vector, _, updated = self.model.streaming_forward(features, lengths, states)
        return self.model.joiner.encoder_proj(vector), *updated


class Decoder(nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, tokens: Tensor) -> Tensor:
        return cast(
            Tensor,
            self.model.joiner.decoder_proj(self.model.decoder(tokens, need_pad=False).squeeze(1)),
        )


class Joiner(nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, encoder: Tensor, decoder: Tensor) -> Tensor:
        return cast(Tensor, self.model.joiner(encoder, decoder, project_input=False))


def reference_sessions(reference: Path) -> dict[str, Any]:
    ort = importlib.import_module("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return {
        part: ort.InferenceSession(
            str(reference / f"{part}-{VARIANT.graph_suffix}"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for part in PARTS
    }


def export(source: Path, reference: Path, run: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    verify_source(source)
    sessions = reference_sessions(reference)
    contract: dict[str, Any] = {}
    with source_package(source) as package:
        config_type = importlib.import_module(
            f"{package}.configuration_kws_zipformer"
        ).KwsZipformerConfig
        model_type = importlib.import_module(f"{package}.model").KwsZipformerModel
        downsample = importlib.import_module(f"{package}.zipformer").SimpleDownsample
        # MIL misinterprets the upstream zero-length pad on these fixed windows.
        downsample.forward = downsample_fixed
        config = config_type.from_pretrained(
            source,
            encoder_chunk_size=VARIANT.encoder_chunk_size,
            features_per_chunk=VARIANT.features_per_chunk,
            features_with_lookahead=VARIANT.features_with_lookahead,
        )
        model = model_type(config).eval()
        model.load_state_dict(load_file(source / "model.safetensors"), strict=True)
        for part, wrapper, session in zip(
            PARTS, (Encoder, Decoder, Joiner), sessions.values(), strict=True
        ):
            inputs = [
                {
                    "name": value.name,
                    "shape": [dim if isinstance(dim, int) else 1 for dim in value.shape],
                    "dtype": "int32" if value.type.startswith("tensor(int") else "float32",
                }
                for value in session.get_inputs()
            ]
            args = tuple(torch.zeros(i["shape"], dtype=getattr(torch, i["dtype"])) for i in inputs)
            names = tuple(value.name for value in session.get_outputs())
            print(f"Exporting {part} ({len(inputs)} inputs, {len(names)} outputs)", flush=True)
            coreml.export_mlprogram(
                wrapper(model).eval(),
                args,
                tuple(i["name"] for i in inputs),
                names,
                run / f"{part}.mlpackage",
                deployment_target="iOS17",
                check_trace=False,
            )
            contract[part] = {"inputs": inputs, "outputs": names}
    return contract
