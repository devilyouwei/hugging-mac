from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hugging_mac_sdk.capabilities import HandDetection
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.converters.registry import ConversionService
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.mediapipe_hand_detection import (
    MEDIAPIPE_HAND_DETECTION_MANIFEST,
    register_mediapipe_hand_detection,
)
from hugging_mac_sdk.models.mediapipe_hand_detection.config import (
    MEDIAPIPE_HAND_DETECTION_MODEL_ID,
    MediaPipeHandDetectionInstanceConfig,
)
from hugging_mac_sdk.models.mediapipe_hand_detection.converter import (
    MediaPipeHandDetectionConverter,
)
from hugging_mac_sdk.models.mediapipe_hand_detection.instance import MediaPipeHandDetectionInstance
from hugging_mac_sdk.models.mediapipe_hand_detection.resources import (
    MediaPipeHandDetectionResourceResolver,
)
from hugging_mac_sdk.models.mediapipe_hand_detection.utils.postprocess import (
    _anchors,
    decode_palms,
)
from hugging_mac_sdk.models.mediapipe_hand_detection.utils.types import PreparedImage
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.detection import ImageInput
from hugging_mac_sdk.schemas.hand import HandDetectionRequest
from hugging_mac_sdk.schemas.resources import ResolvedResource, UrlArchiveSource
from PIL import Image


class FakeEngine:
    runtime_name = "onnx"
    device = "CPUExecutionProvider"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.tensor: Any = None
        self.landmark_calls = 0

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact

    async def infer_detector(self, prepared: Any) -> dict[str, Any]:
        self.tensor = prepared.tensor
        coords = np.zeros((1, 2944, 18), dtype=np.float32)
        scores = np.full((1, 2944, 1), -100.0, dtype=np.float32)
        center_anchor = (16 * 32 + 16) * 2
        coords[0, center_anchor, :4] = (0.0, 0.0, 64.0, 64.0)
        scores[0, center_anchor, 0] = 10.0
        return {"box_coords": coords, "box_scores": scores}

    async def infer_landmarks(self, prepared: Any) -> dict[str, Any]:
        self.landmark_calls += 1
        landmarks = np.zeros((1, 21, 3), dtype=np.float32)
        landmarks[0, :, 0] = np.linspace(0.25, 0.75, 21)
        landmarks[0, :, 1] = np.linspace(0.75, 0.25, 21)
        return {
            "scores": np.asarray([0.95], dtype=np.float32),
            "lr": np.asarray([0.8], dtype=np.float32),
            "landmarks": landmarks,
        }

    async def close(self) -> None:
        return None


async def test_hand_detector_contract_and_letterbox(tmp_path: Path) -> None:
    image_path = tmp_path / "hand.png"
    Image.new("RGB", (640, 320), (255, 255, 255)).save(image_path)
    artifact = tmp_path / "hand_detector.onnx"
    artifact.touch()
    engine = FakeEngine(artifact)
    instance = MediaPipeHandDetectionInstance(
        MediaPipeHandDetectionInstanceConfig(runtime="onnx"), engine
    )

    await instance.load()
    capability = instance.require(HandDetection)
    response = await capability.detect_hands(
        HandDetectionRequest(image=ImageInput(path=image_path), confidence=0.9)
    )

    assert engine.tensor.shape == (1, 3, 256, 256)
    assert engine.tensor.dtype == np.float32
    assert len(response.hands) == 1
    assert response.hands[0].landmarks == ()
    assert engine.landmark_calls == 0
    landmark_response = await capability.detect_hands(
        HandDetectionRequest(
            image=ImageInput(path=image_path),
            confidence=0.9,
            include_landmarks=True,
        )
    )
    assert len(landmark_response.hands[0].landmarks) == 21
    assert landmark_response.hands[0].handedness == "right"
    assert landmark_response.hands[0].handedness_confidence == pytest.approx(0.8)
    assert landmark_response.hands[0].landmark_confidence == pytest.approx(0.95)
    assert landmark_response.hands[0].landmarks[0].z == 0.0
    assert engine.landmark_calls == 1
    mirrored_response = await capability.detect_hands(
        HandDetectionRequest(
            image=ImageInput(path=image_path),
            confidence=0.9,
            include_landmarks=True,
            input_mirrored=True,
        )
    )
    assert mirrored_response.hands[0].handedness == "left"
    assert mirrored_response.hands[0].handedness_confidence == pytest.approx(0.8)
    assert response.image_size.width == 640
    assert 0 <= response.hands[0].box.x1 < response.hands[0].box.x2 <= 640
    await instance.unload()


def test_manifest_factory_and_anchor_contract() -> None:
    assert MEDIAPIPE_HAND_DETECTION_MANIFEST.capabilities == frozenset({"hand-detection"})
    assert MEDIAPIPE_HAND_DETECTION_MANIFEST.revision == "013e27b599e37c3b4c69439de15de53cc5b5708e"
    assert MEDIAPIPE_HAND_DETECTION_MANIFEST.default_runtime == "coreml"
    assert MediaPipeHandDetectionInstanceConfig().runtime == "coreml"
    assert _anchors().shape == (2944, 2)
    definition = register_mediapipe_hand_detection(ModelRegistry(), ConverterRegistry())
    instance = definition.create(runtime="onnx", variant="float")
    assert isinstance(instance, MediaPipeHandDetectionInstance)
    assert instance.supports(HandDetection)
    assert set(definition.runtime_factories) == {"onnx", "coreml"}


def test_coreml_status_requires_both_internal_models(tmp_path: Path) -> None:
    resolver = MediaPipeHandDetectionResourceResolver(
        UrlArchiveSource(url="https://models.example/mediapipe.zip"),
        MediaPipeHandDetectionInstanceConfig(runtime="coreml", model_home=tmp_path),
    )
    detector = resolver.coreml_path / "hand_detector.mlpackage" / "Manifest.json"
    landmarker = resolver.coreml_path / "hand_landmark_detector.mlpackage" / "Manifest.json"

    detector.parent.mkdir(parents=True)
    detector.write_text("{}")
    assert resolver.status().artifacts[1].available is False

    landmarker.parent.mkdir(parents=True)
    landmarker.write_text("{}")
    assert resolver.status().artifacts[1].available is True


def test_weighted_nms_merges_close_palm_candidates_at_high_ui_iou() -> None:
    coords = np.zeros((1, 2944, 18), dtype=np.float32)
    scores = np.full((1, 2944, 1), -100.0, dtype=np.float32)
    center_anchor = (16 * 32 + 16) * 2
    coords[0, center_anchor, :4] = (0.0, 0.0, 100.0, 100.0)
    coords[0, center_anchor + 1, :4] = (25.0, 0.0, 100.0, 100.0)
    scores[0, center_anchor, 0] = 10.0
    scores[0, center_anchor + 1, 0] = 9.0
    prepared = PreparedImage(
        tensor=np.empty((1, 3, 256, 256), dtype=np.float32),
        pixels=np.zeros((256, 256, 3), dtype=np.uint8),
        original_width=256,
        original_height=256,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        preprocess_ms=0.0,
    )

    palms, _ = decode_palms(
        {"box_coords": coords, "box_scores": scores},
        HandDetectionRequest(
            image=ImageInput(data=b"unused"),
            confidence=0.5,
            iou_threshold=0.7,
        ),
        prepared,
    )

    assert len(palms) == 1
    assert 92.0 < palms[0].box[0] < 105.0
    assert palms[0].confidence > 0.99


async def test_coreml_converter_is_registered_and_commits_package(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source_dir = tmp_path / "onnx"
    source_dir.mkdir()
    (source_dir / "hand_detector.onnx").write_bytes(b"onnx")
    (source_dir / "hand_landmark_detector.onnx").write_bytes(b"onnx")
    source = ResolvedResource(
        path=source_dir,
        source=UrlArchiveSource(url="https://models.example/mediapipe.zip"),
        digest="1" * 64,
        size_bytes=4,
    )

    def fake_convert(
        _: MediaPipeHandDetectionConverter,
        source_path: Path,
        output: Path,
        options: Any,
    ) -> None:
        assert source_path == source_dir
        assert options.precision == "float32"
        for name in ("hand_detector.mlpackage", "hand_landmark_detector.mlpackage"):
            package = output / name
            package.mkdir(parents=True)
            (package / "Manifest.json").write_text("{}")

    monkeypatch.setattr(MediaPipeHandDetectionConverter, "_convert", fake_convert)
    registry = ConverterRegistry()
    definition = register_mediapipe_hand_detection(ModelRegistry(), registry)
    output = tmp_path / "coreml"
    result = await ConversionService(registry).convert(
        ConversionRequest(
            model_id=MEDIAPIPE_HAND_DETECTION_MODEL_ID,
            model_revision=definition.manifest.revision,
            variant="float",
            source=source,
            source_format=ArtifactFormat.ONNX,
            target_format=ArtifactFormat.COREML,
            output_path=output,
        ),
        definition=definition,
    )

    assert result.converter_id == "qualcomm.mediapipe-hand-detection"
    assert result.options["precision"] == "float32"
    assert (result.path / "hand_detector.mlpackage" / "Manifest.json").read_text() == "{}"
    assert (result.path / "hand_landmark_detector.mlpackage" / "Manifest.json").read_text() == "{}"
