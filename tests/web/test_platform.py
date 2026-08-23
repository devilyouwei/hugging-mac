from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from hugging_mac_sdk import (
    ArtifactFormat,
    BaseModelInstance,
    ConversionRequest,
    ConversionResult,
    ModelDefinition,
)
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.models.yolov8.converter import YoloV8Converter
from hugging_mac_sdk.models.yolov8.instance import YoloV8Instance
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    Detection,
    DetectionResponse,
    DetectionTimings,
    FaceDetection,
    FaceDetectionResponse,
    FaceLandmarks5,
    ImageSize,
    Point2D,
)
from hugging_mac_sdk.schemas.hand import HandDetectionResponse, HandResult
from hugging_mac_sdk.schemas.pose import Keypoint, Pose, PoseEstimationResponse
from hugging_mac_sdk.schemas.resources import ResolvedResource
from hugging_mac_sdk.schemas.segmentation import (
    PolygonPoint,
    Segmentation,
    SegmentationResponse,
)
from hugging_mac_web.chat.schemas import ChatReplyView, ChatStreamEventView
from hugging_mac_web.config import WebSettings
from hugging_mac_web.live_transcription.config import (
    AUDIO8_PROFILE,
    NEMOTRON_3_5_ASR_PROFILE,
    QWEN3_ASR_PROFILE,
    SENSEVOICE_PROFILE,
    LiveTranscriptionSettings,
)
from hugging_mac_web.live_transcription.manifest import LIVE_TRANSCRIPTION_MANIFEST
from hugging_mac_web.live_transcription.schemas import (
    ArtifactResourceView,
    LoadedModelView,
    ResourceStatusView,
    RuntimeResourceView,
)
from hugging_mac_web.live_transcription.service import LiveTranscriptionService
from hugging_mac_web.main import create_app
from hugging_mac_web.text_to_speech.audio import float32le_to_wav
from hugging_mac_web.text_to_speech.config import (
    AUDIO8_TTS_PROFILE,
    KOKORO_82M_PROFILE,
    MOSS_TTS_NANO_PROFILE,
    QWEN3_TTS_0_6B_BASE_4BIT_PROFILE,
)
from hugging_mac_web.text_to_speech.manifest import TEXT_TO_SPEECH_MANIFEST
from PIL import Image


def test_live_transcription_stability_override_does_not_change_audio8(tmp_path: Path) -> None:
    context = SimpleNamespace(settings=SimpleNamespace(model_home=tmp_path))
    service = LiveTranscriptionService(  # type: ignore[arg-type]
        context,
        LiveTranscriptionSettings(coreml_compute_units="cpu-only"),
    )

    assert service._model_options_for(AUDIO8_PROFILE.model_id) == {"model_home": tmp_path}
    assert service._model_options_for(SENSEVOICE_PROFILE.model_id)["compute_units"] == "cpu-only"
    assert service._model_options_for(QWEN3_ASR_PROFILE.model_id)["device"] == "cpu-only"


class PlatformDummyInstance(BaseModelInstance):
    async def _load(self) -> None:
        return None

    async def _unload(self) -> None:
        return None


def _settings(tmp_path: Path) -> WebSettings:
    return WebSettings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        model_home=tmp_path / "models",
        database_path=tmp_path / "data" / "platform.json",
        runtime_preference="coreml,pytorch-mps",
    )


def test_default_runtime_preference_is_env_configurable(monkeypatch: Any) -> None:
    monkeypatch.setenv("WEB_RUNTIME_PREFERENCE", "onnx,mps,mlx,coreml")

    settings = WebSettings(_env_file=None)

    assert settings.runtime_preferences == ("onnx", "mps", "mlx", "coreml")


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


def test_platform_catalog_system_and_cors(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        root = client.get("/")
        health = client.get("/api/v1/system/health")
        info = client.get("/api/v1/system/info")
        models = client.get("/api/v1/catalog/models")
        deepfilternet_resources = client.get(
            "/api/v1/catalog/models/deepfilternet/deepfilternet3/resources"
        )
        qwen3_asr_resources = client.get("/api/v1/catalog/models/qwen/qwen3-asr/resources")
        qwen3_asr_inventory = client.get("/api/v1/catalog/models/qwen/qwen3-asr/inventory")
        apps = client.get("/api/v1/catalog/apps")
        games = client.get("/api/v1/catalog/games")
        palm_trace_status = client.get("/api/v1/games/palm-trace/status")
        asr_models = client.get("/api/v1/apps/live-transcription/models")
        tts_models = client.get("/api/v1/apps/text-to-speech/models")
        chat_models = client.get("/api/v1/apps/chat/models")
        resources = client.get("/api/v1/apps/object-detection/resources")
        openapi = client.get("/openapi.json")
        preflight = client.options(
            "/api/v1/catalog/models",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert root.status_code == 200
    assert health.json()["data"]["status"] == "healthy"
    assert "memory_total_bytes" in info.json()["data"]
    assert {
        "chip_name",
        "cpu_performance_cores",
        "cpu_efficiency_cores",
        "gpu_cores",
        "neural_engine_cores",
    } <= info.json()["data"].keys()
    app_summaries = {item["manifest"]["app_id"]: item for item in apps.json()["data"]}
    assert set(app_summaries) == {
        "chat",
        "instance-segmentation",
        "live-transcription",
        "object-detection",
        "pose-estimation",
        "text-to-speech",
    }
    assert all(item["status"] == "available" for item in app_summaries.values())
    assert {item["model_id"] for item in asr_models.json()["data"]} == {
        AUDIO8_PROFILE.model_id,
        NEMOTRON_3_5_ASR_PROFILE.model_id,
        SENSEVOICE_PROFILE.model_id,
        "qwen/qwen3-asr",
    }
    assert all(item["ready_instance_id"] is None for item in asr_models.json()["data"])
    asr_model_views = {item["model_id"]: item for item in asr_models.json()["data"]}
    assert asr_model_views[AUDIO8_PROFILE.model_id]["runtime"] == "coreml"
    assert asr_model_views[SENSEVOICE_PROFILE.model_id]["runtime"] == "coreml"
    assert asr_model_views[NEMOTRON_3_5_ASR_PROFILE.model_id]["streaming"]
    assert asr_model_views[NEMOTRON_3_5_ASR_PROFILE.model_id]["streaming_chunk_seconds"] == 2.24
    assert {item["model_id"] for item in tts_models.json()["data"]} == {
        "audio8/audio8-tts-preview",
        "hexgrad/kokoro",
        "openmoss-team/moss-tts-nano-100m",
        "qwen/qwen3-tts-12hz",
    }
    moss_tts_view = next(
        item
        for item in tts_models.json()["data"]
        if item["model_id"] == MOSS_TTS_NANO_PROFILE.model_id
    )
    assert moss_tts_view["supports_reference_audio"]
    assert not moss_tts_view["requires_reference_audio"]
    assert not moss_tts_view["requires_reference_text"]
    qwen3_tts_view = next(
        item for item in tts_models.json()["data"] if item["model_id"] == "qwen/qwen3-tts-12hz"
    )
    assert qwen3_tts_view["display_name"] == "Qwen3-TTS"
    assert qwen3_tts_view["supports_reference_audio"]
    assert not qwen3_tts_view["requires_reference_audio"]
    assert qwen3_tts_view["requires_reference_text"]
    assert {item["runtime"] for item in qwen3_tts_view["resource"]["runtimes"]} == {
        "mlx",
        "coreml",
    }
    chat_model_views = {item["model_id"]: item for item in chat_models.json()["data"]}
    assert set(chat_model_views) == {"qwen/qwen3.5", "google/gemma-4"}
    assert {item["variant"] for item in chat_models.json()["data"]} == {
        "9b",
        "4b",
        "2b",
        "e4b",
        "e2b",
    }
    assert {item["profile_id"] for item in chat_models.json()["data"]} == {
        "9b",
        "4b",
        "2b",
        "gemma-4-e4b",
        "gemma-4-e2b",
    }
    assert all(item["supports_images"] for item in chat_models.json()["data"])
    assert games.status_code == 200
    assert palm_trace_status.json()["data"] == {
        "status": "ready",
        "control": "mediapipe-palm-detection",
        "landmarks": "disabled",
    }
    game_summaries = {item["manifest"]["app_id"]: item for item in games.json()["data"]}
    assert set(game_summaries) == {
        "palm-thunder",
        "palm-trace",
        "yolo-fruit-slice",
        "yolo-pose-follow",
    }
    assert game_summaries["yolo-pose-follow"]["manifest"]["category"] == "game"
    assert game_summaries["yolo-pose-follow"]["status"] == "available"
    fruit_slice = game_summaries["yolo-fruit-slice"]
    assert fruit_slice["manifest"]["category"] == "game"
    palm_thunder = game_summaries["palm-thunder"]
    assert palm_thunder["manifest"]["frontend_route"] == "/games/palm-thunder"
    assert palm_thunder["status"] == "available"
    palm_trace = game_summaries["palm-trace"]
    assert palm_trace["manifest"]["frontend_route"] == "/games/palm-trace"
    assert palm_trace["manifest"]["required_models"][0]["capabilities"] == ["hand-detection"]
    assert palm_trace["status"] == "available"
    assert fruit_slice["status"] == "available"
    assert fruit_slice["manifest"]["required_models"] == [
        {
            "model_id": "ultralytics/yolov8-pose",
            "capabilities": ["pose-estimation"],
            "preferred_runtime": None,
            "required": True,
        }
    ]
    assert not any(item["available"] for item in resources.json()["data"]["artifacts"])
    assert "/api/v1/apps/object-detection/detect" in openapi.json()["paths"]
    assert "/api/v1/apps/pose-estimation/estimate" in openapi.json()["paths"]
    assert "/api/v1/apps/instance-segmentation/segment" in openapi.json()["paths"]
    assert "/api/v1/apps/live-transcription/models/load" in openapi.json()["paths"]
    assert "/api/v1/apps/text-to-speech/synthesize" in openapi.json()["paths"]
    assert "/api/v1/apps/text-to-speech/synthesize/reference" in openapi.json()["paths"]
    assert "/api/v1/apps/text-to-speech/models/load" in openapi.json()["paths"]
    assert "/api/v1/apps/chat/messages" in openapi.json()["paths"]
    assert "/api/v1/apps/chat/messages/stream" in openapi.json()["paths"]
    assert "/api/v1/apps/chat/model/load" in openapi.json()["paths"]
    assert "/api/v1/apps/chat/resources/source/download" not in openapi.json()["paths"]
    assert "/api/v1/apps/chat/resources" not in openapi.json()["paths"]
    assert "/api/v1/apps/object-detection/resources/source/download" not in openapi.json()["paths"]
    assert "/api/v1/apps/object-detection/resources/coreml/convert" not in openapi.json()["paths"]
    assert "/api/v1/apps/pose-estimation/resources/source/download" not in openapi.json()["paths"]
    assert "/api/v1/apps/pose-estimation/resources/coreml/convert" not in openapi.json()["paths"]
    assert (
        "/api/v1/apps/instance-segmentation/resources/source/download"
        not in openapi.json()["paths"]
    )
    assert (
        "/api/v1/apps/instance-segmentation/resources/coreml/convert" not in openapi.json()["paths"]
    )
    assert "/api/v1/games/yolo-pose-follow/templates" in openapi.json()["paths"]
    assert "/api/v1/games/yolo-pose-follow/match" in openapi.json()["paths"]
    assert "/api/v1/games/yolo-fruit-slice/status" in openapi.json()["paths"]
    assert "/api/v1/games/palm-thunder/status" in openapi.json()["paths"]
    model = next(item for item in models.json()["data"] if item["model_id"] == "ultralytics/yolov8")
    silero = next(
        item for item in models.json()["data"] if item["model_id"] == "snakers4/silero-vad"
    )
    deepfilternet = next(
        item for item in models.json()["data"] if item["model_id"] == "deepfilternet/deepfilternet3"
    )
    gemma4 = next(item for item in models.json()["data"] if item["model_id"] == "google/gemma-4")
    qwen3_asr = next(item for item in models.json()["data"] if item["model_id"] == "qwen/qwen3-asr")
    assert {item["model_id"] for item in models.json()["data"]} >= {
        "audio8/audio8-asr",
        "audio8/audio8-tts-preview",
        "hexgrad/kokoro",
        "openmoss-team/moss-tts-nano-100m",
        "deepfilternet/deepfilternet3",
        "funaudiollm/sensevoice",
        "py-feat/retinaface",
        "google/gemma-4",
        "qwen/qwen3.5",
        "qwen/qwen3-asr",
        NEMOTRON_3_5_ASR_PROFILE.model_id,
        "snakers4/silero-vad",
        "ultralytics/yolov8",
        "ultralytics/yolov8-pose",
        "ultralytics/yolov8-seg",
    }
    assert model["model_id"] == "ultralytics/yolov8"
    assert [variant["name"] for variant in gemma4["variants"]] == ["e4b", "e2b"]
    assert gemma4["default_variant"] == "e4b"
    assert {runtime["name"] for runtime in gemma4["runtimes"]} == {"mlx"}
    assert gemma4["capabilities"] == ["chat"]
    assert [variant["name"] for variant in model["variants"]] == ["n", "s", "m"]
    assert model["variants"][0]["default"]
    assert model["default_variant"] == "n"
    assert model["instance_count"] == 0
    assert model["instances"] == []
    assert {runtime["name"] for runtime in model["runtimes"]} == {
        "coreml",
        "onnx",
        "pytorch-mps",
    }
    assert silero["default_runtime"] == "coreml"
    assert silero["license"] == "MIT"
    assert silero["source_url"].startswith("https://")
    assert {runtime["name"] for runtime in silero["runtimes"]} == {"coreml", "onnx"}
    assert deepfilternet["default_runtime"] == "coreml"
    assert [variant["name"] for variant in deepfilternet["variants"]] == ["default"]
    assert {runtime["name"] for runtime in deepfilternet["runtimes"]} == {"coreml"}
    assert qwen3_asr["default_runtime"] == "coreml"
    assert [variant["name"] for variant in qwen3_asr["variants"]] == ["0.6b"]
    assert {runtime["name"] for runtime in qwen3_asr["runtimes"]} == {"coreml"}
    assert deepfilternet_resources.status_code == 200
    assert deepfilternet_resources.json()["data"]["artifacts"][0] == {
        "artifact_id": "coreml-int8",
        "format": "coreml",
        "runtime": "coreml",
        "available": False,
        "size_bytes": None,
        "provisioning": "download",
    }
    assert qwen3_asr_resources.status_code == 200
    assert [item["artifact_id"] for item in qwen3_asr_resources.json()["data"]["artifacts"]] == [
        "coreml-int8",
        "tokenizer",
    ]
    assert qwen3_asr_inventory.status_code == 200
    inventory_artifacts = qwen3_asr_inventory.json()["data"]["artifacts"]
    assert [item["artifact_id"] for item in inventory_artifacts] == [
        "coreml-int8",
        "tokenizer",
    ]
    assert inventory_artifacts[0]["source"]["kind"] == "huggingface"
    assert not inventory_artifacts[0]["shared"]
    assert inventory_artifacts[0]["required_shares"] == ["tokenizer"]
    assert inventory_artifacts[1]["source"]["kind"] == "huggingface"
    assert inventory_artifacts[1]["shared"]
    assert inventory_artifacts[1]["variant"] is None
    assert inventory_artifacts[1]["runtime"] is None
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_chat_accepts_text_and_image(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def fake_chat(*_: object, **kwargs: object) -> ChatReplyView:
        image_paths = kwargs.get("image_paths")
        assert image_paths is None  # Positional in the service route.
        return ChatReplyView(
            content="图中是一块深色背景。",
            model_id="mlx-community/qwen3.5-9b-mlx-4bit",
            instance_id="chat-instance",
            runtime="mlx",
            device="gpu",
            prompt_tokens=12,
            generated_tokens=8,
            inference_ms=18.5,
        )

    monkeypatch.setattr("hugging_mac_web.chat.service.ChatService.chat", fake_chat)
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/chat/messages",
            data={
                "instance_id": "chat-instance",
                "prompt": "描述这张图片",
                "history_json": '[{"role":"assistant","content":"请上传图片"}]',
                "max_tokens": "128",
            },
            files={"images": ("sample.png", _png(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["data"]["content"] == "图中是一块深色背景。"
    assert not list((tmp_path / "cache" / "chat-uploads").iterdir())


def test_chat_streams_deltas_and_cleans_up_images(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def fake_stream(*_: object, **__: object) -> AsyncIterator[ChatStreamEventView]:
        for delta, tokens, finish_reason in (("你", 1, None), ("好", 2, None), ("", 2, "stop")):
            yield ChatStreamEventView(
                delta=delta,
                finish_reason=finish_reason,
                model_id="mlx-community/qwen3.5-9b-mlx-4bit",
                instance_id="chat-instance",
                runtime="mlx",
                device="gpu",
                generated_tokens=tokens,
                inference_ms=12.0 if finish_reason else None,
            )

    monkeypatch.setattr(
        "hugging_mac_web.chat.service.ChatService.stream_chat",
        fake_stream,
    )
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/chat/messages/stream",
            data={"instance_id": "chat-instance", "prompt": "你好"},
            files={"images": ("sample.png", _png(), "image/png")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("event: delta") == 2
    assert "event: done" in response.text
    assert '"delta":"你"' in response.text
    assert not list((tmp_path / "cache" / "chat-uploads").iterdir())


def test_yolo_pose_follow_templates_and_matching(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        templates = client.get("/api/v1/games/yolo-pose-follow/templates")
        first = templates.json()["data"][0]
        matched = client.post(
            "/api/v1/games/yolo-pose-follow/match",
            json={
                "template_id": first["template_id"],
                "keypoints": [
                    {
                        "x": point["x"] * 1000,
                        "y": point["y"] * 1000,
                        "confidence": 0.95,
                    }
                    for point in first["points"]
                ],
                "allow_mirror": True,
            },
        )

    assert templates.status_code == 200
    assert len(templates.json()["data"]) >= 30
    assert matched.status_code == 200
    assert matched.json()["data"]["matched"]
    assert matched.json()["data"]["score"] > 0.99


def test_live_transcription_load_forwards_selected_runtime(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def fake_load_model(
        self: object,
        model_id: str,
        *,
        variant: str | None = None,
        runtime: str | None = None,
    ) -> LoadedModelView:
        del self
        assert model_id == AUDIO8_PROFILE.model_id
        assert variant == "0.1b"
        assert runtime == "pytorch-mps"
        return LoadedModelView(
            instance_id="audio8-mps",
            model_id=model_id,
            variant="0.1b",
            runtime=runtime,
            device="mps",
            state="ready",
        )

    monkeypatch.setattr(
        "hugging_mac_web.live_transcription.service.LiveTranscriptionService.load_model",
        fake_load_model,
    )
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/live-transcription/models/load",
            json={
                "model_id": AUDIO8_PROFILE.model_id,
                "variant": "0.1b",
                "runtime": "pytorch-mps",
            },
        )

    assert response.status_code == 200
    assert response.json()["data"]["runtime"] == "pytorch-mps"


def test_live_transcription_vad_websocket_validates_stream_instance(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    with (
        TestClient(app) as client,
        client.websocket_connect("/api/v1/apps/live-transcription/live/ws") as socket,
    ):
        socket.send_json(
            {
                "type": "start",
                "model_id": AUDIO8_PROFILE.model_id,
                "instance_id": "missing-asr-instance",
                "sample_rate": 16000,
                "vad_threshold": 0.5,
            }
        )
        response = socket.receive_json()

    assert response["type"] == "error"


def test_live_transcription_reports_optional_pipeline_components(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/v1/apps/live-transcription/pipeline/components")

    assert response.status_code == 200
    components = {item["component_id"]: item for item in response.json()["data"]}
    assert set(components) == {"vad", "enhancement"}
    assert components["vad"]["model_id"] == "snakers4/silero-vad"
    assert components["vad"]["state"] == "not-downloaded"
    assert not components["vad"]["downloaded"]
    assert components["enhancement"]["state"] == "not-downloaded"
    assert components["enhancement"]["loaded_model"] is None


def test_models_catalog_exposes_all_nemotron_download_variants(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        inventory = client.get(
            "/api/v1/catalog/models/nvidia/nemotron-3.5-asr-streaming-0.6b/inventory"
        )

    assert inventory.status_code == 200
    artifacts = inventory.json()["data"]["artifacts"]
    assert len(artifacts) == 8
    assert {item["variant"] for item in artifacts} == {
        f"{script}-{chunk_ms}ms"
        for script in ("latin", "multilingual")
        for chunk_ms in (560, 1120, 2240, 4480)
    }
    assert all(item["runtime"] == "coreml" for item in artifacts)
    assert all(item["source"]["kind"] == "huggingface" for item in artifacts)


def test_models_catalog_downloads_one_shared_artifact_independently(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def fake_download(
        _: ResourceDownloader,
        source: Any,
        destination: Path,
        **__: object,
    ) -> ResolvedResource:
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("vocab.json", "merges.txt", "tokenizer_config.json"):
            (destination / filename).write_text("{}")
        return ResolvedResource(
            path=destination,
            source=source,
            digest="0" * 64,
            size_bytes=2,
        )

    monkeypatch.setattr(ResourceDownloader, "download", fake_download)
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/catalog/models/qwen/qwen3-asr/resources/download-one",
            json={
                "variant": "0.6b",
                "runtime": "coreml",
                "artifact_id": "tokenizer",
            },
        )

    assert response.status_code == 200
    artifacts = {item["artifact_id"]: item for item in response.json()["data"]["artifacts"]}
    assert artifacts["tokenizer"]["available"]
    assert not artifacts["coreml-int8"]["available"]


def test_chat_streams_nemotron_over_websocket(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class FakeSession:
        session_id = "chat-session"
        closed = False

        def __init__(self, emit: Any) -> None:
            self.emit = emit

        async def enqueue(self, audio: bytes) -> None:
            assert audio == b"\0\0" * 160
            await self.emit({"type": "speech_start", "utterance_id": 1})
            await self.emit(
                {"type": "partial", "utterance_id": 1, "text": "hello", "delta": "hello"}
            )

        async def stop(self) -> None:
            self.closed = True
            await self.emit(
                {
                    "type": "transcript",
                    "utterance_id": 1,
                    "text": "hello world",
                    "vad_inference_ms": 2.0,
                    "preprocess_ms": 1.0,
                    "inference_ms": 12.0,
                }
            )

        async def cancel(self) -> None:
            self.closed = True

    class FakeSessionFactory:
        @staticmethod
        async def create(
            _: object,
            __: object,
            start: Any,
            emit: Any,
            *,
            include_input_audio: bool,
        ) -> FakeSession:
            assert start.model_id == NEMOTRON_3_5_ASR_PROFILE.model_id
            assert start.instance_id == "nemotron-instance"
            assert not start.use_vad
            assert not start.use_enhancement
            assert not include_input_audio
            return FakeSession(emit)

    monkeypatch.setattr(
        "hugging_mac_web.chat.routes.LiveTranscriptionSession",
        FakeSessionFactory,
    )
    app = create_app(_settings(tmp_path))

    with (
        TestClient(app) as client,
        client.websocket_connect("/api/v1/apps/chat/asr/ws") as socket,
    ):
        socket.send_json(
            {
                "type": "start",
                "instance_id": "nemotron-instance",
                "sample_rate": 16000,
                "streaming_chunk_seconds": 2.24,
            }
        )
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(b"\0\0" * 160)
        assert socket.receive_json()["type"] == "speech_start"
        assert socket.receive_json()["type"] == "partial"
        socket.send_json({"type": "stop"})
        final = socket.receive_json()
        stopped = socket.receive_json()

    assert final["type"] == "transcript"
    assert final["text"] == "hello world"
    assert stopped["type"] == "stopped"


def test_live_transcription_declares_inference_ready_asr_models() -> None:
    requirements = {
        requirement.model_id: requirement
        for requirement in LIVE_TRANSCRIPTION_MANIFEST.required_models
    }

    assert AUDIO8_PROFILE.runtime == "coreml"
    assert SENSEVOICE_PROFILE.runtime == "coreml"
    assert requirements[AUDIO8_PROFILE.model_id].preferred_runtime == "coreml"
    assert requirements[SENSEVOICE_PROFILE.model_id].preferred_runtime == "coreml"
    assert requirements["qwen/qwen3-asr"].preferred_runtime == "coreml"
    assert requirements[NEMOTRON_3_5_ASR_PROFILE.model_id].preferred_runtime == "coreml"
    assert NEMOTRON_3_5_ASR_PROFILE.streaming
    assert requirements["snakers4/silero-vad"].preferred_runtime == "coreml"
    assert requirements["snakers4/silero-vad"].capabilities == ("voice-activity-detection",)


def test_live_transcription_prefers_available_coreml_artifact() -> None:
    resource = ResourceStatusView(
        model_id=SENSEVOICE_PROFILE.model_id,
        revision="revision",
        variant="small",
        artifacts=(
            ArtifactResourceView(
                artifact_id="source",
                format="pytorch",
                runtime="pytorch-mps",
                available=True,
                size_bytes=1,
            ),
            ArtifactResourceView(
                artifact_id="coreml",
                format="coreml",
                runtime="coreml",
                available=True,
                size_bytes=1,
            ),
        ),
        runtimes=(
            RuntimeResourceView(
                runtime="pytorch-mps",
                available=True,
                size_bytes=1,
                artifact_ids=("source",),
            ),
            RuntimeResourceView(
                runtime="coreml",
                available=True,
                size_bytes=1,
                artifact_ids=("coreml",),
            ),
        ),
    )

    selected = LiveTranscriptionService._select_runtime(SENSEVOICE_PROFILE, resource)

    assert selected.runtime == "coreml"
    assert selected.required_artifact_id == "coreml"


def test_text_to_speech_returns_playable_wave(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def fake_synthesize(*_: object, **__: object) -> tuple[bytes, dict[str, str]]:
        return (
            float32le_to_wav(np.zeros(2400, dtype=np.float32).tobytes(), 24000),
            {
                "x-model-id": "hexgrad/kokoro",
                "x-runtime": "pytorch-mps",
                "x-device": "mps",
                "x-duration-seconds": "0.1",
                "x-inference-ms": "12.0",
            },
        )

    monkeypatch.setattr(
        "hugging_mac_web.text_to_speech.service.TextToSpeechService.synthesize",
        fake_synthesize,
    )
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/text-to-speech/synthesize",
            json={
                "model_id": "hexgrad/kokoro",
                "instance_id": "test-instance",
                "text": "Hello from Kokoro.",
                "voice": "af_heart",
                "language": "a",
                "speed": 1.0,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-runtime"] == "pytorch-mps"
    assert response.content.startswith(b"RIFF")


def test_text_to_speech_returns_actionable_inference_reason(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def fail_synthesis(*_: object, **__: object) -> tuple[bytes, dict[str, str]]:
        raise InferenceError(
            "Audio8-TTS MLX BF16 synthesis failed",
            details={"reason": "reference audio duration must be between 0.5 and 30 seconds"},
        )

    monkeypatch.setattr(
        "hugging_mac_web.text_to_speech.service.TextToSpeechService.synthesize",
        fail_synthesis,
    )
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/text-to-speech/synthesize",
            json={
                "model_id": AUDIO8_TTS_PROFILE.model_id,
                "instance_id": "mlx-instance",
                "text": "你好。",
                "voice": "speaker_a",
                "speed": 1.0,
            },
        )

    assert response.status_code == 500
    assert response.json()["error"]["details"]["reason"] == (
        "reference audio duration must be between 0.5 and 30 seconds"
    )


def test_kokoro_tts_uses_downloaded_coreml_artifact() -> None:
    assert KOKORO_82M_PROFILE.runtime == "coreml"
    assert KOKORO_82M_PROFILE.variant == "v1.0"
    assert KOKORO_82M_PROFILE.required_artifact_id == "coreml"


def test_audio8_tts_uses_one_canonical_model_requirement() -> None:
    requirements = {
        requirement.model_id: requirement for requirement in TEXT_TO_SPEECH_MANIFEST.required_models
    }

    assert AUDIO8_TTS_PROFILE.runtime == "pytorch"
    assert AUDIO8_TTS_PROFILE.variant == "0.1b-preview"
    assert requirements[AUDIO8_TTS_PROFILE.model_id].preferred_runtime == "pytorch"
    assert QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.runtime == "mlx"
    assert QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.variant == "0.6b-base"
    assert QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.supports_reference_audio
    assert not QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.requires_reference_audio
    assert QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.requires_reference_text
    assert QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.display_name == "Qwen3-TTS"
    assert QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.short_name == "Qwen3-TTS"
    assert requirements[QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.model_id].preferred_runtime == "mlx"
    assert MOSS_TTS_NANO_PROFILE.runtime == "mlx"
    assert MOSS_TTS_NANO_PROFILE.variant == "nano-100m"
    assert MOSS_TTS_NANO_PROFILE.supports_reference_audio
    assert not MOSS_TTS_NANO_PROFILE.requires_reference_audio
    assert not MOSS_TTS_NANO_PROFILE.requires_reference_text
    assert requirements[MOSS_TTS_NANO_PROFILE.model_id].preferred_runtime == "mlx"


def test_moss_tts_accepts_reference_audio_without_transcript(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, object] = {}

    async def fake_synthesize(
        _: object,
        request: object,
        *,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        captured["request"] = request
        captured["reference_audio"] = reference_audio
        captured["reference_text"] = reference_text
        return (
            float32le_to_wav(np.zeros(4800, dtype=np.float32).tobytes(), 48000),
            {
                "x-model-id": MOSS_TTS_NANO_PROFILE.model_id,
                "x-runtime": "mlx",
                "x-device": "gpu",
                "x-duration-seconds": "0.1",
                "x-inference-ms": "20.0",
            },
        )

    monkeypatch.setattr(
        "hugging_mac_web.text_to_speech.service.TextToSpeechService.synthesize",
        fake_synthesize,
    )
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/text-to-speech/synthesize/reference",
            files={"file": ("reference.wav", b"RIFF-moss-reference", "audio/wav")},
            data={
                "model_id": MOSS_TTS_NANO_PROFILE.model_id,
                "instance_id": "moss-instance",
                "text": "Hello in the cloned voice.",
                "speed": "1.0",
            },
        )

    assert response.status_code == 200
    assert captured["reference_audio"] == b"RIFF-moss-reference"
    assert captured["reference_text"] is None


def test_moss_tts_accepts_synthesis_without_reference(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, object] = {}

    async def fake_synthesize(
        _: object,
        request: object,
        *,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        captured["request"] = request
        captured["reference_audio"] = reference_audio
        captured["reference_text"] = reference_text
        return (
            float32le_to_wav(np.zeros(4800, dtype=np.float32).tobytes(), 48000),
            {
                "x-model-id": MOSS_TTS_NANO_PROFILE.model_id,
                "x-runtime": "mlx",
                "x-device": "gpu",
                "x-duration-seconds": "0.1",
                "x-inference-ms": "20.0",
            },
        )

    monkeypatch.setattr(
        "hugging_mac_web.text_to_speech.service.TextToSpeechService.synthesize",
        fake_synthesize,
    )
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/text-to-speech/synthesize",
            json={
                "model_id": MOSS_TTS_NANO_PROFILE.model_id,
                "instance_id": "moss-instance",
                "text": "Hello without a reference voice.",
                "speed": 1.0,
            },
        )

    assert response.status_code == 200
    assert captured["reference_audio"] is None
    assert captured["reference_text"] is None


def test_text_to_speech_accepts_reference_voice_upload(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, object] = {}

    async def fake_synthesize(
        _: object,
        request: object,
        *,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        captured["request"] = request
        captured["reference_audio"] = reference_audio
        captured["reference_text"] = reference_text
        return (
            float32le_to_wav(np.zeros(4410, dtype=np.float32).tobytes(), 44100),
            {
                "x-model-id": AUDIO8_TTS_PROFILE.model_id,
                "x-runtime": "mlx",
                "x-device": "gpu",
                "x-duration-seconds": "0.1",
                "x-inference-ms": "20.0",
            },
        )

    monkeypatch.setattr(
        "hugging_mac_web.text_to_speech.service.TextToSpeechService.synthesize",
        fake_synthesize,
    )
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/text-to-speech/synthesize/reference",
            files={"file": ("reference.wav", b"RIFF-reference", "audio/wav")},
            data={
                "model_id": AUDIO8_TTS_PROFILE.model_id,
                "instance_id": "mlx-instance",
                "text": "你好。",
                "voice": "speaker_a",
                "reference_text": "参考录音原文。",
                "speed": "1.0",
            },
        )

    assert response.status_code == 200
    assert response.headers["x-runtime"] == "mlx"
    assert captured["reference_audio"] == b"RIFF-reference"
    assert captured["reference_text"] == "参考录音原文。"


def test_qwen3_tts_accepts_direct_reference_and_language(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, object] = {}

    async def fake_synthesize(
        _: object,
        request: object,
        *,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        captured["request"] = request
        captured["reference_audio"] = reference_audio
        captured["reference_text"] = reference_text
        return (
            float32le_to_wav(np.zeros(2400, dtype=np.float32).tobytes(), 24000),
            {
                "x-model-id": QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.model_id,
                "x-runtime": "mlx",
                "x-device": "gpu",
                "x-duration-seconds": "0.1",
                "x-inference-ms": "20.0",
            },
        )

    monkeypatch.setattr(
        "hugging_mac_web.text_to_speech.service.TextToSpeechService.synthesize",
        fake_synthesize,
    )
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/text-to-speech/synthesize/reference",
            files={"file": ("reference.wav", b"RIFF-qwen-reference", "audio/wav")},
            data={
                "model_id": QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.model_id,
                "instance_id": "qwen-instance",
                "text": "Hello in the cloned voice.",
                "reference_text": "This is the reference voice.",
                "language": "english",
                "speed": "1.0",
            },
        )

    assert response.status_code == 200
    assert captured["reference_audio"] == b"RIFF-qwen-reference"
    request = captured["request"]
    assert request.voice is None
    assert request.language == "english"


def test_qwen3_tts_accepts_synthesis_without_reference(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, object] = {}

    async def fake_synthesize(
        _: object,
        request: object,
        *,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        captured["request"] = request
        captured["reference_audio"] = reference_audio
        captured["reference_text"] = reference_text
        return (
            float32le_to_wav(np.zeros(2400, dtype=np.float32).tobytes(), 24000),
            {
                "x-model-id": QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.model_id,
                "x-runtime": "mlx",
                "x-device": "gpu",
                "x-duration-seconds": "0.1",
                "x-inference-ms": "20.0",
            },
        )

    monkeypatch.setattr(
        "hugging_mac_web.text_to_speech.service.TextToSpeechService.synthesize",
        fake_synthesize,
    )
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/text-to-speech/synthesize",
            json={
                "model_id": QWEN3_TTS_0_6B_BASE_4BIT_PROFILE.model_id,
                "instance_id": "qwen-instance",
                "text": "Hello without a reference voice.",
                "language": "english",
                "speed": 1.0,
            },
        )

    assert response.status_code == 200
    assert captured["reference_audio"] is None
    assert captured["reference_text"] is None


def test_platform_loads_and_unloads_model_instances_with_metrics(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        registry = app.state.context.models.registry
        current = registry.get("ultralytics/yolov8")
        registry.register(
            ModelDefinition(
                manifest=current.manifest,
                runtime_factories={
                    "pytorch-mps": lambda _: PlatformDummyInstance(),
                    "coreml": lambda _: PlatformDummyInstance(),
                    "onnx": lambda _: PlatformDummyInstance(),
                },
                artifacts=current.artifacts,
                converter_ids=current.converter_ids,
                resource_provider=current.resource_provider,
            ),
            replace=True,
        )

        loaded = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/instances",
            json={"runtime": "coreml", "variant": "s", "warmup": False},
        )
        instance_id = loaded.json()["data"]["instance_id"]
        reused = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/instances",
            json={"runtime": "coreml", "variant": "s"},
        )
        catalog = client.get("/api/v1/catalog/models")
        blocked_delete = client.delete(
            "/api/v1/catalog/models/ultralytics/yolov8/resources?variant=s"
        )
        independent_default_delete = client.delete(
            "/api/v1/catalog/models/ultralytics/yolov8/resources"
        )
        blocked_convert = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/convert",
            json={
                "variant": "s",
                "runtime": "coreml",
                "artifact_id": "coreml",
                "overwrite": True,
            },
        )
        unloaded = client.delete(f"/api/v1/catalog/instances/{instance_id}")

    assert loaded.status_code == 201
    assert loaded.json()["data"]["state"] == "ready"
    assert loaded.json()["data"]["variant"] == "s"
    assert loaded.json()["data"]["load_metrics"]["operation"] == "load"
    assert loaded.json()["data"]["load_metrics"]["duration_ms"] >= 0
    assert reused.json()["data"]["instance_id"] == instance_id
    loaded_model = next(
        item for item in catalog.json()["data"] if item["model_id"] == "ultralytics/yolov8"
    )
    assert loaded_model["instance_count"] == 1
    assert blocked_delete.status_code == 409
    assert blocked_delete.json()["error"]["details"]["instance_count"] == 1
    assert independent_default_delete.status_code == 200
    assert independent_default_delete.json()["data"]["variant"] == "n"
    assert blocked_convert.status_code == 409
    assert unloaded.status_code == 200
    assert unloaded.json()["data"]["state"] == "unloaded"
    assert unloaded.json()["data"]["metrics"]["operation"] == "unload"
    assert unloaded.json()["data"]["metrics"]["duration_ms"] >= 0


def test_platform_loads_registered_yolov8_factory(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Keep the Web endpoint wired to the SDK's real registered model factory."""

    async def fake_resolve(_: YoloV8Instance) -> None:
        return None

    async def fake_load(_: YoloV8Instance) -> None:
        return None

    monkeypatch.setattr(YoloV8Instance, "_resolve", fake_resolve)
    monkeypatch.setattr(YoloV8Instance, "_load", fake_load)
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        loaded = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/instances",
            json={"runtime": "pytorch-mps", "variant": "n", "warmup": False},
        )
        instance_id = loaded.json()["data"]["instance_id"]
        unloaded = client.delete(f"/api/v1/catalog/instances/{instance_id}")

    assert loaded.status_code == 201
    assert loaded.json()["data"]["state"] == "ready"
    assert loaded.json()["data"]["runtime"] == "pytorch-mps"
    assert loaded.json()["data"]["variant"] == "n"
    assert unloaded.status_code == 200
    assert unloaded.json()["data"]["state"] == "unloaded"


def test_platform_downloads_overwrites_and_deletes_model_resources(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    download_overwrite_values: list[bool] = []

    async def fake_download(
        _: ResourceDownloader,
        source: Any,
        destination: Path,
        **kwargs: object,
    ) -> ResolvedResource:
        download_overwrite_values.append(bool(kwargs.get("overwrite")))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"replacement weights")
        return ResolvedResource(
            path=destination,
            source=source,
            digest="downloaded",
            size_bytes=destination.stat().st_size,
        )

    async def fake_convert(
        _: YoloV8Converter,
        request: ConversionRequest,
    ) -> ConversionResult:
        if request.target_format is ArtifactFormat.COREML:
            request.output_path.mkdir(parents=True, exist_ok=True)
            (request.output_path / "model.mlmodel").write_bytes(b"coreml")
        else:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            request.output_path.write_bytes(b"onnx")
        return ConversionResult(
            path=request.output_path,
            format=request.target_format,
            digest="0" * 64,
            size_bytes=6,
            converter_id="ultralytics.yolov8",
        )

    monkeypatch.setattr(ResourceDownloader, "download", fake_download)
    monkeypatch.setattr(YoloV8Converter, "convert", fake_convert)
    app = create_app(_settings(tmp_path))
    source_command = {
        "variant": "n",
        "runtime": "pytorch-mps",
        "artifact_id": "source",
    }
    coreml_command = {
        "variant": "n",
        "runtime": "coreml",
        "artifact_id": "coreml",
        "overwrite": True,
    }
    onnx_command = {
        "variant": "n",
        "runtime": "onnx",
        "artifact_id": "onnx",
        "overwrite": True,
    }

    with TestClient(app) as client:
        initial = client.get("/api/v1/catalog/models/ultralytics/yolov8/resources")
        downloaded = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/download-one?overwrite=true",
            json=source_command,
        )
        redownloaded = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/download-one?overwrite=true",
            json=source_command,
        )
        converted = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/convert",
            json=coreml_command,
        )
        reconverted = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/convert",
            json=coreml_command,
        )
        onnx_conversion = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/convert",
            json=onnx_command,
        )
        deleted = client.delete("/api/v1/catalog/models/ultralytics/yolov8/resources")

    assert initial.status_code == 200
    assert initial.json()["data"]["total_size_bytes"] == 0
    assert downloaded.status_code == 200
    assert next(
        item
        for item in downloaded.json()["data"]["artifacts"]
        if item["artifact_id"] == "source" and item["variant"] == "n"
    )["available"]
    assert redownloaded.status_code == 200
    assert download_overwrite_values == [True, True]
    assert converted.status_code == 200
    conversion_target = converted.json()["data"]["conversion_targets"][0]
    assert conversion_target["target_format"] == "coreml"
    assert conversion_target["runtime"] == "coreml"
    assert conversion_target["available"]
    assert conversion_target["size_bytes"] == 6
    assert reconverted.status_code == 200
    assert onnx_conversion.status_code == 200
    assert onnx_conversion.json()["data"]["conversion_targets"][1]["available"]
    assert deleted.status_code == 200
    assert deleted.json()["data"]["total_size_bytes"] == 0
    assert not any(item["available"] for item in deleted.json()["data"]["artifacts"])


def test_media_upload_is_validated_and_stored_in_local_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/media/uploads",
            files={"file": ("sample.png", _png(), "image/png")},
        )
        rejected = client.post(
            "/api/v1/media/uploads",
            files={"file": ("payload.bin", b"not-media", "application/octet-stream")},
        )

    assert response.status_code == 201
    uploaded = response.json()["data"]
    assert uploaded["media_type"] == "image"
    assert uploaded["metadata"]["width"] == 8
    assert uploaded["metadata"]["height"] == 6
    cached_files = tuple(settings.cache_dir.rglob("*.png"))
    assert len(cached_files) == 1
    assert cached_files[0].read_bytes() == _png()
    assert settings.resolved_database_path.exists()
    assert rejected.status_code == 415
    assert rejected.json()["error"]["code"] == "http_error"


class FakeDetector:
    async def detect(self, request: Any) -> DetectionResponse:
        return DetectionResponse(
            model_id="ultralytics/yolov8",
            instance_id="fake-instance",
            runtime="coreml",
            device="all",
            image_size=ImageSize(width=8, height=6),
            detections=(
                Detection(
                    box=BoundingBox(x1=1, y1=1, x2=6, y2=5),
                    confidence=0.91,
                    class_id=0,
                    label="person",
                ),
            ),
            timings=DetectionTimings(
                preprocess_ms=1.0,
                inference_ms=4.0,
                postprocess_ms=0.5,
            ),
        )


class FakeHandle:
    async def __aenter__(self) -> FakeHandle:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def require(self, capability: object) -> FakeDetector:
        return FakeDetector()


def test_object_detection_app_maps_sdk_result(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        acquired_variants: list[str] = []

        async def fake_acquire(*_: object, **kwargs: object) -> FakeHandle:
            acquired_variants.append(str(kwargs["variant"]))
            return FakeHandle()

        app.state.context.models.acquire = fake_acquire
        resource_status = client.get("/api/v1/apps/object-detection/resources?variant=m")
        response = client.post(
            "/api/v1/apps/object-detection/detect",
            files={"file": ("sample.png", _png(), "image/png")},
            data={
                "runtime": "auto",
                "variant": "s",
                "confidence": "0.25",
                "iou_threshold": "0.7",
                "max_detections": "100",
            },
        )
        uncached_response = client.post(
            "/api/v1/apps/object-detection/detect/frame?variant=m",
            content=_png(),
            headers={"Content-Type": "image/png"},
        )

    assert response.status_code == 200
    result = response.json()["data"]
    assert resource_status.status_code == 200
    assert resource_status.json()["data"]["variant"] == "m"
    assert [item["name"] for item in resource_status.json()["data"]["variants"]] == [
        "n",
        "s",
        "m",
    ]
    assert result["variant"] == "s"
    assert result["runtime"] == "coreml"
    assert result["detections"][0]["label"] == "person"
    assert result["detections"][0]["confidence"] == 0.91
    assert result["input_cache_id"].startswith("object-detection-inputs:")
    assert uncached_response.status_code == 200
    assert uncached_response.json()["data"]["input_cache_id"] is None
    assert uncached_response.json()["data"]["variant"] == "m"
    assert acquired_variants == ["s", "m"]


class FakePoseEstimator:
    events: list[str] | None = None
    requests: list[Any] | None = None

    async def estimate_pose(self, request: Any) -> PoseEstimationResponse:
        if self.events is not None:
            self.events.append("pose:start")
            await asyncio.sleep(0)
            self.events.append("pose:end")
        if self.requests is not None:
            self.requests.append(request)
        return PoseEstimationResponse(
            model_id="ultralytics/yolov8-pose",
            instance_id="fake-pose-instance",
            runtime="coreml",
            device="all",
            image_size=ImageSize(width=8, height=6),
            poses=(
                Pose(
                    box=BoundingBox(x1=1, y1=0, x2=7, y2=6),
                    confidence=0.94,
                    class_id=0,
                    label="person",
                    keypoints=(
                        Keypoint(x=4, y=1, confidence=0.98),
                        Keypoint(x=3, y=2, confidence=0.91),
                    ),
                ),
            ),
            timings=DetectionTimings(inference_ms=3.5),
        )


class FakeFaceDetector:
    events: list[str] | None = None
    requests: list[Any] | None = None

    async def detect_faces(self, request: Any) -> FaceDetectionResponse:
        if self.events is not None:
            self.events.append("face:start")
            await asyncio.sleep(0)
            self.events.append("face:end")
        if self.requests is not None:
            self.requests.append(request)
        return FaceDetectionResponse(
            model_id="py-feat/retinaface",
            instance_id="fake-face-instance",
            runtime="coreml",
            device="all",
            image_size=ImageSize(width=8, height=6),
            faces=(
                FaceDetection(
                    box=BoundingBox(x1=2, y1=1, x2=6, y2=5),
                    confidence=0.97,
                    landmarks=FaceLandmarks5(
                        left_eye=Point2D(x=3, y=2),
                        right_eye=Point2D(x=5, y=2),
                        nose=Point2D(x=4, y=3),
                        left_mouth=Point2D(x=3, y=4),
                        right_mouth=Point2D(x=5, y=4),
                    ),
                ),
            ),
            timings=DetectionTimings(inference_ms=2.5),
        )


class FakeHandDetector:
    events: list[str] | None = None
    requests: list[Any] | None = None

    async def detect_hands(self, request: Any) -> HandDetectionResponse:
        if self.events is not None:
            self.events.append("hand:start")
            await asyncio.sleep(0)
            self.events.append("hand:end")
        if self.requests is not None:
            self.requests.append(request)
        return HandDetectionResponse(
            model_id="qualcomm/mediapipe-hand-detection",
            instance_id="fake-hand-instance",
            runtime="coreml",
            device="cpu-and-neural-engine",
            image_size=ImageSize(width=8, height=6),
            hands=(
                HandResult(
                    box=BoundingBox(x1=1, y1=1, x2=4, y2=5),
                    confidence=0.91,
                ),
            ),
            landmarks_enabled=request.include_landmarks,
            timings=DetectionTimings(inference_ms=1.5),
        )


class FakeSegmenter:
    async def segment(self, request: Any) -> SegmentationResponse:
        return SegmentationResponse(
            model_id="ultralytics/yolov8-seg",
            instance_id="fake-seg-instance",
            runtime="coreml",
            device="all",
            image_size=ImageSize(width=8, height=6),
            segments=(
                Segmentation(
                    box=BoundingBox(x1=1, y1=1, x2=7, y2=5),
                    confidence=0.89,
                    class_id=0,
                    label="person",
                    polygons=(
                        (
                            PolygonPoint(x=1, y=1),
                            PolygonPoint(x=7, y=1),
                            PolygonPoint(x=7, y=5),
                        ),
                    ),
                ),
            ),
            timings=DetectionTimings(inference_ms=5.0),
        )


class FakeCapabilityHandle:
    def __init__(self, capability: object) -> None:
        self.capability = capability

    async def __aenter__(self) -> FakeCapabilityHandle:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def require(self, capability: object) -> object:
        return self.capability


def test_pose_and_segmentation_apps_map_sdk_results(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    inference_events: list[str] = []
    pose_requests: list[Any] = []
    face_requests: list[Any] = []
    hand_requests: list[Any] = []
    FakePoseEstimator.events = inference_events
    FakeFaceDetector.events = inference_events
    FakeHandDetector.events = inference_events
    FakePoseEstimator.requests = pose_requests
    FakeFaceDetector.requests = face_requests
    FakeHandDetector.requests = hand_requests

    with TestClient(app) as client:
        acquired: list[tuple[str, str]] = []
        acquired_runtimes: list[tuple[str, object]] = []

        async def fake_acquire(model_id: str, **kwargs: object) -> FakeCapabilityHandle:
            acquired.append((model_id, str(kwargs["variant"])))
            acquired_runtimes.append((model_id, kwargs.get("runtime")))
            capability: object
            if model_id == "ultralytics/yolov8-pose":
                capability = FakePoseEstimator()
            elif model_id == "py-feat/retinaface":
                capability = FakeFaceDetector()
            elif model_id == "qualcomm/mediapipe-hand-detection":
                capability = FakeHandDetector()
            else:
                capability = FakeSegmenter()
            return FakeCapabilityHandle(capability)

        app.state.context.models.acquire = fake_acquire
        pose_resources = client.get("/api/v1/apps/pose-estimation/resources?variant=s")
        face_resources = client.get("/api/v1/apps/pose-estimation/resources/face")
        hand_resources = client.get("/api/v1/apps/pose-estimation/resources/hand")
        pose_response = client.post(
            "/api/v1/apps/pose-estimation/estimate",
            files={"file": ("sample.png", _png(), "image/png")},
            data={
                "runtime": "auto",
                "variant": "m",
                "confidence": "0.61",
                "face_confidence": "0.72",
                "hand_confidence": "0.83",
                "hand_landmarks_enabled": "true",
                "hand_input_mirrored": "true",
            },
        )
        seg_resources = client.get("/api/v1/apps/instance-segmentation/resources?variant=m")
        seg_response = client.post(
            "/api/v1/apps/instance-segmentation/segment/frame?variant=s",
            content=_png(),
            headers={"Content-Type": "image/png"},
        )
        face_only_response = client.post(
            "/api/v1/apps/pose-estimation/estimate/frame"
            "?pose_enabled=false&face_enabled=true&face_confidence=0.8&hand_enabled=false",
            content=_png(),
            headers={"Content-Type": "image/png"},
        )
        hand_only_response = client.post(
            "/api/v1/apps/pose-estimation/estimate/frame"
            "?pose_enabled=false&face_enabled=false&hand_enabled=true&hand_confidence=0.83"
            "&variant=float&hand_landmarks_enabled=true&hand_input_mirrored=true",
            content=_png(),
            headers={"Content-Type": "image/png"},
        )
        disabled_response = client.post(
            "/api/v1/apps/pose-estimation/estimate/frame"
            "?pose_enabled=false&face_enabled=false&hand_enabled=false",
            content=_png(),
            headers={"Content-Type": "image/png"},
        )

    assert pose_resources.status_code == 200
    assert pose_resources.json()["data"]["variant"] == "s"
    assert face_resources.status_code == 200
    assert face_resources.json()["data"]["model_id"] == "py-feat/retinaface"
    assert hand_resources.status_code == 200
    assert hand_resources.json()["data"]["model_id"] == "qualcomm/mediapipe-hand-detection"
    assert pose_response.status_code == 200
    pose_result = pose_response.json()["data"]
    assert pose_result["variant"] == "m"
    assert pose_result["poses"][0]["keypoints"][0] == {
        "x": 4.0,
        "y": 1.0,
        "confidence": 0.98,
    }
    assert pose_result["input_cache_id"].startswith("pose-estimation-inputs:")
    assert pose_result["faces"][0]["landmarks"]["nose"] == {
        "x": 4.0,
        "y": 3.0,
        "confidence": None,
    }
    assert pose_result["hands"][0] == {
        "box": {"x1": 1.0, "y1": 1.0, "x2": 4.0, "y2": 5.0},
        "confidence": 0.91,
        "label": "hand",
        "handedness": None,
        "handedness_confidence": None,
        "landmark_confidence": None,
        "landmarks": [],
    }
    parallel_timings = pose_result["parallel_timings"]
    assert parallel_timings["pose_ms"] >= 0
    assert parallel_timings["face_ms"] >= 0
    assert parallel_timings["hand_ms"] >= 0
    assert parallel_timings["sum_ms"] == (
        parallel_timings["pose_ms"] + parallel_timings["face_ms"] + parallel_timings["hand_ms"]
    )
    assert parallel_timings["round_ms"] >= max(
        parallel_timings["pose_ms"],
        parallel_timings["face_ms"],
        parallel_timings["hand_ms"],
    )

    assert seg_resources.status_code == 200
    assert seg_resources.json()["data"]["variant"] == "m"
    assert seg_response.status_code == 200
    seg_result = seg_response.json()["data"]
    assert seg_result["variant"] == "s"
    assert seg_result["segments"][0]["polygons"][0][2] == {
        "x": 7.0,
        "y": 5.0,
    }
    assert seg_result["input_cache_id"] is None
    assert face_only_response.status_code == 200
    assert face_only_response.json()["data"]["poses"] == []
    assert len(face_only_response.json()["data"]["faces"]) == 1
    assert face_only_response.json()["data"]["parallel_timings"]["pose_ms"] is None
    assert face_only_response.json()["data"]["parallel_timings"]["hand_ms"] is None
    assert hand_only_response.status_code == 200
    assert len(hand_only_response.json()["data"]["hands"]) == 1
    assert hand_only_response.json()["data"]["parallel_timings"]["pose_ms"] is None
    assert hand_only_response.json()["data"]["parallel_timings"]["face_ms"] is None
    assert hand_requests[-1].confidence == 0.83
    assert pose_requests[0].confidence == 0.61
    assert face_requests[0].confidence == 0.72
    assert hand_requests[0].confidence == 0.83
    assert hand_requests[0].include_landmarks is True
    assert hand_requests[0].input_mirrored is True
    assert disabled_response.status_code == 422
    assert inference_events[:6] == [
        "pose:start",
        "face:start",
        "hand:start",
        "pose:end",
        "face:end",
        "hand:end",
    ]
    assert acquired == [
        ("ultralytics/yolov8-pose", "m"),
        ("py-feat/retinaface", "mobilenet0.25"),
        ("qualcomm/mediapipe-hand-detection", "float"),
        ("ultralytics/yolov8-seg", "s"),
        ("py-feat/retinaface", "mobilenet0.25"),
        ("qualcomm/mediapipe-hand-detection", "float"),
    ]
    assert acquired_runtimes[:3] == [
        ("ultralytics/yolov8-pose", None),
        ("py-feat/retinaface", None),
        ("qualcomm/mediapipe-hand-detection", None),
    ]
    FakePoseEstimator.events = None
    FakeFaceDetector.events = None
    FakeHandDetector.events = None
    FakePoseEstimator.requests = None
    FakeFaceDetector.requests = None
    FakeHandDetector.requests = None
