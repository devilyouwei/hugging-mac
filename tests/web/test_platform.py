from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    Detection,
    DetectionResponse,
    DetectionTimings,
    ImageSize,
)
from hugging_mac_web.config import WebSettings
from hugging_mac_web.main import create_app
from PIL import Image


def _settings(tmp_path: Path) -> WebSettings:
    return WebSettings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        model_home=tmp_path / "models",
        database_path=tmp_path / "data" / "platform.json",
        runtime_preference="coreml,pytorch-mps",
    )


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
        apps = client.get("/api/v1/catalog/apps")
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
    app_summary = apps.json()["data"][0]
    assert app_summary["manifest"]["app_id"] == "object-detection"
    assert app_summary["status"] == "available"
    assert not any(item["available"] for item in resources.json()["data"]["artifacts"])
    assert "/api/v1/apps/object-detection/detect" in openapi.json()["paths"]
    model = models.json()["data"][0]
    assert model["model_id"].startswith("ultralytics/yolov8")
    assert model["instance_count"] == 0
    assert model["instances"] == []
    assert {runtime["name"] for runtime in model["runtimes"]} == {
        "coreml",
        "pytorch-mps",
    }
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"


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
            model_id="ultralytics/yolov8n",
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

        async def fake_acquire(*_: object, **__: object) -> FakeHandle:
            return FakeHandle()

        app.state.context.models.acquire = fake_acquire
        response = client.post(
            "/api/v1/apps/object-detection/detect",
            files={"file": ("sample.png", _png(), "image/png")},
            data={
                "runtime": "auto",
                "confidence": "0.25",
                "iou_threshold": "0.7",
                "max_detections": "100",
            },
        )
        uncached_response = client.post(
            "/api/v1/apps/object-detection/detect/frame",
            content=_png(),
            headers={"Content-Type": "image/png"},
        )

    assert response.status_code == 200
    result = response.json()["data"]
    assert result["runtime"] == "coreml"
    assert result["detections"][0]["label"] == "person"
    assert result["detections"][0]["confidence"] == 0.91
    assert result["input_cache_id"].startswith("object-detection-inputs:")
    assert uncached_response.status_code == 200
    assert uncached_response.json()["data"]["input_cache_id"] is None
