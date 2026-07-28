from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from hugging_mac_sdk import (
    BaseModelInstance,
    ConversionRequest,
    ConversionResult,
    ModelDefinition,
)
from hugging_mac_sdk.models.yolov8.config import YOLOV8N_SHA256
from hugging_mac_sdk.models.yolov8.converter import YoloV8Converter
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    Detection,
    DetectionResponse,
    DetectionTimings,
    ImageSize,
)
from hugging_mac_sdk.schemas.resources import ResolvedResource
from hugging_mac_web.config import WebSettings
from hugging_mac_web.main import create_app
from PIL import Image


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
    assert {
        "chip_name",
        "cpu_performance_cores",
        "cpu_efficiency_cores",
        "gpu_cores",
        "neural_engine_cores",
    } <= info.json()["data"].keys()
    app_summary = apps.json()["data"][0]
    assert app_summary["manifest"]["app_id"] == "object-detection"
    assert app_summary["status"] == "available"
    assert not any(item["available"] for item in resources.json()["data"]["artifacts"])
    assert "/api/v1/apps/object-detection/detect" in openapi.json()["paths"]
    model = models.json()["data"][0]
    assert model["model_id"] == "ultralytics/yolov8"
    assert [variant["name"] for variant in model["variants"]] == ["n", "s", "m"]
    assert model["variants"][0]["default"]
    assert model["default_variant"] == "n"
    assert model["instance_count"] == 0
    assert model["instances"] == []
    assert {runtime["name"] for runtime in model["runtimes"]} == {
        "coreml",
        "pytorch-mps",
    }
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"


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
        catalog = client.get("/api/v1/catalog/models")
        blocked_delete = client.delete(
            "/api/v1/catalog/models/ultralytics/yolov8/resources?variant=s"
        )
        independent_default_delete = client.delete(
            "/api/v1/catalog/models/ultralytics/yolov8/resources"
        )
        blocked_convert = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/convert",
            json={"target_format": "coreml", "variant": "s"},
        )
        unloaded = client.delete(f"/api/v1/catalog/instances/{instance_id}")

    assert loaded.status_code == 201
    assert loaded.json()["data"]["state"] == "ready"
    assert loaded.json()["data"]["variant"] == "s"
    assert loaded.json()["data"]["load_metrics"]["operation"] == "load"
    assert loaded.json()["data"]["load_metrics"]["duration_ms"] >= 0
    assert catalog.json()["data"][0]["instance_count"] == 1
    assert blocked_delete.status_code == 409
    assert blocked_delete.json()["error"]["details"]["instance_count"] == 1
    assert independent_default_delete.status_code == 200
    assert independent_default_delete.json()["data"]["variant"] == "n"
    assert blocked_convert.status_code == 409
    assert unloaded.status_code == 200
    assert unloaded.json()["data"]["state"] == "unloaded"
    assert unloaded.json()["data"]["metrics"]["operation"] == "unload"
    assert unloaded.json()["data"]["metrics"]["duration_ms"] >= 0


def test_platform_downloads_overwrites_and_deletes_model_resources(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def fake_download(
        _: ResourceDownloader,
        source: Any,
        destination: Path,
        **__: object,
    ) -> ResolvedResource:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"replacement weights")
        return ResolvedResource(
            path=destination,
            source=source,
            digest=YOLOV8N_SHA256,
            size_bytes=destination.stat().st_size,
        )

    async def fake_convert(
        _: YoloV8Converter,
        request: ConversionRequest,
    ) -> ConversionResult:
        request.output_path.mkdir(parents=True, exist_ok=True)
        (request.output_path / "model.mlmodel").write_bytes(b"coreml")
        return ConversionResult(
            path=request.output_path,
            format=request.target_format,
            digest="0" * 64,
            size_bytes=6,
            converter_id="ultralytics.yolov8",
        )

    monkeypatch.setattr(ResourceDownloader, "download", fake_download)
    monkeypatch.setattr(YoloV8Converter, "convert", fake_convert)
    monkeypatch.setattr(
        "hugging_mac_sdk.models.yolov8.assets.file_sha256",
        lambda _: YOLOV8N_SHA256,
    )
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        initial = client.get(
            "/api/v1/catalog/models/ultralytics/yolov8/resources"
        )
        downloaded = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/download"
        )
        redownloaded = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/download"
        )
        converted = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/convert",
            json={"target_format": "coreml"},
        )
        reconverted = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/convert",
            json={"target_format": "coreml"},
        )
        unsupported = client.post(
            "/api/v1/catalog/models/ultralytics/yolov8/resources/convert",
            json={"target_format": "onnx"},
        )
        deleted = client.delete(
            "/api/v1/catalog/models/ultralytics/yolov8/resources"
        )

    assert initial.status_code == 200
    assert initial.json()["data"]["total_size_bytes"] == 0
    assert downloaded.status_code == 200
    assert downloaded.json()["data"]["total_size_bytes"] == len(b"replacement weights")
    assert redownloaded.status_code == 200
    assert redownloaded.json()["data"]["artifacts"][0]["available"]
    assert converted.status_code == 200
    conversion_target = converted.json()["data"]["conversion_targets"][0]
    assert conversion_target["target_format"] == "coreml"
    assert conversion_target["runtime"] == "coreml"
    assert conversion_target["available"]
    assert conversion_target["size_bytes"] == 6
    assert reconverted.status_code == 200
    assert unsupported.status_code == 409
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
        resource_status = client.get(
            "/api/v1/apps/object-detection/resources?variant=m"
        )
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
