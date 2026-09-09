from __future__ import annotations

import io
from pathlib import Path

import pytest
from hugging_mac_sdk import ArtifactFormat, DetectionRequest, ImageInput, ModelSdk
from hugging_mac_sdk.capabilities import ObjectDetection
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.yolov8 import register_yolov8
from PIL import Image, ImageDraw


@pytest.mark.smoke
@pytest.mark.parametrize("runtime", ("pytorch-mps", "coreml"))
async def test_yolov8_real_model_detection(runtime: str, tmp_path: Path) -> None:
    sdk = ModelSdk()
    register_yolov8(sdk.registry, ConverterRegistry())
    options = {
        "model_home": tmp_path / "models",
        "allow_cpu_fallback": False,
    }

    await sdk.resources.download_source(
        "ultralytics/yolov8",
        variant="n",
        options=options,
    )
    if runtime == "coreml":
        await sdk.resources.convert(
            "ultralytics/yolov8",
            ArtifactFormat.COREML,
            variant="n",
            options=options,
        )

    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 80, 480, 440), fill="navy")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    async with await sdk.acquire(
        "ultralytics/yolov8",
        variant="n",
        runtime=runtime,
        options=options,
    ) as handle:
        result = await handle.require(ObjectDetection).detect(  # type: ignore[type-abstract]
            DetectionRequest(image=ImageInput(data=encoded.getvalue()))
        )

    assert result.runtime == runtime
    assert result.image_size.width == 640
    assert result.image_size.height == 480
