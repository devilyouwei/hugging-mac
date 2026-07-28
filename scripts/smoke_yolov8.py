"""Explicitly prepare one YOLOv8 variant and run an SDK detection."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path

from hugging_mac_sdk import ArtifactFormat, DetectionRequest, ImageInput, ModelSdk
from hugging_mac_sdk.capabilities import ObjectDetection
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.yolov8 import register_yolov8
from PIL import Image, ImageDraw


async def run(
    runtime: str,
    model_home: Path,
    allow_cpu_fallback: bool,
    variant: str,
    *,
    download: bool,
    convert: bool,
) -> None:
    sdk = ModelSdk()
    register_yolov8(sdk.registry, ConverterRegistry())
    options = {
        "model_home": model_home,
        "allow_cpu_fallback": allow_cpu_fallback,
    }
    if download:
        await sdk.resources.download_source(
            "ultralytics/yolov8",
            variant=variant,
            options=options,
        )
    if convert:
        await sdk.resources.convert(
            "ultralytics/yolov8",
            ArtifactFormat.COREML,
            variant=variant,
            options=options,
        )

    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 80, 480, 440), fill="navy")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    async with await sdk.acquire(
        "ultralytics/yolov8",
        variant=variant,
        runtime=runtime,
        options=options,
    ) as handle:
        result = await handle.require(ObjectDetection).detect(  # type: ignore[type-abstract]
            DetectionRequest(image=ImageInput(data=encoded.getvalue()))
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        choices=("pytorch-mps", "coreml"),
        default="pytorch-mps",
    )
    parser.add_argument(
        "--model-home",
        type=Path,
        default=Path(".cache/hugging-mac-smoke"),
    )
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--variant", choices=("n", "s", "m"), default="n")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download the pinned source before loading",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Explicitly create the Core ML artifact before loading",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.runtime,
            args.model_home,
            args.allow_cpu_fallback,
            args.variant,
            download=args.download,
            convert=args.convert,
        )
    )


if __name__ == "__main__":
    main()
