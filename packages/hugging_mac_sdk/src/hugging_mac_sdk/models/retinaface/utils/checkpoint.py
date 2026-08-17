"""Load the pinned py-feat RetinaFace checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import ResourceIntegrityError

from .modeling import build_retinaface


def load_retinaface_checkpoint(source: Path, torch: Any) -> Any:
    config_path = source / "config.json"
    weights_path = source / "mobilenet0.25_Final.pth"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("name") != "mobilenet0.25":
        raise ResourceIntegrityError("Unsupported py-feat RetinaFace architecture")
    model = build_retinaface(torch)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def normalize_retinaface_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise ValueError("RetinaFace must return locations, scores, and landmarks")
    return {"locations": output[0], "scores": output[1], "landmarks": output[2]}
