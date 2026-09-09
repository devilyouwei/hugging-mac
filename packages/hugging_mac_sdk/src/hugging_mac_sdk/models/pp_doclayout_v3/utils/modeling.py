"""Load PP-DocLayoutV3 without leaking Transformers objects into SDK contracts."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError


def load_pp_doclayout_v3(source: Path, torch: Any) -> Any:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise UnsupportedRuntimeError(
            "PP-DocLayoutV3 requires the SDK layout extra", cause=error
        ) from error
    model = transformers.AutoModelForObjectDetection.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False
    )
    model.eval()
    return _RawOutputWrapper(model)


class _RawOutputWrapper:
    """Small callable wrapper used by the generic Torch runtime."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def to(self, *args: Any, **kwargs: Any) -> _RawOutputWrapper:
        self.model.to(*args, **kwargs)
        return self

    def eval(self) -> _RawOutputWrapper:
        self.model.eval()
        return self

    def __call__(self, pixel_values: Any) -> tuple[Any, Any, Any, Any]:
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs.logits, outputs.pred_boxes, outputs.order_logits, outputs.out_masks


def normalize_outputs(output: Any) -> dict[str, Any]:
    logits, boxes, order_logits, masks = output
    return {
        "logits": _numpy(logits),
        "pred_boxes": _numpy(boxes),
        "order_logits": _numpy(order_logits),
        "out_masks": _numpy(masks),
    }


def _numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.numpy() if hasattr(value, "numpy") else value
