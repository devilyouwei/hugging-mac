"""Private loader for the pinned YOLOv8 Pose PyTorch checkpoints.

The upstream files pickle model class names instead of storing only a state
dictionary. This loader maps that small, known YOLOv8 graph vocabulary to
local implementations, so inference and conversion do not require the
Ultralytics Python package.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError


def load_yolov8_checkpoint(path: Path, torch: Any) -> Any:
    classes = _compatibility_types(torch)

    class CompatibilityUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module.startswith("ultralytics.nn."):
                try:
                    return classes[name]
                except KeyError as error:
                    raise pickle.UnpicklingError(
                        f"Unsupported YOLOv8 checkpoint class: {module}.{name}"
                    ) from error
            return super().find_class(module, name)

    pickle_module = SimpleNamespace(
        __name__="hugging_mac_sdk_yolov8_pickle",
        Unpickler=CompatibilityUnpickler,
        load=pickle.load,
        loads=pickle.loads,
        dump=pickle.dump,
        dumps=pickle.dumps,
        Pickler=pickle.Pickler,
    )
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            pickle_module=pickle_module,
            weights_only=False,
        )
    except Exception as error:
        raise UnsupportedRuntimeError(
            f"Unsupported or invalid YOLOv8 checkpoint: {path}",
            cause=error,
        ) from error

    model = checkpoint.get("ema") if isinstance(checkpoint, dict) else checkpoint
    if model is None and isinstance(checkpoint, dict):
        model = checkpoint.get("model")
    if model is None:
        raise UnsupportedRuntimeError(f"YOLOv8 checkpoint contains no model: {path}")
    model = model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def normalize_yolov8_output(output: Any) -> dict[str, Any]:
    """Normalize local detection/pose/seg graph outputs for TorchSession."""

    if isinstance(output, tuple):
        if len(output) == 2 and _is_tensor(output[0]) and _is_tensor(output[1]):
            return {"predictions": output[0], "prototypes": output[1]}
        output = output[0]
    return {"predictions": output}


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "detach")


def _compatibility_types(torch: Any) -> dict[str, type[Any]]:
    nn = torch.nn

    class Conv(nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, value: Any) -> Any:
            return self.act(self.bn(self.conv(value)))

        def forward_fuse(self, value: Any) -> Any:
            return self.act(self.conv(value))

    class Bottleneck(nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, value: Any) -> Any:
            result = self.cv2(self.cv1(value))
            return value + result if getattr(self, "add", False) else result

    class C2f(nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, value: Any) -> Any:
            parts = list(self.cv1(value).chunk(2, 1))
            parts.extend(module(parts[-1]) for module in self.m)
            return self.cv2(torch.cat(parts, 1))

        def forward_split(self, value: Any) -> Any:
            parts = list(self.cv1(value).split((self.c, self.c), 1))
            parts.extend(module(parts[-1]) for module in self.m)
            return self.cv2(torch.cat(parts, 1))

    class SPPF(nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, value: Any) -> Any:
            first = self.cv1(value)
            second = self.m(first)
            third = self.m(second)
            fourth = self.m(third)
            return self.cv2(torch.cat((first, second, third, fourth), 1))

    class Concat(nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, values: list[Any]) -> Any:
            return torch.cat(values, getattr(self, "d", 1))

    class DFL(nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, value: Any) -> Any:
            batch, channels, anchors = value.shape
            bins = channels // 4
            value = value.view(batch, 4, bins, anchors).transpose(2, 1)
            return self.conv(value.softmax(1)).view(batch, 4, anchors)

    class Detect(nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, features: list[Any]) -> Any:
            batch = features[0].shape[0]
            levels = [
                torch.cat((self.cv2[index](feature), self.cv3[index](feature)), 1)
                for index, feature in enumerate(features)
            ]
            merged = torch.cat(
                [level.view(batch, int(self.no), -1) for level in levels],
                2,
            )
            distributions, scores = merged.split((int(self.reg_max) * 4, int(self.nc)), 1)
            anchors, strides = _make_anchors(levels, self.stride, torch)
            distances = self.dfl(distributions)
            boxes = _distance_to_box(distances, anchors, torch) * strides
            return torch.cat((boxes, scores.sigmoid()), 1)

    class Pose(Detect):
        def forward(self, features: list[Any]) -> Any:
            batch = features[0].shape[0]
            raw_keypoints = torch.cat(
                [
                    self.cv4[index](feature).view(batch, int(self.nk), -1)
                    for index, feature in enumerate(features)
                ],
                2,
            )
            detections = super().forward(features)
            anchors, strides = _make_anchors(features, self.stride, torch)
            dimensions = int(self.kpt_shape[1])
            keypoints = raw_keypoints.clone()
            if dimensions == 3:
                keypoints[:, 2::dimensions] = keypoints[:, 2::dimensions].sigmoid()
            keypoints[:, 0::dimensions] = (
                keypoints[:, 0::dimensions] * 2.0 + (anchors[0] - 0.5)
            ) * strides
            keypoints[:, 1::dimensions] = (
                keypoints[:, 1::dimensions] * 2.0 + (anchors[1] - 0.5)
            ) * strides
            return torch.cat((detections, keypoints), 1)

    class Proto(nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, value: Any) -> Any:
            return self.cv3(self.cv2(self.upsample(self.cv1(value))))

    class Segment(Detect):
        def forward(self, features: list[Any]) -> tuple[Any, Any]:
            prototypes = self.proto(features[0])
            batch = prototypes.shape[0]
            coefficients = torch.cat(
                [
                    self.cv4[index](feature).view(batch, int(self.nm), -1)
                    for index, feature in enumerate(features)
                ],
                2,
            )
            detections = super().forward(features)
            return torch.cat((detections, coefficients), 1), prototypes

    class DetectionModel(nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, value: Any, *_: Any, **__: Any) -> Any:
            saved: list[Any] = []
            current = value
            save_indices = set(getattr(self, "save", ()))
            for module in self.model:
                source = getattr(module, "f", -1)
                if source != -1:
                    if isinstance(source, int):
                        current = saved[source]
                    else:
                        current = [current if index == -1 else saved[index] for index in source]
                current = module(current)
                saved.append(current if getattr(module, "i", -1) in save_indices else None)
            return current

        def predict(self, value: Any, *_: Any, **__: Any) -> Any:
            return self.forward(value)

        def fuse(self, *_: Any, **__: Any) -> Any:
            return self

    class PoseModel(DetectionModel):
        pass

    class SegmentationModel(DetectionModel):
        pass

    return {
        "Bottleneck": Bottleneck,
        "C2f": C2f,
        "Concat": Concat,
        "Conv": Conv,
        "DFL": DFL,
        "Detect": Detect,
        "DetectionModel": DetectionModel,
        "Pose": Pose,
        "PoseModel": PoseModel,
        "Proto": Proto,
        "SPPF": SPPF,
        "Segment": Segment,
        "SegmentationModel": SegmentationModel,
    }


def _make_anchors(features: list[Any], stride: Any, torch: Any) -> tuple[Any, Any]:
    points: list[Any] = []
    scales: list[Any] = []
    dtype, device = features[0].dtype, features[0].device
    for index, feature in enumerate(features):
        height, width = feature.shape[2:]
        x = torch.arange(width, device=device, dtype=dtype) + 0.5
        y = torch.arange(height, device=device, dtype=dtype) + 0.5
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        points.append(torch.stack((grid_x, grid_y), -1).view(-1, 2))
        scales.append(
            torch.full(
                (height * width, 1),
                float(stride[index]),
                dtype=dtype,
                device=device,
            )
        )
    return torch.cat(points).transpose(0, 1), torch.cat(scales).transpose(0, 1)


def _distance_to_box(distance: Any, anchors: Any, torch: Any) -> Any:
    left_top, right_bottom = distance.chunk(2, 1)
    minimum = anchors.unsqueeze(0) - left_top
    maximum = anchors.unsqueeze(0) + right_bottom
    center = (minimum + maximum) / 2
    size = maximum - minimum
    return torch.cat((center, size), 1)
