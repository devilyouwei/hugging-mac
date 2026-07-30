# YOLOv8 Pose 与 Seg SDK

状态：已实现并通过 Core ML 推理验证。

`yolov8_pose` 与 `yolov8_seg` 是独立模型包，分别注册为 `ultralytics/yolov8-pose` 和
`ultralytics/yolov8-seg`。它们不会复用 detection 的 model ID、资源目录、实例或 capability；仅复用
无任务语义的资源管理、YOLO 基础张量算法和通用 runtime provider。

| 模型 | Capability | 方法 | Variant |
|---|---|---|---|
| YOLOv8 Pose | `PoseEstimation` | `estimate_pose(PoseRequest)` | `n`、`s`、`m` |
| YOLOv8 Seg | `InstanceSegmentation` | `segment(SegmentationRequest)` | `n`、`s`、`m` |

两个模型固定到 Ultralytics 官方 release `v8.2.0`。该历史 release 提供了稳定的版本化 URL 与文件大小，
但没有发布机器可读的 SHA-256，因此 manifest 不会伪造摘要；下载器会计算实际摘要并将其保存到
`ResolvedResource`。若上游后续发布官方 checksum，应补入 `expected_sha256`，使下载时启用强制校验。

## 包结构

```text
models/
├── yolov8_pose/
│   ├── __init__.py
│   ├── model.yaml
│   ├── definition.py
│   ├── config.py
│   ├── converter.py
│   ├── resources.py
│   ├── instance.py
│   ├── torch.py
│   ├── coreml.py
│   ├── onnx.py
│   └── utils/
│       ├── __init__.py
│       ├── types.py
│       ├── preprocess.py
│       ├── postprocess.py
│       └── checkpoint.py
└── yolov8_seg/
    ├── __init__.py
    ├── model.yaml
    ├── definition.py
    ├── config.py
    ├── converter.py
    ├── resources.py
    ├── instance.py
    ├── torch.py
    ├── coreml.py
    ├── onnx.py
    └── utils/
        ├── __init__.py
        ├── types.py
        ├── preprocess.py
        ├── postprocess.py
        └── checkpoint.py
```

每个 package 有独立 manifest、variant sources、专用 converter ID、runtime factories 和 resource provider。
Pose 与 Seg 各自在自己的 `utils/` 中保留中间类型、预处理、后处理和 checkpoint 实现；即使 letterbox、
box 解码与 NMS 相同也不跨模型导入。统一 instance 组合来自 `torch.py`、`coreml.py` 或 `onnx.py`
的 engine；engine 调用对应公共 provider。这些实现不依赖 Ultralytics Python 包。

## 资源与转换

```python
await sdk.resources.download_source(
    "ultralytics/yolov8-pose",
    variant="s",
)
await sdk.resources.convert(
    "ultralytics/yolov8-seg",
    ArtifactFormat.COREML,
    variant="m",
)
await sdk.resources.convert(
    "ultralytics/yolov8-pose",
    ArtifactFormat.ONNX,
    variant="n",
)
```

资源目录相互隔离，例如：

```text
<model-home>/ultralytics/yolov8-pose/v8.2.0/s/source/yolov8s-pose.pt
<model-home>/ultralytics/yolov8-seg/v8.2.0/m/coreml/yolov8m-seg.mlpackage
```

加载、下载、转换、删除和共享实例 identity 都包含 variant。Core ML 转换使用任务专用 converter：Pose 与
Seg 强制 `nms=False`；detection 同样导出原始输出。各自 instance 在 SDK 内完成 class-aware NMS、
关键点或 mask 解析，使 PyTorch、Core ML 和 ONNX 得到一致协议。

## 输出边界

Pose 返回检测框、人物标签与每个关键点的 `(x, y, confidence)`；Seg 返回检测框、标签与实例轮廓 polygon。
Seg `utils/postprocess.py` 将 mask coefficient 与 prototype 合成二值 mask，再提取并转换为不可变
`PolygonPoint` tuple。公共响应不暴露 torch tensor、NumPy array、Core ML feature 或二进制 mask；
业务层如需要栅格化 mask，应根据 polygon 在自己的 media/cache 层完成。
