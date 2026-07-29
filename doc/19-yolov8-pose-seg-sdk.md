# YOLOv8 Pose 与 Seg SDK

状态：已实现。

`yolov8_pose` 与 `yolov8_seg` 是独立模型包，分别注册为 `ultralytics/yolov8-pose` 和
`ultralytics/yolov8-seg`。它们不会复用 detection 的 model ID、资源目录、实例或 capability；仅复用
Ultralytics 的下载、转换和 runtime 支持代码。

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
│   ├── model.yaml
│   ├── config.py
│   ├── assets.py
│   ├── converter.py
│   ├── instance.py
│   ├── catalog.py
│   └── __init__.py
└── yolov8_seg/
    ├── model.yaml
    ├── config.py
    ├── assets.py
    ├── converter.py
    ├── instance.py
    ├── catalog.py
    └── __init__.py
```

每个 package 有独立 manifest、variant sources、专用 converter ID、runtime factories 和 resource provider。
公共 `_ultralytics_assets.py` 与 `_ultralytics_runtime.py` 只负责无任务语义的资源生命周期、图片解码和预测
参数；不承载 Pose/Seg 的业务输出解析。

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
```

资源目录相互隔离，例如：

```text
<model-home>/ultralytics/yolov8-pose/v8.2.0/s/source/yolov8s-pose.pt
<model-home>/ultralytics/yolov8-seg/v8.2.0/m/coreml/yolov8m-seg.mlpackage
```

加载、下载、转换、删除和共享实例 identity 都包含 variant。Core ML 转换使用任务专用 converter，输出的
关键点或 mask 解析仍在各自 instance 内完成。

## 输出边界

Pose 返回检测框、人物标签与每个关键点的 `(x, y, confidence)`；Seg 返回检测框、标签与实例轮廓 polygon。
SDK 不在公共响应中暴露 torch tensor、NumPy array、Core ML feature 或二进制 mask。业务层如需要栅格化 mask，
应根据 polygon 在自己的 media/cache 层完成。
