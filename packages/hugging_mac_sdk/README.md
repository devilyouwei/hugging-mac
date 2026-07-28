# hugging-mac-sdk

`hugging-mac` 的独立模型 SDK。它提供运行时无关的模型生命周期、可组合 capability、
模型注册、实例管理与显式模型资源操作。`load` 和推理永远不会自动下载或转换模型。

当前是首个内部版本，公共 API 仍可能随架构验证调整。

## 当前模块

```text
src/hugging_mac_sdk/
├── capabilities/       # detect、chat、transcribe、embedding 等任务协议
├── core/               # 生命周期、模型注册和实例复用
├── resources/          # 下载、hash、原子提交与安全解压
├── runtime/            # PyTorch MPS、MLX、Core ML 等 adapter 契约
├── schemas/            # manifest、资源和健康状态 schema
└── errors.py           # 稳定错误层级
```

## 注册与创建实例

具体模型实现继承薄生命周期基类，通过组合注册 capability：

```python
class YoloInstance(BaseModelInstance):
    def __init__(self) -> None:
        super().__init__()
        self.register_capability(ObjectDetection, YoloDetection(self))

    async def _load(self) -> None:
        ...

    async def _unload(self) -> None:
        ...
```

静态 manifest、artifact 与独立的 runtime factory 组成 definition：

```python
registry.register(
    ModelDefinition(
        manifest=yolo_manifest,
        runtime_factories={
            "pytorch-mps": create_yolo_mps,
            "coreml": create_yolo_coreml,
        },
        artifacts=yolo_artifacts,
    )
)
instance = registry.create_instance("ultralytics/yolo", runtime="pytorch-mps")
await instance.load()
detector = instance.require(ObjectDetection)
```

没有实际 factory 的 runtime 不允许出现在 definition 中。模型 metadata 也可以放在 package 内的
`model.yaml`，通过 `load_model_config()` 安全读取；YAML 不负责导入或执行 runtime 代码。

## Registry、runtime 与实例管理

Registry 只查询，不下载或加载模型：

```python
runtimes = registry.supported_runtimes("ultralytics/yolov8n")
artifact = registry.get_artifact("ultralytics/yolov8n", "coreml")
path = registry.resolve_artifact_path("ultralytics/yolov8n", "coreml")
```

`InstanceManager` 把 runtime policy 纳入创建链路，支持 `auto`、显式 device、多实例和安全 runtime 切换：

```python
manager = InstanceManager(registry)
model = await manager.load("ultralytics/yolov8n", runtime="auto")
print(model.info())

replacement = await manager.switch_runtime(
    str(model.instance_id),
    "pytorch-mps",
    device="mps",
)
await manager.unload_all()
```

加载指标可从实例 snapshot 获取，卸载指标由 `unload_with_metrics()` 返回：

```python
snapshot = await manager.snapshot(str(model.instance_id))
print(snapshot.load_metrics.duration_ms)
print(snapshot.load_metrics.memory_allocated_bytes)

result = await manager.unload_with_metrics(str(model.instance_id))
print(result.metrics.duration_ms)
print(result.metrics.memory_released_bytes)
```

内存值是操作前后 Python 进程 RSS 的观测差值，并非模型独占内存；allocator cache、并发任务和
GPU/ANE 分配可能影响结果。

模型能力仍通过组合接口获取，例如 `instance.require(ObjectDetection)`；不会为所有模型添加含义模糊的
通用 `predict(**kwargs)`。

## 下载模型资源

目标路径始终由调用方明确提供。下载先写入同目录 staging，校验成功后再原子提交。

已注册模型优先通过统一 facade 操作：

```python
status = await sdk.resources.status("ultralytics/yolov8n")
status = await sdk.resources.download_source("ultralytics/yolov8n")
status = await sdk.resources.convert(
    "ultralytics/yolov8n",
    ArtifactFormat.COREML,
)
```

这些调用必须由业务层或用户动作主动触发。

资源状态的 `conversion_targets` 描述可转换的目标格式、对应 runtime、artifact 和本地可用状态，
供 API 或 UI 动态生成 `Convert/Re-convert` 操作。新增 ONNX 等转换时，只需由模型 resource provider
声明目标并实现转换，无需修改平台层路由。

资源状态也会聚合每个 runtime 的本地文件/目录大小以及模型总大小。删除同样是显式操作：

```python
status = await sdk.resources.delete(
    "ultralytics/yolov8n",
    runtime="coreml",  # 省略 runtime 时删除整个模型 revision 目录
)
```

删除不会移除模型 definition。模型实现必须把删除范围限制在自身受管 storage root 内，外部自定义路径和
路径逃逸会被拒绝。

Hugging Face 完整模型目录：

```python
source = HuggingFaceSource(
    repo_id="org/model",
    revision="main",
    allow_patterns=("*.json", "*.safetensors"),
)
resource = await ResourceDownloader().download(source, Path("models/org/model"))
```

Hugging Face 单文件，例如某个 YOLO 变体：

```python
source = HuggingFaceSource(
    repo_id="org/yolo",
    filename="yolo-small.safetensors",
    expected_sha256="...",
)
resource = await ResourceDownloader().download(source, Path("models/yolo-small.safetensors"))
```

指定 URL 单文件：

```python
source = UrlFileSource(url="https://example.com/model.safetensors")
resource = await ResourceDownloader().download(source, Path("models/model.safetensors"))
```

指定 URL 模型目录使用 ZIP/TAR 归档：

```python
source = UrlArchiveSource(
    url="https://example.com/llm.tar.gz",
    strip_components=1,
)
resource = await ResourceDownloader().download(source, Path("models/example-llm"))
```

URL 不存在通用的“列出远端目录”协议，因此目录下载明确要求 ZIP/TAR；Hugging Face 多文件模型使用
snapshot。归档解压默认拒绝路径穿越、链接和特殊文件。

## 转换模型

转换器分为两层：

- 通用转换器按 source/target format 自动匹配；
- 模型 definition 可以通过 `converter_ids` 绑定优先的模型专用转换器。

没有绑定专用转换器，或专用转换器不支持目标格式时，registry 回退到通用转换器。调用方也可以显式指定
converter ID。

安装 YOLO 转换依赖：

```bash
uv sync --package hugging-mac-sdk --extra yolo
```

下载并转换已注册的 YOLOv8n：

```python
from pathlib import Path

from hugging_mac_sdk import (
    ArtifactFormat,
    ConversionRequest,
    ConversionService,
    ConverterRegistry,
    ModelRegistry,
    ResourceDownloader,
)
from hugging_mac_sdk.models.yolov8 import register_yolov8

models = ModelRegistry()
converters = ConverterRegistry()
definition = register_yolov8(models, converters)

source_spec = definition.manifest.resources[0]
source = await ResourceDownloader().download(
    source_spec,
    Path("models/ultralytics/yolov8n.pt"),
)

result = await ConversionService(converters).convert(
    ConversionRequest(
        model_id=definition.manifest.model_id,
        model_revision=definition.manifest.revision,
        source=source,
        source_format=ArtifactFormat.PYTORCH,
        target_format=ArtifactFormat.COREML,
        output_path=Path("models/ultralytics/yolov8n.mlpackage"),
    ),
    definition=definition,
)
```

YOLOv8n 的专用转换器默认使用固定 `640×640`、batch 1、FP16、静态 shape 和导出内置 NMS。
Core ML 运行时配置为 `ComputeUnit.ALL`，但该配置在模型加载阶段应用，不是 Ultralytics 导出参数。

源模型来自 `Ultralytics/YOLOv8` 的 `yolov8n.pt`，revision 与 SHA-256 已固定。该文件包含 Python
pickle 对象，只能作为 manifest 中明确允许的受信任来源加载，不能把任意第三方 `.pt` 当成安全数据文件。

## YOLOv8 推理

注册后可创建 PyTorch MPS 或 Core ML 实例。Core ML 是默认 runtime，但实例只加载已经存在的资产；
缺失时抛出 `ResourceNotFoundError`：

```python
from pathlib import Path

from hugging_mac_sdk import DetectionRequest, ImageInput
from hugging_mac_sdk.capabilities import ObjectDetection
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.yolov8 import register_yolov8

models = ModelRegistry()
register_yolov8(models, ConverterRegistry())

instance = models.create_instance(
    "ultralytics/yolov8n",
    runtime="coreml",
    options={"model_home": Path("models")},
)
await instance.load()

detector = instance.require(ObjectDetection)
result = await detector.detect(
    DetectionRequest(
        image=ImageInput(path=Path("example.jpg")),
        confidence=0.25,
    )
)

await instance.unload()
```

拿到 definition 时也可以直接使用同一个工厂契约：

```python
definition = models.get("ultralytics/yolov8n")
instance = definition.create(
    runtime="coreml",
    options={"model_home": Path("models")},
)
```

`runtime` 会根据 manifest 中的 `RuntimeSpec` 校验。如果 `runtime` 参数与 `options["runtime"]` 冲突，
或者模型没有声明该 runtime，会抛出 `UnsupportedRuntimeError`。

`DetectionResponse` 不包含 torch tensor 或 Core ML 对象，统一返回：

- 原图宽高；
- `xyxy` bounding box；
- confidence；
- class ID 和 COCO label；
- preprocess、inference、postprocess 耗时；
- 声明 runtime 和实际 execution device。

`pytorch-mps` 默认要求 MPS 真实可用，不会静默回退 CPU。诊断环境可以显式设置
`allow_cpu_fallback=True`，响应中的 `device` 会报告实际执行设备。

运行真实 smoke test：

```bash
uv run --all-packages --extra yolo scripts/smoke_yolov8.py --download
uv run --all-packages --extra yolo scripts/smoke_yolov8.py \
  --runtime coreml --download --convert
```
