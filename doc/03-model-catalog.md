# 模型目录与能力

状态：`Proposed`

## 分类不是实现

模型按“任务能力”注册，而不是按页面或脚本注册。同一模型可以暴露多个能力，同一能力也可以有多个 runtime。
业务层按 capability 查找和调用模型，不依赖 YOLO、Whisper、Llama 等具体模型类型。

初始覆盖矩阵：

| 家族 | 典型 capability 组合 | 优先 runtime | 首阶段意义 |
|---|---|---|---|
| CNN | `ImageClassification` | PyTorch MPS / Core ML | 验证基础推理与转换 |
| RNN/LSTM | `SequenceClassification`、`TextCompletion` | PyTorch MPS | 验证有状态输入 |
| YOLO | `ObjectDetection`、可选 `InstanceSegmentation` | PyTorch MPS / Core ML | 验证结构化视觉输出 |
| CLIP | `ImageEmbedding`、`TextEmbedding`、`ImageTextSimilarity` | PyTorch MPS | 验证多输入、多能力 |
| Whisper | `SpeechTranscription`、`SpeechTranslation` | PyTorch MPS / Core ML | 验证音频、时间戳与长任务 |
| 小型 LLM | `TextCompletion`、`Chat`、`StreamingGeneration` | MLX | 验证流式生成与量化 |
| 小型 VLM | `VisionLanguageChat`、`ImageUnderstanding` | MLX / PyTorch MPS | 验证多模态和较大资源压力 |

manifest 中声明的是 capability ID 和 schema version；Python 类只是 SDK 内部实现。即使两个模型都支持
`ObjectDetection`，只要遵守同一 capability contract，Demo 和评估业务就可以复用。

## Manifest

每个可运行模型需要声明：

- 稳定 `model_id`、显示名称、家族和 revision；
- 来源、文件列表、hash、总大小；
- 模型许可证、数据/使用限制与来源链接；
- capability 与输入输出 schema 版本；
- capability 的限制，例如是否支持流式、多图、batch、时间戳或 tool calling；
- 支持的 runtime、dtype、quantization；
- 推荐/最低统一内存与 macOS 要求；
- 默认预处理、后处理和 generation 配置；
- 可用评估套件与 Demo schema；
- 转换产物与原始产物之间的 provenance。

manifest 是声明，不执行任意 Python。自定义模型逻辑通过受信任 plugin/adapter 注册。

## 注册与发现

registry 提供：

- `list_models(filters)`
- `get_manifest(model_id, revision)`
- `list_capabilities(model_id)`
- `resolve_runtime(model_id, constraints)`
- `create_instance(model_id, options)`

注册来源按优先级合并：SDK 内置目录、项目配置目录、用户本地目录。冲突必须显式报错或由配置指定，
不能静默覆盖。

典型发现流程：

```text
按 capability 搜索候选模型
→ 根据 runtime、资源和许可证过滤
→ 选择 model definition
→ 创建独立 instance
→ 从 instance 获取 capability
→ 调用强类型任务方法
```

业务层不能使用 `isinstance(model, YoloModel)` 选择工作流，也不能根据模型 ID 字符串猜测功能。

## Runtime 选择

自动选择依据包括：

1. manifest 明确支持；
2. 当前机器与可选依赖可用；
3. 任务能力完整；
4. 内存预算允许；
5. 用户偏好与历史基准。

自动选择结果必须可解释，并允许用户强制指定。Core ML 更适合稳定部署/ANE 路径，MLX 更适合 Apple Silicon
上的生成式模型实验，PyTorch MPS 更适合广泛模型兼容；它们不是简单的全局优先级。

## 模型准入

新增模型在进入主目录前至少需要：

- manifest schema 校验；
- 最小推理 smoke test；
- 标准输入输出 contract test；
- load/unload 重复测试；
- 资源校验与许可证审查；
- 一份可复现的设备基线结果。

首个通过该流程的内置定义是 `ultralytics/yolov8`，其 `n/s/m` 权重作为同一模型的 variant 管理，
详见 [YOLOv8 SDK](14-yolov8-sdk.md)。
