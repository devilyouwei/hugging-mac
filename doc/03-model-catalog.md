# 模型目录与能力

状态：`内置模型包、显式注册与资源管理已实现；外部目录自动发现与合并仍在规划`

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
| Audio8-ASR | `SpeechTranscription` | PyTorch MPS | 验证短音频、自回归 ASR 与多文件资源 |
| Audio8-TTS | `SpeechSynthesis` | PyTorch MPS | 验证多语言、自回归语音生成与参考音频克隆 |
| Kokoro-82M | `SpeechSynthesis` | PyTorch MPS | 验证轻量本地语音合成与包内 voice 资源 |
| 小型 LLM | `TextCompletion`、`Chat`、`StreamingGeneration` | MLX | 验证流式生成与量化 |
| 小型 VLM | `VisionLanguageChat`、`ImageUnderstanding` | MLX / PyTorch MPS | 验证多模态和较大资源压力 |

当前 manifest 以 capability 字符串声明能力；强类型 request/response contract 由 Python Protocol 和
schema 表达，尚未在 manifest 中逐项声明 schema version。即使两个模型都支持 `ObjectDetection`，只要
遵守同一 capability contract，Demo 和评估业务就可以复用。

## Manifest

每个可运行模型需要声明：

- 稳定 `model_id`、显示名称、家族和 revision；
- 来源、文件列表、hash、总大小；
- 模型许可证、数据/使用限制与来源链接；
- capability；输入输出 schema version 是未来 manifest 扩展项；
- capability 的限制，例如是否支持流式、多图、batch、时间戳或 tool calling；
- 支持的 runtime、dtype、quantization；
- 推荐/最低统一内存与 macOS 要求；
- 默认预处理、后处理和 generation 配置；
- 可用评估套件与 Demo schema；
- 转换产物与原始产物之间的 provenance。

manifest 是声明，不执行任意 Python。自定义模型逻辑通过受信任 plugin/adapter 注册。

## 注册与发现

当前 `ModelRegistry` 提供：

- `list_models(capability=..., runtime=...)`
- `get(model_id, revision)`，返回 `ModelDefinition` 与其 manifest；
- `supported_runtimes(model_id, revision)`；
- `get_artifact(...)`、`get_artifacts(...)` 与 `resolve_artifact_path(...)`；
- `create_instance(model_id, options)`

runtime 的 `auto` 选择不在 registry 内，而在 `RuntimePolicy` / `InstanceManager` 内完成。

当前注册由 Python 代码显式调用 `register_*()` 完成；同一 `(model_id, revision)` 重复注册会报冲突，除非
明确传入 `replace=True`。SDK 内置目录、项目配置目录、用户本地目录的自动发现与优先级合并尚未实现。

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

当前自动选择依据：

1. manifest 明确支持；
2. 当前机器与可选依赖可用；
3. `RuntimePolicy` 的有序偏好（通常来自平台环境变量）；
4. manifest 的 `default_runtime`；
5. 声明顺序。

内存预算、任务级完整性和历史基准尚未参与自动选择，属于后续 runtime policy 的扩展项。

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

当前内置定义包括：

- `ultralytics/yolov8` 及 Pose/Seg 模型包，详见 [YOLOv8 SDK](14-yolov8-sdk.md)；
- `audio8/audio8-asr-0.1b`，以 `base` variant 和 `SpeechTranscription` 接入，详见
  [Audio8-ASR SDK](20-audio8-asr-sdk.md)。
- `audio8/audio8-tts-preview-0.6b`，以 `preview` variant 和 `SpeechSynthesis` 接入；
- `hexgrad/kokoro-82m`，以 `v1.0` variant 和 `SpeechSynthesis` 接入；
  两者详见 [TTS SDK 与应用](22-text-to-speech-sdk.md)。
