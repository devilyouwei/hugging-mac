# hugging-mac 设计文档

这里是项目设计的主页。文档按“稳定内核优先、外围能力分层”的顺序组织。

## 本地阅读

在项目根目录运行：

```bash
uv run --group docs mkdocs serve
```

浏览器打开 <http://127.0.0.1:8000/>。文档站支持全文搜索、浅色/深色主题和 Mermaid 图表；
保存 Markdown 后会自动刷新。

## 阅读路径

第一次参与项目，建议依次阅读：

1. [产品边界与原则](00-vision-and-principles.md)
2. [总体架构](01-architecture.md)
3. [模型 SDK](02-model-sdk.md)
4. [模型目录与能力](03-model-catalog.md)
5. [模型资源管理](04-resource-management.md)

按工作领域继续阅读：

| 领域 | 文档 | 解决的问题 |
|---|---|---|
| Demo | [05-demo-platform.md](05-demo-platform.md) | 如何用统一协议展示不同模型 |
| 评估 | [06-evaluation.md](06-evaluation.md) | 如何保证结果可复现、可比较 |
| 训练 | [07-training.md](07-training.md) | 如何组织微调、训练与产物 |
| 应用 | [08-frontend-backend.md](08-frontend-backend.md) | API、业务层与前端如何解耦 |
| 数据 | [09-data-and-storage.md](09-data-and-storage.md) | 本地资源、元数据和缓存如何落盘 |
| 工程 | [10-engineering.md](10-engineering.md) | 测试、配置、日志、兼容与安全 |
| 规划 | [11-roadmap.md](11-roadmap.md) | 分阶段交付顺序与验收条件 |
| 决策 | [12-architecture-decisions.md](12-architecture-decisions.md) | 如何记录影响长期架构的决定 |
| 转换 | [13-model-conversion.md](13-model-conversion.md) | 通用与模型专用转换器如何选择 |
| YOLOv8 | [14-yolov8-sdk.md](14-yolov8-sdk.md) | 首个可运行模型 SDK 的契约与使用方式 |
| YOLOv8 Pose / Seg | [19-yolov8-pose-seg-sdk.md](19-yolov8-pose-seg-sdk.md) | 姿态与实例分割模型包、输出与资源边界 |
| Audio8-ASR 0.1B | [20-audio8-asr-sdk.md](20-audio8-asr-sdk.md) | 不执行远端代码的 MPS 语音识别模型包 |
| Web | [15-web-application-architecture.md](15-web-application-architecture.md) | Index、模型状态与应用目录 |
| App 模块 | [16-app-blueprint-specification.md](16-app-blueprint-specification.md) | Blueprint 目录、注册、模型调用和错误边界 |
| Platform 后端 | [17-platform-backend.md](17-platform-backend.md) | FastAPI、TinyDB、缓存、SSE 与启动方式 |
| Vue 与首个 App | [18-web-frontend-and-object-detection.md](18-web-frontend-and-object-detection.md) | Vue 首页与 Object Detection 闭环 |

## 设计状态

- `Proposed`：当前方案，允许讨论和调整。
- `Accepted`：已确认，进入实现约束。
- `Superseded`：已被新文档或 ADR 替代。

Web 应用架构、App Blueprint 和 Platform 后端已进入 `Accepted/Implemented`；其余文档仍按各自
页首状态演进。

## 文档原则

- 文档描述边界、契约和不变量，不把某个实现细节永久固化。
- 示例 API 是设计草案，不代表已经存在的代码。
- 运行时或模型特有逻辑必须留在 adapter 内。
- 每个模块同时写清“负责什么”和“不负责什么”。
