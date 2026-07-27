# 工程规范

状态：`Proposed`

## Python 与包管理

- Python `>=3.12`；
- uv 管理解释器、依赖与 lockfile；
- 根项目承载工作区和统一工具配置；
- 模型 SDK 最终拥有独立包元数据和最小依赖；
- 大型/runtime 特有依赖放入可选 extra，并使用平台 marker。

根 `pyproject.toml` 使用 `tool.uv.package = false`，只负责 workspace、依赖组和统一工具配置。
SDK 已作为 uv workspace member 存放在 `packages/hugging_mac_sdk`，并由自己的 `pyproject.toml`
声明可独立构建的 `hugging-mac-sdk` 包。

## 配置

优先级：内置默认值 < 配置文件 < 环境变量 < CLI/API 显式参数。配置用 typed schema 校验；
密钥只来自环境变量或系统凭据存储，不进入配置文件和运行报告。

## 日志与指标

- 结构化日志包含 timestamp、level、component、trace/job/instance ID；
- 错误保留异常链；
- 模型输入输出默认不入日志；
- 性能指标区分 queue/load/preprocess/infer/postprocess/transport；
- 不默认发送外部遥测。

## 测试金字塔

- 单元测试：schema、状态机、选择策略；
- contract test：每个 model/runtime adapter 必须通过；
- 集成测试：下载 → load → infer → unload；
- golden test：容忍跨 runtime 的合理数值差异；
- 故障测试：下载中断、hash 错误、取消、worker 崩溃、低内存；
- 性能基线：只做趋势告警，不把不同硬件的绝对值混用。

## CI

普通 CI 可在非 Mac 环境运行纯 Python 测试；MPS、MLX、Core ML 测试需要 Apple Silicon runner 并单独标记。
网络模型测试不进入默认快速测试集，使用已校验的小 fixture。

## 兼容矩阵

每次发布记录：

- macOS 主版本；
- M 芯片家族与统一内存；
- Python 和 lockfile 摘要；
- runtime 版本；
- 已验证模型/能力。

依赖升级先跑最小兼容矩阵，再更新 lockfile。TensorFlow/Metal 等支持节奏不同的 runtime 不进入默认依赖组，
待建立独立验证 lane 后加入。

## 代码质量

计划使用 Ruff、mypy、pytest 和 pre-commit。公共 API 强类型；跨层 schema 禁止无约束 `dict[str, Any]`。
注释解释原因和约束，不复述代码。
