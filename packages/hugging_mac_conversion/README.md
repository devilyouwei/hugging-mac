# Hugging Mac Conversion

独立的模型转换、评测和 Hugging Face 发布包，与 `hugging_mac_sdk` 平级。
本包拥有自己的源码、模型配置、用例、CLI、测试、依赖锁和 uv 环境。
依赖方向为 conversion → SDK：复用 SDK 的 Core AI API 与运行时；SDK 不依赖或打包本包。

从仓库根目录执行：

```bash
uv sync --project packages/hugging_mac_conversion --locked
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert list
```

本包不加入主 uv workspace 的统一锁定：应用模型 extras 固定 Torch 2.7，
转换工具使用 Torch 2.14。独立 `uv.lock` 和 `.venv` 隔离这两组依赖。
SDK 通过相邻目录的 editable path dependency 安装，不启用应用模型 extras。

原始模型、转换制品、评测报告分别保存在项目根目录下 Git 忽略的
`tmp/originals`、`tmp/conversions`、`tmp/evaluations`，不进入任一 Python 包。

源码分工、变体配置、完整能力要求、转换与评测命令见
[转换工作区说明](src/hugging_mac_conversion/README.md)。

开发检查：

```bash
cd packages/hugging_mac_conversion
uv run --locked pytest
uv run --locked ruff check src tests
uv run --locked mypy src/hugging_mac_conversion
uv build --wheel
```
