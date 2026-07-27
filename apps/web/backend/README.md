# hugging-mac Web backend

本地 FastAPI 平台后端。它通过 workspace 中的 `hugging-mac-sdk` 暴露模型/App catalog、系统信息、
SSE 与媒体上传，并用 TinyDB + 本地内容寻址缓存持久化元数据和文件。内置的首个业务 App 是
`object_detection`，接口为 `POST /api/v1/apps/object-detection/detect`。

```bash
uv sync --all-packages --extra yolo
uv run hugging-mac-web
```

配置项使用 `HUGGING_MAC_WEB_*` 环境变量，完整说明见
[`doc/17-platform-backend.md`](../../../doc/17-platform-backend.md)。
