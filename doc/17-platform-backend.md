# Platform Python 后端

状态：`Implemented`，首版。

## 技术决策

- FastAPI + `APIRouter`，路由由装饰器声明；
- CORS 默认允许 Vue/Vite 本地开发地址；
- multipart 文件上传；
- SSE catalog event 与 heartbeat；
- TinyDB 单进程本地 JSON 文档库，不启动额外服务；
- 内容寻址的本地文件缓存，用于上传文件、图片切片、embedding 和后续派生产物；
- runtime policy 通过环境变量动态排序。

## 当前实现目录

```text
apps/web/backend/
├── pyproject.toml
└── src/hugging_mac_web/
    ├── main.py                     # app factory、lifespan、CORS
    ├── index.py                    # App catalog 与 SSE
    ├── models.py                   # 模型 catalog 与实例
    ├── system.py                   # health 与机器信息
    ├── media.py                    # 平台级上传
    ├── cli.py
    ├── app_blueprint.py
    ├── app_registry.py
    ├── shared/
    │   ├── cache/                  # 内容寻址文件缓存
    │   ├── storage/                # TinyDB 文档存储
    │   └── utils/                  # system/log/time/media/upload/SSE
    └── object_detection/
        ├── manifest.py
        ├── blueprint.py
        ├── routes.py
        ├── service.py
        ├── schemas.py
        └── config.py
```

现有 `AppBlueprint` 契约提供 typed manifest 和 `APIRouter` 注册。import blueprint 不得下载模型、
打开数据库或创建实例。

## API

| Method | Path | 作用 |
|---|---|---|
| GET | `/api/v1/catalog/models` | 模型、runtime 可用性与实例明细 |
| POST | `/api/v1/catalog/models/{model_id}/instances` | 按 runtime/device 加载独立模型实例 |
| DELETE | `/api/v1/catalog/instances/{instance_id}` | 卸载指定实例并返回耗时与 RSS 释放指标 |
| GET | `/api/v1/catalog/models/{model_id}/resources` | artifact、runtime 与模型总大小 |
| POST | `/api/v1/catalog/models/{model_id}/resources/download` | 覆盖下载源权重 |
| POST | `/api/v1/catalog/models/{model_id}/resources/convert` | 按 `target_format` 覆盖转换目标产物 |
| DELETE | `/api/v1/catalog/models/{model_id}/resources` | 删除该 revision 的本地权重目录 |
| GET | `/api/v1/catalog/apps` | 已注册业务 App |
| GET | `/api/v1/catalog/events` | SSE catalog 初始快照与 heartbeat |
| GET | `/api/v1/system/health` | 存活状态 |
| GET | `/api/v1/system/info` | OS、CPU、内存与 Python 信息 |
| POST | `/api/v1/media/uploads` | 图片/视频/音频上传到本地缓存 |
| GET | `/api/v1/apps/object-detection/resources` | 查看 YOLOv8 本地资产状态 |
| POST | `/api/v1/apps/object-detection/resources/source/download` | 显式下载源模型 |
| POST | `/api/v1/apps/object-detection/resources/coreml/convert` | 显式转换 Core ML |
| POST | `/api/v1/apps/object-detection/detect` | YOLOv8 图片目标检测 |
| POST | `/api/v1/apps/object-detection/detect/frame` | 低开销实时帧目标检测 |

模型 catalog 是轻量只读查询，不会下载、转换或加载模型。前端首页分别并行请求 models/apps。
加载接口接收 `runtime`、可选 `device` 和 `warmup`；模型文件缺失、runtime 不支持或加载失败时直接返回
SDK 的稳定错误。实例仍被业务请求 retain 时，卸载接口拒绝操作，避免中断正在执行的推理。

模型 manifest 同时暴露 `variants` 和 `default_variant`。加载请求可传 `variant`；资源查询、下载和删除通过
query 选择 variant，转换请求在 JSON 中携带 variant。不传时使用模型默认值。实例 snapshot、卸载结果和
资源状态都会回显最终选择的 variant。

实例 snapshot 包含首次加载的耗时、进程 RSS 前后值和 RSS 分配差值；卸载响应包含卸载耗时与 RSS
释放差值。它们是本地 Python 进程的观测指标，不代表模型独占内存或完整的 GPU/ANE 内存。

覆盖下载、转换和删除属于资源变更操作。只要该模型仍有受管实例，平台就返回 `409 Conflict`，要求先卸载实例。
删除只移除 SDK 声明的本地权重目录，不删除 manifest、模型代码或 registry definition。删除接口幂等，
资源不存在时仍返回当前的空资源状态。

转换接口使用统一请求体，例如 `{"target_format": "coreml"}`。目标格式由 SDK
`ModelResourceStatus.conversion_targets` 暴露，前端据此显示 `Convert to Core ML` 或
`Re-convert to Core ML`；以后模型注册 ONNX、MLX 等目标后无需增加专用平台路由。

## 终端日志

业务日志通过 `shared/utils/log_util.py` 统一配置：

- 自动输出调用源码的 `pathname`、`lineno` 和 `func_name`；
- `logger.exception()` 输出完整 Python traceback；
- INFO 使用终端默认颜色；
- DEBUG 青色、WARNING 黄色、ERROR 红色、CRITICAL 加粗红色；
- 非 TTY 环境默认关闭 ANSI 颜色，也可由嵌入方显式覆盖。

## 本地数据

TinyDB 只保存文档元数据；二进制媒体和派生产物保存在 cache。缓存路径使用
`namespace/sha256-prefix/sha256.suffix`，相同内容会复用文件。API 永远不返回本机绝对路径。

默认使用 macOS 的用户数据与缓存目录。开发时可通过 `.env` 覆盖：

```dotenv
HUGGING_MAC_WEB_DATA_DIR=./data/web
HUGGING_MAC_WEB_CACHE_DIR=./data/cache
HUGGING_MAC_WEB_DATABASE_PATH=./data/web/platform.json
HUGGING_MAC_WEB_RUNTIME_PREFERENCE=coreml,pytorch-mps
```

## 启动与验证

```bash
uv sync --all-packages --extra yolo
uv run hugging-mac-web
```

OpenAPI UI 位于 <http://127.0.0.1:8000/docs>。测试命令：

```bash
uv run pytest
uv run ruff check .
uv run mypy apps/web/backend/src packages/hugging_mac_sdk/src
```
