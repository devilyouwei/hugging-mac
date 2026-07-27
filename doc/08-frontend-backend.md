# 前后端与 API

状态：`Proposed`

## 边界

前端是独立工程，只依赖版本化 API schema；后端包含传输层与业务层，但两者仍分包。
SDK 不知道 HTTP，前端不知道模型本地路径。

## API 资源

初始资源模型：

- `/models`：目录、能力、runtime 可用性；
- `/instances`：创建、状态、卸载；
- `/inference`：同步或短任务；
- `/streams`：SSE/WebSocket 流式事件；
- `/jobs`：下载、评估、训练等长任务；
- `/artifacts`：受控读取结果；
- `/system`：设备、资源预算、健康状态。

具体路径在实现时通过 OpenAPI contract 确认。

## 业务层

业务用例负责权限、会话、任务编排、幂等键和展示友好结果。它只调用 SDK facade 和 repository interface，
不 import runtime adapter。

Web Index、模型状态目录与 App 模块总览见 [Web 应用层与 Index](15-web-application-architecture.md)。
每个业务 App 的目录、blueprint、模型调用和错误规范见
[App Blueprint 模块规范](16-app-blueprint-specification.md)。

## 长任务与流

- 创建长任务立即返回 job ID；
- 状态持久化，客户端断线不取消任务；
- 明确的 cancel endpoint；
- SSE 适合单向 token/progress，WebSocket 只在双向实时交互必要时使用；
- 事件带递增 sequence，支持重连续传。

## Schema 版本

SDK schema 与外部 API schema 分开。API 层负责转换，这允许内部协议演进而不立即破坏 UI。
破坏性 API 变更使用版本路径或明确迁移期。

## 安全默认

- 默认只监听 loopback；
- 本地文件访问使用 artifact ID，不接受任意路径；
- 上传限制类型、大小和像素/解压上限；
- 日志不记录 prompt、图片或 token，除非用户显式开启；
- 开放局域网访问时必须增加鉴权和来源限制。
