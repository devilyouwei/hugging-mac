# 模型资源管理

状态：`Proposed`

资源管理同时覆盖磁盘资源和运行时内存，二者不可混为一个 cache。

## 磁盘资源

推荐内容寻址布局：

```text
models/
  blobs/<sha256>
  manifests/<model-id>/<revision>.json
  snapshots/<model-id>/<revision>/...
  conversions/<source-digest>/<target-runtime>/<options-digest>/...
```

下载流程：解析 → 预检磁盘 → 临时文件下载 → hash 校验 → 原子移动 → 建立 snapshot 引用。
中断下载可恢复；校验失败文件进入隔离区或删除，不可被 loader 看见。

## Resource Resolver

上层只提供 `ResourceSpec`，resolver 决定资源来自：

- 已验证本地 blob；
- Hugging Face Hub 等远端来源；
- 用户显式导入的本地文件；
- 从原模型转换得到的 Core ML/量化产物。

所有来源最终产生统一的 `ResolvedResource`：只读路径、digest、size、provenance 与 lease。

第一版 SDK 已实现以下下载来源：

| 来源 | schema | 目标 |
|---|---|---|
| Hugging Face 单文件 | `HuggingFaceSource(filename=...)` | 精确文件路径 |
| Hugging Face snapshot | `HuggingFaceSource` | 完整模型目录 |
| HTTP(S) 单文件 | `UrlFileSource` | 精确文件路径 |
| HTTP(S) ZIP/TAR | `UrlArchiveSource` | 安全解压后的模型目录 |

下载器位于 `hugging_mac_sdk.resources`。目标先写入同级唯一 staging 路径，校验成功后才提交；
单文件和归档支持来源文件 SHA-256，Hugging Face snapshot 可校验确定性的目录摘要。通用 URL 不具备
远端目录枚举协议，因此多文件 URL 模型必须提供 ZIP/TAR，或在 manifest 中声明多个单文件资源。

## 租约与清理

加载中的实例持有 lease。清理器只能删除：

- 没有活跃 lease；
- 不被 pinned manifest 引用；
- 超过保留时间或磁盘预算；
- 非进行中任务所需。

先做 dry-run 清单，再执行回收。用户导入的原始文件默认不由系统删除。

## 运行时资源

Apple Silicon 使用统一内存，不能只看“GPU 显存”。资源管理器至少采集：

- 进程 RSS 与系统可用内存；
- 模型预估权重、KV cache 和峰值工作区；
- 当前实例引用数、在途请求和最近使用时间；
- 内存压力、设备热状态（可获得时）；
- 加载、首 token、吞吐和卸载耗时。

加载前执行 admission control：

```text
estimated_peak + active_reserved + safety_margin <= usable_budget
```

估算不可靠时采用保守预算，并用真实运行数据迭代 manifest 的资源画像。

## 淘汰策略

默认优先级：

1. 引用为零且空闲最久的实例；
2. 可快速重建的小实例；
3. 未 pinned 的共享实例；
4. 最后才拒绝新请求。

训练任务和交互推理默认不竞争同一预算池。发生内存压力时，先停止接纳，再有序卸载，不依赖 OOM 后恢复。

## 安全

- 模型来源默认使用允许列表策略；
- 优先 safetensors 等非任意代码格式；
- `trust_remote_code` 默认关闭，并需要逐模型批准；
- hash 和来源信息写入运行记录；
- 解压时防止路径穿越和压缩炸弹。
