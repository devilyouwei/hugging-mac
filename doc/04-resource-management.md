# 模型资源管理

状态：`下载与模型级资源操作已实现；租约、配额与回收仍在规划`

资源管理同时覆盖磁盘资源和运行时内存，二者不可混为一个 cache。

## 磁盘资源

目标中的内容寻址布局：

```text
models/
  blobs/<sha256>
  manifests/<model-id>/<revision>.json
  snapshots/<model-id>/<revision>/...
  conversions/<source-digest>/<target-runtime>/<options-digest>/...
```

当前实现采用模型 package 决定的受管目录，例如
`<model_home>/ultralytics/yolov8-seg/v8.2.0/n/source/...` 与 `coreml/...`。下载流程为目标相邻的
staging 路径 → 下载/解压 → digest 校验 → 原子替换目标；失败 staging 会被清理，loader 不会看到半成品。
内容寻址 blob、断点续传、隔离区与 snapshot 引用尚未实现。

## Resource Resolver

上层只提供 `ResourceSpec`，resolver 决定资源来自：

- 已验证本地 blob；
- Hugging Face Hub 等远端来源；
- 用户显式导入的本地文件；
- 从原模型转换得到的 Core ML/量化产物。

所有来源最终产生统一的 `ResolvedResource`：路径、digest、size 与来源声明。当前没有 lease 字段或全局
资源索引；资源状态由各模型的 `ModelResourceProvider` 汇总。

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

## 共享 artifact

不同 variant、runtime 或量化版本共用的 tokenizer、processor、词表等资源，可以在模型 YAML 的
artifact 上声明 `shared: true`：

```yaml
artifacts:
  - artifact_id: tokenizer
    variant: 0.6b       # 仅作为目录清单元数据，不限制共享范围
    runtime: coreml     # 仅作为目录清单元数据，不限制共享范围
    format: tokenizer
    path: qwen/qwen3-asr/shared/tokenizer/<tokenizer-revision>
    kind: directory
    shared: true
    source:
      kind: huggingface
      repo_id: Qwen/Qwen3-ASR-0.6B
      revision: <pinned-revision>
      allow_patterns: [vocab.json, merges.txt, tokenizer_config.json]
```

共享 artifact 的作用域是整个 `model_id@revision`，不会被声明中的 variant/runtime 限制。它必须具有
可直接下载的 `source`，不能是 conversion target。下载任意 scoped artifact 或执行 conversion 之前，
`ModelResourceService` 都会先检查全部共享 artifact，只下载缺失项。共享项会参与每个 runtime 的 Ready
判定，但本地空间只统计一次；删除单个 runtime/variant 不会自动删除共享项。模型实例加载仍然坚持
“只解析本地资源、不隐式联网下载”的原则。

若第三方 loader 强制要求权重与 tokenizer 位于同一目录，runtime 会在加载期间建立只含符号链接的临时
合并视图；视图退出后立即清理，不会把共享文件复制回每个 variant。当前 Qwen3-ASR、Qwen3.5、
Audio8-ASR、Audio8-TTS 与 SenseVoice 已使用模型级共享 tokenizer artifact。Qwen3-TTS 的 tokenizer
是 MLX-scoped artifact；Core ML bundle 自带独立 `vocab.json` 和 `merges.txt`。processor、
chat template、generation config 和 speech/model 权重只有在远端固定版本逐文件一致时才允许共享。
旧版 scoped artifact 若仍含同名 tokenizer，临时视图会让 shared 文件优先，但不会改写或删除旧目录，
因此已下载的大模型权重可以原地继续使用。

## 租约与清理

后续加载中的实例将持有 lease。清理器应只删除：

- 没有活跃 lease；
- 不被 pinned manifest 引用；
- 超过保留时间或磁盘预算；
- 非进行中任务所需。

当前删除操作由 `sdk.resources.delete()` 显式触发，并且只允许删除模型 resource provider 所管理目录内的
source、Core ML artifact 或 variant 根目录；自动回收、dry-run、pin 与用户导入策略尚未实现。

## 运行时资源

Apple Silicon 使用统一内存，不能只看“GPU 显存”。当前 `InstanceManager` 在 load/unload 前后记录进程 RSS
与耗时；以下是后续资源管理器应采集的完整集合：

- 进程 RSS 与系统可用内存；
- 模型预估权重、KV cache 和峰值工作区；
- 当前实例引用数、在途请求和最近使用时间；
- 内存压力、设备热状态（可获得时）；
- 加载、首 token、吞吐和卸载耗时。

后续加载前应执行 admission control：

```text
estimated_peak + active_reserved + safety_margin <= usable_budget
```

估算不可靠时采用保守预算，并用真实运行数据迭代 manifest 的资源画像。

## 淘汰策略

后续自动淘汰的默认优先级：

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
