# Demo 平台

状态：`Proposed`

## 目标

Demo 是可复用 SDK 能力的交互呈现，不是模型脚本集合。每个 Demo 由 declarative schema 描述输入组件、
输出 renderer、模型 capability 与默认配置。

## Demo 类型

- 单次：图像分类、目标检测、embedding 相似度；
- 会话：LLM/VLM 多轮生成；
- 对比：同一输入运行不同模型/runtime/量化；
- 诊断：显示预处理、token、timing 和资源变化。

## 会话模型

业务层拥有会话和用户状态；模型实例只拥有推理必要状态。上下文窗口、图片引用和生成参数由业务层组装为
SDK request，避免把会话数据库耦合进模型包。

流式事件统一为：

- `started`
- `delta`
- `progress`
- `artifact`
- `metrics`
- `completed`
- `error`

LLM token、训练进度和模型下载可共享事件外壳，但 payload schema 分开版本化。

## 对比模式

对比必须固定输入、预处理版本和随机种子，并显示：

- runtime/dtype/quantization；
- 预热与非预热延迟；
- 峰值内存；
- 输出差异；
- 不能直接比较时的原因。

## 可观测性

每次运行产生 trace ID。UI 可展示总耗时，也能展开资源解析、排队、加载、预处理、推理、后处理和传输耗时。
默认不展示虚假的单一“模型速度”数字。
