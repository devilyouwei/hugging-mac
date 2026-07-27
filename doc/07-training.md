# 训练体系

状态：`Proposed`

## 边界

训练服务负责任务编排、数据、checkpoint、指标与导出。模型 SDK 提供可训练模型/adapter 的构建契约，
但不把完整训练循环塞进推理实例。

优先支持 Apple Silicon 可实际完成的任务：

- 小型 CNN/RNN 从头训练；
- 视觉模型微调；
- LLM/VLM 的参数高效微调；
- 量化和 Core ML 转换验证。

## 训练契约

训练 recipe 声明：

- base model snapshot；
- dataset snapshot 与 transform；
- trainer/optimizer/scheduler；
- precision、batch、gradient accumulation；
- seed、checkpoint、early stopping；
- 资源预算和输出策略。

recipe 配置需可序列化、可 hash。不可序列化的 callable 通过已注册名称引用。

## Worker 隔离

训练默认在独立 worker 中运行，原因是运行时间长、内存峰值高且取消语义不同。训练 worker 与 Demo worker
使用不同资源队列；用户可显式允许共享，但系统不默认超售统一内存。

## Checkpoint 与恢复

- checkpoint 写临时目录后原子提交；
- 保存 recipe hash、代码 revision 和数据摘要；
- 恢复时校验兼容性，禁止静默套用不匹配状态；
- 保留策略按 latest/best/pinned，清理前检查 lease；
- 导出的推理 artifact 生成新的 manifest revision，并记录 provenance。

## 训练到 SDK

训练完成不自动上线。必须经过：

1. artifact 完整性检查；
2. SDK contract smoke test；
3. 代表性样本评估；
4. 资源画像；
5. 注册为候选 revision；
6. 用户批准后才成为 Demo 默认项。
