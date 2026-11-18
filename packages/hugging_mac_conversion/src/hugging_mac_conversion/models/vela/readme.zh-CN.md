# Vela 转换

[English](readme.md)

Vela 保持一个模型目录，变体由 model.yaml 管理：3-fast 对应 Vela3-fast，3_1-fast 对应 Vela3.1-fast。full.py 负责完整模型组装，utils/components.py 负责原始神经网络图；evaluation/full.py 负责逐组件和完整响应对照。SDK 独立管理运行时、上游接口适配和 demo，两个包各自维护 Apple 导出/执行和模型工具，互不导入。

3.1 的决策主干和九个适配器全部纳入转换，不过滤权重。保留原始 FP32、LoRA 运算、输入输出结构、任务路由、阈值、字符偏移、窗口和上下文默认值。文本、batch、问题、选项、词和标签维度动态。权重分离存储并作为 Core AI 只读权重输入缓存，避免重复存储共享主干。

原始与转换模型均存入项目 tmp/，被 Git 忽略。具体命令见英文说明；当前结果见 [评估记录](evaluation/coreai-3_1-fast-2026-11-17.md)。原始权重、逐组件检查、完整响应或 SDK 用例任一未通过，都不能标记为完成或发布。
