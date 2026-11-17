# 完整 Vela Core AI

[English](README.md)

本制品保留决策主干和所有任务适配器，包括 choice/noul、router choice/set、普通 span、实体、关系、PII、幻觉片段与证据提取。沿用上游的 state/questions 结构、路由、分词、阈值、窗口聚合、字符偏移、评分、错误和 usage；没有通过删除功能简化转换。

所有神经网络计算由 Core AI 执行。原始 FP32 权重在 weights/ 中保存一次，缓存为 Core AI 的只读权重输入；LoRA 不合并、不量化。运行时按需切换组件，避免同时驻留全部权重。上游 Python 代码会导入 Torch，但原生运行路径不调用 PyTorch 神经网络。

batch、文本长度、问题数、选项数、词单元数、标签数均为动态维度。保留原始 32768 token 决策上限、8192 token schema 窗口及 512 token 重叠、32768 token 证据窗口。最大长度是否实测以评估报告为准；导出符号上限不代表任意设备都能在内存中运行该长度。

SDK 调用示例和 uv 命令见 [英文说明](README.md#python-sdk)。实际验收状态见 build.json、evaluation.json；仅导出计算图不等于通过模型评估。运行时必须包含所有 graphs/ 和 weights/，不能只下载决策图。
