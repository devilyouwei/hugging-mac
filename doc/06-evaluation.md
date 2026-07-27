# 评估体系

状态：`Proposed`

## 结构

一次评估由以下不可变描述组成：

- model snapshot；
- runtime 与实例配置；
- dataset snapshot 与 split；
- evaluator/metric 版本；
- 预处理和后处理版本；
- seed、硬件、OS、Python 与依赖锁摘要。

## 评估类型

- 正确性：accuracy、F1、mAP、Recall@K、困惑度或任务专用指标；
- 性能：冷启动、预热延迟、吞吐、首 token、token/s；
- 资源：磁盘、峰值统一内存、能耗/热状态（可可靠采集时）；
- 稳定性：长时间运行、反复 load/unload、取消与异常恢复；
- 质量回归：固定 golden samples 和容差。

## Runner 与 Metric 解耦

runner 负责遍历数据、调用 SDK、保存预测和恢复进度；metric 只消费标准预测与标签。
原始预测应可复算指标，避免为了换指标重新运行昂贵模型。

## 公平比较

- 冷启动和 warm run 分开；
- 不同量化明确标注，不能只比较速度；
- 批量大小、上下文长度和生成上限固定；
- 至少报告分位数和样本数，而非只有平均值；
- 机器处于明显热节流时标记结果；
- Core ML、MLX 与 MPS 的不同数值行为使用任务级容差。

## 产物

```text
artifacts/evaluations/<run-id>/
  run.json
  predictions/
  metrics.json
  environment.json
  report/
```

run ID 不包含用户隐私。报告可提交摘要，小规模 golden fixtures 可入库，大体积预测默认不提交。
