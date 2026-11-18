---
language:
- en
- zh
license: apache-2.0
base_model: vllm-sr/vela3-fast
pipeline_tag: zero-shot-classification
tags:
- coreai
- apple-silicon
- aimodel
- experimental
- decision-model
---

# Vela3-fast Core AI — 实验性决策图

[English](README.md) | 简体中文

由 hugging-mac 从 [vllm-sr/vela3-fast](https://huggingface.co/vllm-sr/vela3-fast) 转换的 Apple Core AI 模型。
本仓库发布的是经过验证的 **choice / noul 决策主干**：每次图调用一个样本，序列长度和选项数量均为动态。
导出保留原始 **32768-token** 上限；CPU 实际对照验证覆盖到 **2048 tokens**，包括精确 1024 tokens。
**这是实验性子集，不是包含所有任务 adapter 的完整 Vela3-fast。**

This is an experimental, fixed-shape decision-backbone export, not a complete replacement for upstream Vela3-fast.
It runs neural inference through Core AI with host-side tokenization and response formatting. It does not generate text.

## 支持范围 / Contract

- `choice`：一个或多个有序选项，返回选项概率和选择结果。
- `noul`：固定 `no` / `yes` 两个选项，返回 `p(yes)`。
- 输入正文、问题、选项描述和特殊标记合计不超过 原始 **32768 tokens** 上限；超长输入报错，不截断。
- 每次图调用的 batch size 为 1。`predict_questions(state, questions)` 接受问题映射并返回 `(answers, input_tokens)`，与上游决策接口对应；内部逐个执行。保留原始 JSON 序列化，choice criteria 支持字典或选项列表。
- CPU 模式已验证。自动设备模式此前因测试主机资源压力未完成验证，不宣称 GPU/ANE 性能。
- 原始 FP32 决策权重，无量化、训练或微调。
- 原模型的 router、span、relation、NER、evidence、domain/factcheck/feedback 等 adapter 不包含在此制品中；
  当前决策图未实现完整 `system_one` adapter 路由。未测试不代表这些能力无法转换。
- 不提供完整 `hugging_mac_sdk` 模型注册或生产验收结论。概率是给定选项间的相对值，不是校准置信度。

## Python 基本调用

验证环境：Apple Silicon、macOS 27.2.1、Python 3.12.13、`coreai-core==1.0.0b3`。
需要支持 Core AI 系统运行时的 macOS。示例环境使用 uv 管理，**不需要 PyTorch、transformers 或原始 safetensors 权重**。

```bash
uvx --from huggingface-hub hf download hugging-mac/vela3-fast-CoreAI --local-dir vela3-fast-CoreAI
cd vela3-fast-CoreAI
uv sync --locked
USE_OS_COREAI=1 uv run --locked python example.py
```

```python
import asyncio
from vela_coreai import VelaCoreAI


async def main():
    # root 指向下载后的仓库目录；适配器显式使用已验证的 CPU 模式。
    async with VelaCoreAI.open(".") as model:
        result = await model.predict(
            "请写一个 Python 排序函数。",
            {
                "type": "choice",
                "instructions": "What is the topic?",
                "criteria": {"coding": "Software", "cooking": "Food"},
            },
        )
        print(result)  # type, choice, probabilities

        yes_no = await model.predict(
            "Python code",
            {"type": "noul", "instructions": "Is this about software?"},
        )
        print(yes_no)  # {"type": "noul", "noul": p_yes}


asyncio.run(main())
```

`predict` 执行 tokenizer、标记位置和 pooling mask 的准备，再调用 Core AI。
同一个实例一次只接收一个推理请求；请等待前一次调用完成。未知任务、空选项集合和超长输入都会报错。

运行独立包验证：

```bash
USE_OS_COREAI=1 uv run --locked python smoke.py
```

该脚本比较12 个用例与上游参考概率，校验输入张量与上游编码完全相同，验证三类无效请求拒绝行为，
并生成 `package_smoke.json` 和用于 Swift 的 `swift_fixture.json`。

## Swift 基本调用

需要 Xcode 27 / macOS 27 SDK 或更新版本。本仓库的 `verify_coreai.swift` 是完整、可编译的示例，
从 `swift_fixture.json` 加载一组预分词输入，用原生 `CoreAI` 执行，并与 Python 概率比较。
**Swift 示例演示原生模型调用，不包含 Swift tokenizer；它不是任意文本输入的完整 Swift SDK。**

```bash
# 在模型目录中执行；如 Xcode 安装位置不同，请调整 DEVELOPER_DIR。
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swiftc -parse-as-library -target arm64-apple-macos27.0 \
  verify_coreai.swift -o /tmp/verify-vela
/tmp/verify-vela "$PWD"
```

核心调用如下，完整张量构造见 `verify_coreai.swift`：

```swift
import CoreAI
import Foundation

let model = try await AIModel(
    contentsOf: modelDirectory.appendingPathComponent("model.aimodel"),
    options: .cpuOnly
)
guard let function = try model.loadFunction(named: "main") else {
    fatalError("Missing main function")
}
// inputs: [String: NDArray]，按下表的 dtype、shape 和编码规则构造。
var result = try await function.run(inputs: inputs)
guard let logits = result.remove("logits")?.ndArray else {
    fatalError("Missing logits")
}
// 对输出 logits 计算稳定 softmax，按输入选项顺序解释概率。
```

对任意新文本调用时，必须复用本仓库 `tokenizer.json`、`decision.json` 和 `vela_coreai.py` 的编码规则；
只对正文单独分词无法构造正确输入。随包 fixture 是人工示例，不包含用户数据。

## 张量接口

入口：`main`，无持久 recurrent/KV 状态。序列轴和选项轴均为动态，按实际编码长度输入，不固定补齐。

| 名称 | 类型 | 形状 | 含义 |
| --- | --- | --- | --- |
| `input_ids` | int32 | `[1,sequence]` | 完整问题/选项/正文序列 |
| `attention_mask` | int32 | `[1,sequence]` | 有效 token 为 1，padding 为 0 |
| `option_positions` | int32 | `[1,options]` | 各个 `[O]` 标记的位置 |
| `pool_mask` | float32 | `[1,sequence]` | `[SEG_user]` 到正文末尾为 1，不包括 EOS |
| `logits`（输出） | float32 | `[1,options]` | 对应一个或多个有序选项的未归一化分数 |

上游序列格式：

```text
[BOS] [Q] instructions [O] key_1: description_1 [O] key_2: description_2
[SEP_SCHEMA] [SEG_user] state [EOS]
```

## 验证记录与限制

- 对上游原始模型的 **12/12** 文本用例，转换后决策全部一致，预期答案全部正确。
- 最大概率绝对误差约 **7.80e-7**；验证容差为 `atol=0.02, rtol=0.02`，同时要求决策一致。
- 独立 Python 包在未安装 PyTorch 的环境中通过12 个用例，输入张量逐项匹配上游。
- Swift CPU 原生调用结果见 `swift_smoke.json`；Python 独立包记录见 `package_smoke.json`。
- 这些手工样例不是通用准确率基准；覆盖 JSON 对象、1/2/3 个选项，以及精确 129/1024/2048-token 输入。
- 未验证 iPhone/iPad、iOS AOT、自动设备模式、GPU/ANE 放置、超过 2048 tokens 的输入或其他上游任务。
- 新动态制品通过了初始 CPU 张量验证和后续任务对照评测。工具版本见 `manifest.json` 与 `environment.json`。
- 导出和输入适配器保留 32K 上限，但尚未在这台 8 GB 测试机器上验证完整 32K 推理；更长输入需要更多内存。2048 是已测长度，不是新增硬限制。
- 本版本替换先前固定 128-token、两个选项的制品；上述 adapters 仍不在当前决策图中。

完整用例与概率：`validation_cases.json`、`evaluation.json`。
`SHA256SUMS` 覆盖发布文件；本仓库不上传原始权重副本、开发缓存或虚拟环境。

## 来源与许可证

上游：[vllm-sr/vela3-fast](https://huggingface.co/vllm-sr/vela3-fast/tree/fa60e3aa41132c60bf05a84cabfdced39cb6d4f7)，
固定 commit `fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`，Apache-2.0。
转换图与输入渲染改编自该版本的 `vela3_encoder.py` 和 `ub_d3_common.py`；归属与修改说明见 `NOTICE`，许可证见 `LICENSE`。
这是 hugging-mac 的社区转换，不是 Apple 或上游团队的官方发布。
