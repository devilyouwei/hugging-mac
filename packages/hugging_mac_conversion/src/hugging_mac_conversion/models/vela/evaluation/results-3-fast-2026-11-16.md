# Vela3-fast 转换与评测记录 — 2026-11-16

本轮 Core ML 和 Core AI（显式 CPU）决策主干均通过 4/4 用例，原始与转换后答案均正确、决策一致。
Core ML 最大概率误差 8.24693e-7；Core AI CPU 最大概率误差 5.5666753e-07。张量 smoke 最大绝对误差为 1.66893e-6（atol=rtol=0.02）。
这仍是固定 batch=1、sequence=128、两个选项的诊断图，**不是完整 Vela 模型验收**。

原始完整快照（包括全部任务 adapter）已经保存到项目 `tmp/originals`，36 个文件，
来源 commit 为 fa60e3aa41132c60bf05a84cabfdced39cb6d4f7。全部原始文件与制品均被 Git 忽略。

## Core AI 状态

`tmp/conversions/vela/3-fast/coreai/20261116T131004Z-490d417e/model.aimodel` 已导出。
默认 Core AI runtime 在加载/校验阶段产生明显内存与交换空间压力；本机为 8 GiB 内存，
可用磁盘一度降到 2.8 GiB，已停止该进程并将 build.json 标记 interrupted / ResourceLimit。
自动设备模式未取得数值验证结果，不记为通过，也不推断模型无法转换。
随后使用 `USE_OS_COREAI=1` 和 `--coreai-device cpu` 对同一制品完成任务评测，4/4 通过。
该结果只适用于本轮 CPU 任务用例；原始 build.json 的自动设备中止状态保留。
首次网络请求的 503 重试记录另保存在临时构建目录。

## 环境与边界

- Mac17,5，arm64，8 GiB RAM；macOS 27.2.1 (26B131)，Python 3.12.13。
- Core ML 使用 CPU_ONLY、Float32、macOS15 deployment target。
- coremltools 9.0；coreai-torch 0.4.2；coreai-core 1.0.0b2；Torch 2.13.0。
- coremltools 对 Torch 2.13 给出未正式测试的版本警告；以下是本机实测结果。
- 未删除上游能力。router、span、relation、ner、evidence、任务 adapter、变长输入、
  多问题和完整 system_one 返回仍需转换与评测，尚未注册为 SDK 可用模型或发布到 Hub。
- 4 个手工用例仅检验本轮行为一致性，不能代表通用任务准确率或性能基准。

## 复现

在项目根目录运行（先按同一 commit 完整 fetch）：

```bash
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert probe vela --variant 3-fast --runtime coreml --revision fa60e3aa41132c60bf05a84cabfdced39cb6d4f7 --offline
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert evaluate vela --variant 3-fast --run tmp/conversions/vela/3-fast/coreml/20261116T130722Z-de00354f
```

## 本轮原始报告

JSON/Markdown 位于 `tmp/evaluations/vela/3-fast/20261116T132116Z-0d6096d6/`。
以下为同一 JSON 生成的逐项证据。

# vela / 3-fast evaluation

- Status: **diagnostic-passed**
- Scope: `diagnostic`; this is not complete-model or SDK acceptance.
- Source commit: `fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`
- Original files: `/Users/devil/Pro/hugging-mac/tmp/originals/vela/3-fast/fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`
- Source selections: `cached-partial, full, probe`
- Cases SHA-256: `c9152d838e5a29b02ea5b9bc1753350b0e8f9dec34950bd2bd539acb496db5c0`

## Original model

| Case | Capability | Status | Decision | Expected | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | choice | evaluated | coding | coding | True | 88.90737500041723 |
| cooking-zh | choice | evaluated | cooking | cooking | True | 103.32929099968169 |
| noul-positive | noul | evaluated | yes | yes | True | 65.44312500045635 |
| noul-negative | noul | evaluated | no | no | True | 71.8966250133235 |

## Converted models

Probability tolerances: atol=0.02, rtol=0.02. Require decision agreement: True.
Parity and expected-answer correctness are separate measures. Timings are single observations, not a performance benchmark.

### coreml: diagnostic-passed

- Bundle: `/Users/devil/Pro/hugging-mac/tmp/conversions/vela/3-fast/coreml/20261116T130722Z-de00354f`
- Build status: `diagnostic-validated`

| Case | Status | Max probability error | Decision | Agreement | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | passed | 4.5924924128339095e-09 | coding | True | True | 2823.6663329880685 |
| cooking-zh | passed | 8.246929867894437e-07 | cooking | True | True | 304.8574159911368 |
| noul-positive | passed | 7.820040781592041e-08 | yes | True | True | 147.63262499764096 |
| noul-negative | passed | 1.3256583497600616e-08 | no | True | True | 153.17679100553505 |

## Capability coverage

| Capability | Status |
| --- | --- |
| choice | exercised-in-diagnostic |
| noul | exercised-in-diagnostic |
| router_choice | untested |
| router_set | untested |
| span | untested |
| relation | untested |
| ner | untested |
| span_sets | untested |
| evidence | untested |
| router_task_adapters | untested |
| structured_state | untested |
| multiple_questions | untested |
| variable_sequence_length | untested |
| variable_options | untested |
| system_one_response | untested |
| domain_adapter | untested |
| factcheck_adapter | untested |
| feedback_adapter | untested |
| feedback_df_adapter | untested |

Untested capabilities remain required; untested does not mean unsupported.

## Tool versions

- torch: `2.13.0`
- numpy: `2.3.5`
- transformers: `5.17.0`
- safetensors: `0.8.0`

## Core AI CPU 复现与逐例报告

```bash
USE_OS_COREAI=1 uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert evaluate vela --variant 3-fast --coreai-device cpu --run tmp/conversions/vela/3-fast/coreai/20261116T131004Z-490d417e
```

JSON/Markdown：`tmp/evaluations/vela/3-fast/20261116T133836Z-42767dd2/`。

# vela / 3-fast evaluation

- Status: **diagnostic-passed**
- Scope: `diagnostic`; this is not complete-model or SDK acceptance.
- Source commit: `fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`
- Original files: `/Users/devil/Pro/hugging-mac/tmp/originals/vela/3-fast/fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`
- Source selections: `cached-partial, full, probe`
- Cases SHA-256: `c9152d838e5a29b02ea5b9bc1753350b0e8f9dec34950bd2bd539acb496db5c0`

## Original model

| Case | Capability | Status | Decision | Expected | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | choice | evaluated | coding | coding | True | 1394.9623329972383 |
| cooking-zh | choice | evaluated | cooking | cooking | True | 1522.1252080082195 |
| noul-positive | noul | evaluated | yes | yes | True | 822.0592089928687 |
| noul-negative | noul | evaluated | no | no | True | 883.5718329937663 |

## Converted models

Probability tolerances: atol=0.02, rtol=0.02. Require decision agreement: True.
Parity and expected-answer correctness are separate measures. Timings are single observations, not a performance benchmark.

### coreai: diagnostic-passed

- Bundle: `/Users/devil/Pro/hugging-mac/tmp/conversions/vela/3-fast/coreai/20261116T131004Z-490d417e`
- Build status: `interrupted`

| Case | Status | Max probability error | Decision | Agreement | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | passed | 2.504995120188036e-09 | coding | True | True | 8033.972665987676 |
| cooking-zh | passed | 5.566675266299725e-07 | cooking | True | True | 9216.535125000519 |
| noul-positive | passed | 7.820040781592041e-08 | yes | True | True | 8877.145125006791 |
| noul-negative | passed | 1.0605268028346382e-08 | no | True | True | 8188.851582992356 |

## Capability coverage

| Capability | Status |
| --- | --- |
| choice | exercised-in-diagnostic |
| noul | exercised-in-diagnostic |
| router_choice | untested |
| router_set | untested |
| span | untested |
| relation | untested |
| ner | untested |
| span_sets | untested |
| evidence | untested |
| router_task_adapters | untested |
| structured_state | untested |
| multiple_questions | untested |
| variable_sequence_length | untested |
| variable_options | untested |
| system_one_response | untested |
| domain_adapter | untested |
| factcheck_adapter | untested |
| feedback_adapter | untested |
| feedback_df_adapter | untested |

Untested capabilities remain required; untested does not mean unsupported.

## Tool versions

- torch: `2.13.0`
- numpy: `2.3.5`
- transformers: `5.17.0`
- safetensors: `0.8.0`
- platform: `macOS-27.2.1-arm64-arm-64bit`
- python: `3.12.13`
- coreml_device: `cpu`
- coreai_device: `cpu`
- USE_OS_COREAI: `1`
