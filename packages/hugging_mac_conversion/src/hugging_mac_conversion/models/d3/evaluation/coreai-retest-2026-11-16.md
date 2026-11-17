# D3-edge Core AI 复测 — 2026-11-16

结果：**原始模型 8/8 答对；新导出的 Core AI 制品 0/8 通过输出校验**。
CPU 执行的八个用例均出现非有限 logits，分别涉及二选一、三选一、二级/五级评分及结构化文本。
没有有效的转换概率，不能计算转换准确率或用更宽容差判为通过。
本轮只推进 Core AI，没有新增 Core ML 导出。

## 与已有 Hub 仓库的比较

检查了 [hugging-mac/d3-edge-CoreAI](https://huggingface.co/hugging-mac/d3-edge-CoreAI/tree/61a6a13a44d8390a5cf72755a6f581238ca88740)
的文件列表、模型卡、导出配方、运行适配器和环境记录。
已有版本是 512-token 的实验性文本模型，附带更多转换/业务验证和 iOS AOT 制品；其模型卡仍明确说明业务准确率不足。
本地旧 128-token 诊断制品不能作为它的替代品。

本轮将目标配置改为该已有仓库，并对齐其 512-token、三个输入
`input_ids` / `attention_mask` / `position_ids` 和非因果 full-attention 合同。
原始 BF16 文本权重、FP32 readout 不变；未删除原始完整能力清单。
新增打包边界测试，512 tokens 接受、513 tokens 明确拒绝，不截断；位置编码与填充位置分离。
转换配方来源及 Apache-2.0 许可已保留在模型目录的 NOTICE / licenses。

**远端未修改，没有覆盖已有 .aimodel、manifest 或 iOS AOT 文件。**
本地新版本不满足替代条件，不能仅凭同一来源 commit 或同一接口名判断可覆盖。

## 本轮实验和限制

1. 原 128-token PyTorch 诊断图对 coding-en 的输出有限，概率约 [0.999605, 0.000395]，
   上游原始概率约 [0.999618, 0.000382]。至少该用例的错误发生在后续导出/执行路径，不能归因于原始模型。
2. 关闭额外 `program.optimize()` 后，旧接口的 255-way smoke 仍不通过，最大误差约 8.019815；因此没有把优化开关当成已证实根因。
3. 工具链对齐现有仓库：Torch 2.14.0、coreai-torch 0.4.3、coreai-core 1.0.0b3、transformers 5.17.0。
   转换器仍提示 Torch 2.14 超出其声明的 <=2.13 验证范围，该警告未隐藏。
4. 新 512-token 制品成功导出到 `tmp/conversions/d3/edge/coreai/20261116T144214Z-fd0690cd/model.aimodel`，
   但 CPU 张量 smoke 出现 NaN，后续八个真实用例也都失败。
5. 自动设备模式单独尝试，8 GiB 内存主机在加载/编译阶段出现交换空间压力，可用磁盘降至约 390 MiB，已停止该进程。
   该次报告 `tmp/evaluations/d3/edge/20261116T144703Z-4ed6ccd9/` 标记 interrupted；没有完成结果，不能记为通过或数值失败。
6. 导出流程现在在释放 PyTorch 图后显式执行 GC；评测会在启动后端前保存原始基准，便于中断时保留证据。

本机为 Mac17,5 / 8 GiB / macOS 27.2.1。本轮仍缺少一个能够完成自动设备验证的结果，
也未定位 CPU 非有限值的首个产生算子。未宣称模型不可能转换，未放宽容差或删减测试用例。

## 提交决定

按“测试没问题再 commit”的条件：**本轮不提交 Git commit，不上传 Hub**。
38 项开发回归测试、Ruff、mypy 均通过，但开发检查不替代上述模型验证失败。
完整多模态、SDK capability 接入与正式发布验收仍未完成。

## 复现

```bash
USE_OS_COREAI=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert probe d3 --variant edge --runtime coreai --coreai-device cpu --revision bee5e872f75b9c5d1405767bf244a6948b95c0af --offline
USE_OS_COREAI=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert evaluate d3 --variant edge --coreai-device cpu --run tmp/conversions/d3/edge/coreai/20261116T144214Z-fd0690cd
```

以下是本轮实际 CPU 评测的逐项结果。

# d3 / edge evaluation

- Status: **failed**
- Scope: `diagnostic`; this is not complete-model or SDK acceptance.
- Source commit: `bee5e872f75b9c5d1405767bf244a6948b95c0af`
- Original files: `/Users/devil/Pro/hugging-mac/tmp/originals/d3/edge/bee5e872f75b9c5d1405767bf244a6948b95c0af`
- Source selections: `cached-partial, full`
- Cases SHA-256: `10ccd781336588bb63f6988e88975284f57ef421c8851d20bcb05f9f62111071`

## Original model

| Case | Capability | Status | Decision | Expected | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | choice | evaluated | coding | coding | True | 1976.5610419999575 |
| cooking-zh | choice | evaluated | cooking | cooking | True | 1947.2763749945443 |
| noul-positive | noul | evaluated | true | true | True | 1904.629249998834 |
| noul-negative | noul | evaluated | false | false | True | 1930.3162090072874 |
| score-positive | score | evaluated | 1 | 1 | True | 1791.967625002144 |
| choice-three-zh | choice | evaluated | sports | sports | True | 2403.1255829904694 |
| score-five-positive | score | evaluated | 4 | 4 | True | 2391.025167002226 |
| structured-text | text | evaluated | coding | coding | True | 2241.4711250021355 |

## Converted models

Probability tolerances: atol=0.02, rtol=0.02. Require decision agreement: True.
Parity and expected-answer correctness are separate measures. Timings are single observations, not a performance benchmark.

### coreai: failed

- Bundle: `/Users/devil/Pro/hugging-mac/tmp/conversions/d3/edge/coreai/20261116T144214Z-fd0690cd`
- Build status: `failed`

| Case | Status | Max probability error | Decision | Agreement | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | error | — | — | — | — | 8036.699875010527 |
| cooking-zh | error | — | — | — | — | 5783.5757919965545 |
| noul-positive | error | — | — | — | — | 6528.134792009951 |
| noul-negative | error | — | — | — | — | 8504.636208992451 |
| score-positive | error | — | — | — | — | 7850.472541991621 |
| choice-three-zh | error | — | — | — | — | 6008.147083994118 |
| score-five-positive | error | — | — | — | — | 5847.96150001057 |
| structured-text | error | — | — | — | — | 5804.92687500373 |

coding-en: Non-finite converted decision logits: 2/2

cooking-zh: Non-finite converted decision logits: 2/2

noul-positive: Non-finite converted decision logits: 2/2

noul-negative: Non-finite converted decision logits: 2/2

score-positive: Non-finite converted decision logits: 2/2

choice-three-zh: Non-finite converted decision logits: 3/3

score-five-positive: Non-finite converted decision logits: 5/5

structured-text: Non-finite converted decision logits: 2/2

## Capability coverage

| Capability | Status |
| --- | --- |
| text | exercised-in-diagnostic |
| images | untested |
| videos | untested |
| choice | exercised-in-diagnostic |
| noul | exercised-in-diagnostic |
| score | exercised-in-diagnostic |
| multiple_questions | untested |
| variable_sequence_length | untested |
| variable_options | untested |
| system_one_response | untested |

Untested capabilities remain required; untested does not mean unsupported.

## Tool versions

- torch: `2.14.0`
- numpy: `2.3.5`
- transformers: `5.17.0`
- safetensors: `0.8.0`
- coreai-core: `1.0.0b3`
- coreai-torch: `0.4.3`
- platform: `macOS-27.2.1-arm64-arm-64bit`
- python: `3.12.13`
- coreml_device: `cpu`
- coreai_device: `cpu`
- USE_OS_COREAI: `1`
