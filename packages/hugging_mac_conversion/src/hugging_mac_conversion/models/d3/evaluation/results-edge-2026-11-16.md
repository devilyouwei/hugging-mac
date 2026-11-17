# D3-edge 转换与评测记录 — 2026-11-16

本轮原始模型 5/5 用例答对。Core ML 和 Core AI 均已生成诊断制品，但均未通过任务输出校验，
**不能作为可用模型注册到 SDK 或发布到 Hugging Face**。未放宽 atol/rtol，也未删除上游能力。

## 结果汇总

| 项目 | 结果 |
| --- | --- |
| 原始快照 | 完整下载，34 个文件，commit bee5e872f75b9c5d1405767bf244a6948b95c0af |
| 原始 choice / noul / score | 5/5 预期答案正确 |
| Core ML 导出 | 成功生成 model.mlpackage |
| Core ML 张量 smoke | 失败，251/255 超出 atol=rtol=0.02；最大绝对误差约 8.245518 |
| Core ML 任务对比 | 0/5 通过；5 个输出均触发 Non-finite or incompatible decision outputs |
| Core AI 导出 | 成功生成 model.aimodel |
| Core AI 自动设备校验 | 加载时资源压力导致中止；可用磁盘降至 2.0 GiB，保留制品 |
| Core AI 显式 CPU 任务对比 | 完成执行，0/5 通过；5 个输出均触发同一有限性/接口检查错误 |

输出校验错误时没有有效的转换概率或决策，因此不计算虚假的“转换后准确率”或概率误差。
本轮旧格式异常信息没有区分非有限值和形状不匹配，不能单凭该信息断言具体根因；
评测工具已改进异常信息，供后续定位使用。

## 本轮处理过的转换障碍

1. Core ML 无法将 BFloat16 常量转为 NumPy：导出图提升为 Float32，原始模型基准仍为 BFloat16。
2. 旋转位置编码的步长切片赋值转换失败：改为函数式选择，保留三个位置轴；独立测试检验数值完全一致及不修改输入。
3. Core ML 缺少 linalg_solve_triangular 转换：补充下三角、左侧、单位对角系统的前向代入实现，其他形式明确拒绝。
   8×8 和 64×64 的真实 Core ML CPU 测试均通过（atol=rtol=1e-5），8×8 smoke 最大误差约 2.38e-7。
   小算子测试通过不代表整模型通过；整模型仍然存在上述大误差。
4. Core AI 不支持掩码构造中的 aten.remainder.Scalar：将由固定架构宽度决定的掩码改为常量张量；未限制序列维或删掉位置轴。

这些修改目前属于诊断转换路径，不能据此标记完整能力已支持。
额外尝试了同输入下原生 PyTorch BFloat16/Float32 图的对照，但本机内存压力下未完成，
已终止该补充诊断；没有将未完成的实验用于根因结论。

## 环境、制品与范围

- Mac17,5 / arm64 / 8 GiB RAM；macOS 27.2.1 (26B131)；Python 3.12.13。
- Torch 2.13.0、coremltools 9.0、coreai-torch 0.4.2、coreai-core 1.0.0b2。
- Core ML：CPU_ONLY、Float32、macOS15 target；工具提示 Torch 2.13 尚未正式测试。
- Core AI 任务评测：USE_OS_COREAI=1，显式 cpu。自动设备模式的资源中止与 CPU 模式的数值失败分别记录。
- Core ML bundle：`tmp/conversions/d3/edge/coreml/20261116T132202Z-784bf9b5`。
- Core AI bundle：`tmp/conversions/d3/edge/coreai/20261116T132430Z-c1528c3c`。
- 本轮仅固定 batch=1、sequence=128 的文本决策诊断图；完整图像、视频、变长输入、多问题、完整返回结构仍未验证。
- 不将未测试能力标记为无法转换；也不将小规模用例通过视为通用准确率或性能基准。

## 复现

从项目根目录执行，先按同一 commit 完整 fetch：

```bash
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert probe d3 --variant edge --runtime coreml --revision bee5e872f75b9c5d1405767bf244a6948b95c0af --offline
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert evaluate d3 --variant edge --run tmp/conversions/d3/edge/coreml/20261116T132202Z-784bf9b5
USE_OS_COREAI=1 uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert evaluate d3 --variant edge --coreai-device cpu --run tmp/conversions/d3/edge/coreai/20261116T132430Z-c1528c3c
```

后续应先比较同一填充输入的 PyTorch 诊断图与上游概率，再分层定位首个异常输出，
分别检查精度提升、attention mask、gated delta recurrence 和后端执行。
在该问题解决之前，不应继续把完整模型或 SDK 验收标记为通过。

## Core ML 本轮原始报告

JSON/Markdown：`tmp/evaluations/d3/edge/20261116T132612Z-62b8cd0e/`。

# d3 / edge evaluation

- Status: **failed**
- Scope: `diagnostic`; this is not complete-model or SDK acceptance.
- Source commit: `bee5e872f75b9c5d1405767bf244a6948b95c0af`
- Original files: `/Users/devil/Pro/hugging-mac/tmp/originals/d3/edge/bee5e872f75b9c5d1405767bf244a6948b95c0af`
- Source selections: `cached-partial, full`
- Cases SHA-256: `2ba1b479203fe38eaa0c521e9086a111c27a79ed99b7500f51d03dfb6180d3fd`

## Original model

| Case | Capability | Status | Decision | Expected | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | choice | evaluated | coding | coding | True | 3182.692959002452 |
| cooking-zh | choice | evaluated | cooking | cooking | True | 1900.072000004002 |
| noul-positive | noul | evaluated | true | true | True | 1809.034417005023 |
| noul-negative | noul | evaluated | false | false | True | 2099.656791993766 |
| score-positive | score | evaluated | 1 | 1 | True | 1766.161708990694 |

## Converted models

Probability tolerances: atol=0.02, rtol=0.02. Require decision agreement: True.
Parity and expected-answer correctness are separate measures. Timings are single observations, not a performance benchmark.

### coreml: failed

- Bundle: `/Users/devil/Pro/hugging-mac/tmp/conversions/d3/edge/coreml/20261116T132202Z-784bf9b5`
- Build status: `failed`

| Case | Status | Max probability error | Decision | Agreement | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | error | — | — | — | — | 5497.517541996785 |
| cooking-zh | error | — | — | — | — | 254.2706249951152 |
| noul-positive | error | — | — | — | — | 128.01225000293925 |
| noul-negative | error | — | — | — | — | 133.87091699405573 |
| score-positive | error | — | — | — | — | 129.37999999849126 |

coding-en: Non-finite or incompatible decision outputs

cooking-zh: Non-finite or incompatible decision outputs

noul-positive: Non-finite or incompatible decision outputs

noul-negative: Non-finite or incompatible decision outputs

score-positive: Non-finite or incompatible decision outputs

## Capability coverage

| Capability | Status |
| --- | --- |
| text | untested |
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

- torch: `2.13.0`
- numpy: `2.3.5`
- transformers: `5.17.0`
- safetensors: `0.8.0`
- platform: `macOS-27.2.1-arm64-arm-64bit`
- python: `3.12.13`
- coreml_device: `cpu`
- coreai_device: `auto`
- USE_OS_COREAI: `—`

## Core AI CPU 本轮原始报告

JSON/Markdown：`tmp/evaluations/d3/edge/20261116T132713Z-839ded75/`。

# d3 / edge evaluation

- Status: **failed**
- Scope: `diagnostic`; this is not complete-model or SDK acceptance.
- Source commit: `bee5e872f75b9c5d1405767bf244a6948b95c0af`
- Original files: `/Users/devil/Pro/hugging-mac/tmp/originals/d3/edge/bee5e872f75b9c5d1405767bf244a6948b95c0af`
- Source selections: `cached-partial, full`
- Cases SHA-256: `2ba1b479203fe38eaa0c521e9086a111c27a79ed99b7500f51d03dfb6180d3fd`

## Original model

| Case | Capability | Status | Decision | Expected | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | choice | evaluated | coding | coding | True | 1440.344125003321 |
| cooking-zh | choice | evaluated | cooking | cooking | True | 1457.5505829998292 |
| noul-positive | noul | evaluated | true | true | True | 1436.0875840066 |
| noul-negative | noul | evaluated | false | false | True | 1442.699375009397 |
| score-positive | score | evaluated | 1 | 1 | True | 1369.4482920109294 |

## Converted models

Probability tolerances: atol=0.02, rtol=0.02. Require decision agreement: True.
Parity and expected-answer correctness are separate measures. Timings are single observations, not a performance benchmark.

### coreai: failed

- Bundle: `/Users/devil/Pro/hugging-mac/tmp/conversions/d3/edge/coreai/20261116T132430Z-c1528c3c`
- Build status: `interrupted`

| Case | Status | Max probability error | Decision | Agreement | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | error | — | — | — | — | 6450.145291993977 |
| cooking-zh | error | — | — | — | — | 5132.107040990377 |
| noul-positive | error | — | — | — | — | 3982.282957993448 |
| noul-negative | error | — | — | — | — | 1581.1948329937877 |
| score-positive | error | — | — | — | — | 1288.8028330053203 |

coding-en: Non-finite or incompatible decision outputs

cooking-zh: Non-finite or incompatible decision outputs

noul-positive: Non-finite or incompatible decision outputs

noul-negative: Non-finite or incompatible decision outputs

score-positive: Non-finite or incompatible decision outputs

## Capability coverage

| Capability | Status |
| --- | --- |
| text | untested |
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

- torch: `2.13.0`
- numpy: `2.3.5`
- transformers: `5.17.0`
- safetensors: `0.8.0`
- platform: `macOS-27.2.1-arm64-arm-64bit`
- python: `3.12.13`
- coreml_device: `cpu`
- coreai_device: `cpu`
- USE_OS_COREAI: `1`
