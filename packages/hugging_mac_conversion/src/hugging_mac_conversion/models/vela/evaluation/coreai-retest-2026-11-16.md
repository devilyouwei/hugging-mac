# Vela3-fast Core AI 复测 — 2026-11-16

结果：**6/6 转换一致性用例通过，原始和转换后答案均为 6/6 正确**。
最大概率绝对误差 5.566675266299725e-7；atol=rtol=0.02，要求决策一致。
本轮只运行 Core AI，未重新导出 Core ML。

复测使用此前已导出的同一制品：
`tmp/conversions/vela/3-fast/coreai/20261116T131004Z-490d417e/model.aimodel`。
制品由旧工具链导出，本轮运行时为 coreai-core 1.0.0b3，显式 CPU，原始基准为 Torch 2.14.0。
导出时自动设备校验中止的 build.json 状态保留，不能用本轮 CPU 结果冒充自动设备验证结果。
新用例增加了中文编程与英文烘焙；保留原有四个用例。

这仍是固定长度、两个选项的 choice/noul 诊断图，不是全部 adapter / span / router 能力的完整发布。
复测结束时，因为同批 D3 验证未通过，未进行 Git commit 或 Hub 上传。
随后按用户要求单独发布 Vela 实验性制品，记录见下节；D3 远端未改动。

## 独立实验性发布验证

已发布到 [hugging-mac/vela3-fast-CoreAI](https://huggingface.co/hugging-mac/vela3-fast-CoreAI)，
Hub commit：[e5e4742](https://huggingface.co/hugging-mac/vela3-fast-CoreAI/commit/e5e4742e6bcea4ea256fc39515831e9ebc3a3387)。
此次仅发布固定 128-token、两个选项的 choice/noul 图；未改变完整模型发布验收条件。

- 独立 uv Python 环境未安装 PyTorch，六个用例全部通过，输入张量与上游逐项一致。
- 五类无效输入按预期拒绝；Python 基本调用示例运行成功。
- Swift 示例编译并执行成功，CPU 原生推理与 Python 概率最大误差为 `5.9604644775390625e-8`。
- 发布代码通过 Ruff、mypy；转换包测试为 38 passed、2 deselected。
- 远端 24 个发布文件及自动生成的 `.gitattributes` 清单符合预期；文件大小和 LFS SHA-256 核对通过。
  README、manifest、校验清单、Python 适配器、Swift 示例在固定 commit 下载后与本地逐字节一致。
- README 包含 Python/Swift 基本调用、能力边界、来源和许可证；发布包附评测记录及 `SHA256SUMS`。
- 本轮未创建本地 Git commit；未上传原始权重、虚拟环境或缓存。

复现（从项目根目录）：

```bash
USE_OS_COREAI=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert evaluate vela --variant 3-fast --coreai-device cpu --run tmp/conversions/vela/3-fast/coreai/20261116T131004Z-490d417e
```

以下为机器报告生成的详细结果。

# vela / 3-fast evaluation

- Status: **diagnostic-passed**
- Scope: `diagnostic`; this is not complete-model or SDK acceptance.
- Source commit: `fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`
- Original files: `/Users/devil/Pro/hugging-mac/tmp/originals/vela/3-fast/fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`
- Source selections: `cached-partial, full, probe`
- Cases SHA-256: `779c07609db34ddd9b11a8649937ee7db4b45a3db5c1023ffda190339ca36aac`

## Original model

| Case | Capability | Status | Decision | Expected | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | choice | evaluated | coding | coding | True | 95.21100000711158 |
| cooking-zh | choice | evaluated | cooking | cooking | True | 132.54979200428352 |
| noul-positive | noul | evaluated | yes | yes | True | 86.67745799175464 |
| noul-negative | noul | evaluated | no | no | True | 92.96466699743178 |
| coding-zh | choice | evaluated | coding | coding | True | 130.05970799713396 |
| cooking-en | choice | evaluated | cooking | cooking | True | 100.37370900681708 |

## Converted models

Probability tolerances: atol=0.02, rtol=0.02. Require decision agreement: True.
Parity and expected-answer correctness are separate measures. Timings are single observations, not a performance benchmark.

### coreai: diagnostic-passed

- Bundle: `/Users/devil/Pro/hugging-mac/tmp/conversions/vela/3-fast/coreai/20261116T131004Z-490d417e`
- Build status: `interrupted`

| Case | Status | Max probability error | Decision | Agreement | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | passed | 2.504995120188036e-09 | coding | True | True | 5489.578958993661 |
| cooking-zh | passed | 5.566675266299725e-07 | cooking | True | True | 242.97170800855383 |
| noul-positive | passed | 7.820040781592041e-08 | yes | True | True | 200.55320800747722 |
| noul-negative | passed | 1.0605268028346382e-08 | no | True | True | 199.48370799829718 |
| coding-zh | passed | 1.1097869623277745e-08 | coding | True | True | 190.3682919946732 |
| cooking-en | passed | 1.3729699288285782e-09 | cooking | True | True | 196.19354100723285 |

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
