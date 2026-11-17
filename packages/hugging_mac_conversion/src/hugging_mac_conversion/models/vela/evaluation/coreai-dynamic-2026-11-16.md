# Vela3-fast dynamic Core AI export — 2026-11-16

The previous fixed 128-token, two-option export imposed conversion-only restrictions. This export removes both restrictions. The same Core AI artifact handles variable sequence lengths and variable option counts. The host adapter retains the upstream 32,768-token bound without truncation.

- Export sample: 1,024 tokens; initial CPU tensor validation max absolute error: 1.43e-6.
- 12/12 original-versus-CoreAI cases passed, including exact 129/1,024/2,048-token inputs, 1/2/3 options and a JSON object. Maximum probability error: 7.80e-7.
- Standalone Python validation passed without PyTorch installed, including multiple questions and list criteria.
- Native Swift CPU invocation passed; max probability error against Python: 5.96e-8. Swift allocates concrete input shapes for each request.
- Host encoding accepts exactly 32,768 tokens and rejects 32,769 without truncation. This is an encoding boundary test, not native 32K inference verification.
- Native 32K inference has not been tested on this 8 GB host. The graph retains the upstream bound; 2,048 is the tested length, not an imposed input limit.
- Scope remains the decision backbone; router/span and other adapters are not in this artifact. These untested capabilities are not declared unconvertible.

Published to [hugging-mac/vela3-fast-CoreAI](https://huggingface.co/hugging-mac/vela3-fast-CoreAI), commit [6aaa9cc](https://huggingface.co/hugging-mac/vela3-fast-CoreAI/commit/6aaa9ccd747c4bad65037979e7f9373a76fb7683). Remote file inventory, sizes, model LFS SHA-256, and downloaded README/manifest/runtime examples were verified against the local release.

Build: `tmp/conversions/vela/3-fast/coreai/20261116T154942Z-ca377c73`.
Evaluation: `tmp/evaluations/vela/3-fast/20261116T155134Z-c0f38069`.
Publication staging: `tmp/releases/vela3-fast-CoreAI-dynamic`.

```bash
USE_OS_COREAI=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert probe vela --variant 3-fast --runtime coreai --coreai-device cpu --revision fa60e3aa41132c60bf05a84cabfdced39cb6d4f7 --offline
USE_OS_COREAI=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert evaluate vela --variant 3-fast --coreai-device cpu --run tmp/conversions/vela/3-fast/coreai/20261116T154942Z-ca377c73
```

# vela / 3-fast evaluation

- Status: **diagnostic-passed**
- Scope: `diagnostic`; this is not complete-model or SDK acceptance.
- Source commit: `fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`
- Original files: `/Users/devil/Pro/hugging-mac/tmp/originals/vela/3-fast/fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`
- Source selections: `cached-partial, full, probe`
- Cases SHA-256: `184b5ab437c84bea6c1517d84e145ecd68ba7e4fbb7b61d7f628b9d474076e17`

## Original model

| Case | Capability | Status | Decision | Expected | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | choice | evaluated | coding | coding | True | 51.31262500071898 |
| cooking-zh | choice | evaluated | cooking | cooking | True | 68.58845800161362 |
| noul-positive | noul | evaluated | yes | yes | True | 48.34995899000205 |
| noul-negative | noul | evaluated | no | no | True | 52.418874998693354 |
| coding-zh | choice | evaluated | coding | coding | True | 70.75237500248477 |
| cooking-en | choice | evaluated | cooking | cooking | True | 51.56712500320282 |
| three-options | variable_options | evaluated | coding | coding | True | 53.129124993574806 |
| one-option | variable_options | evaluated | coding | coding | True | 49.474542000098154 |
| structured-json | structured_state | evaluated | coding | coding | True | 69.43170799058862 |
| exact-129-tokens | variable_sequence_length | evaluated | coding | coding | True | 141.63116700365208 |
| exact-1024-tokens | variable_sequence_length | evaluated | coding | coding | True | 968.6068749870174 |
| exact-2048-tokens | variable_sequence_length | evaluated | coding | coding | True | 2569.3836659920635 |

## Converted models

Probability tolerances: atol=0.02, rtol=0.02. Require decision agreement: True.
Parity and expected-answer correctness are separate measures. Timings are single observations, not a performance benchmark.

### coreai: diagnostic-passed

- Bundle: `/Users/devil/Pro/hugging-mac/tmp/conversions/vela/3-fast/coreai/20261116T154942Z-ca377c73`
- Build status: `diagnostic-validated`

| Case | Status | Max probability error | Decision | Agreement | Correct | Inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| coding-en | passed | 5.427491678311469e-09 | coding | True | True | 2374.499250014196 |
| cooking-zh | passed | 4.7419820070615515e-07 | cooking | True | True | 91.65208299236838 |
| noul-positive | passed | 1.3685062416612936e-07 | yes | True | True | 47.826540991081856 |
| noul-negative | passed | 2.6513179342962934e-09 | no | True | True | 80.2467909961706 |
| coding-zh | passed | 8.426156995235135e-09 | coding | True | True | 79.13508299679961 |
| cooking-en | passed | 1.5691086767584406e-09 | cooking | True | True | 60.23716600611806 |
| three-options | passed | 3.460630049190172e-08 | coding | True | True | 59.11033399752341 |
| one-option | passed | 0.0 | coding | True | True | 56.663583003683016 |
| structured-json | passed | 1.4141390947131072e-08 | coding | True | True | 75.03466600610409 |
| exact-129-tokens | passed | 7.795353781370196e-07 | coding | True | True | 134.83975001145154 |
| exact-1024-tokens | passed | 1.5057525626360757e-07 | coding | True | True | 978.0740829883143 |
| exact-2048-tokens | passed | 2.872474891835175e-07 | coding | True | True | 2785.3902079950785 |

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
| structured_state | exercised-in-diagnostic |
| multiple_questions | untested |
| variable_sequence_length | exercised-in-diagnostic |
| variable_options | exercised-in-diagnostic |
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
