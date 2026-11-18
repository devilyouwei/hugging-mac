# Laya model package

`register_laya` registers `convaiinnovations/laya` and the shared `DecisionMaking`
capability. Registration imports no Torch, NumPy, tokenizers, Laya or Core AI runtime.

| Variant | Runtime | Checkpoint and limit |
| --- | --- | --- |
| multilingual | pytorch | convaiinnovations/laya-multilingual; original 1024 default, configurable up to 8192 |
| multilingual-legacy-s256 | coreai | older convaiinnovations/laya/multilingual checkpoint; fixed 256 tokens |
| multilingual-legacy-s512 (default) | coreai | same older checkpoint; fixed 512 tokens |

The Core AI exports come from mlboydaisuke/Laya-Multilingual-CoreAI and identify
source revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 (laya 0.3.4). They are not
claimed equivalent to the newer independent multilingual checkpoint. FP16 weight
storage with FP32 compute is the publisher's WFP16 format. Only macOS CPU execution
is exposed here; GPU/ANE and compiled iOS assets are not registered.

`model.yaml` owns sources, variant metadata, required files and tokenizer shares.
`definition.py` injects runtime engines; `instance.py` owns lifecycle and serialized
decisions; `resources.py` owns download/status/delete and merged tokenizer views.
`coreai.py` calls both main and act; `torch.py` uses the original local Laya Agent.
Private rendering and drained worker execution live under `utils/`.

Original and exported tokenizers are separate shares. Loading never downloads or
converts. Select both variant and runtime explicitly for the PyTorch variant.
No local SDK converter is supplied for this model.

Core AI renders the upstream head layout (256-token head budget, option text up to
48 tokens). State overflow is rejected explicitly instead of silently truncating.
Answers preserve `action.act_probability`. Core AI uses source temperatures (1),
not the third-party optional fitted calibration. PyTorch rejects answers whose upstream usage reports truncated state; its rounded probabilities are normalized for the shared typed
SDK contract, with score recomputed as the expected level index. Confidence and
act probability are preserved.

Install `hugging-mac-sdk[laya]` for PyTorch or `[laya-coreai]` for Core AI.

## Verification — 2026-11-18

Both downloaded macOS WFP16 artifacts (256 and 512 tokens) passed all 201
publisher reference rows each: main/act inference, answer probabilities, act
probability, option markers and rendered question headers (`atol=rtol=1e-4`).
Both also passed public SDK choice/score/noul calls on structured Chinese state,
including typed act probabilities. These are conversion-parity checks, not task
accuracy benchmarks.

The current original PyTorch checkpoint previously passed the same public SDK
question types. It is a different checkpoint and is not treated as numerically
equivalent to these exports. No 8192-token stress test was run.

Repeat the publisher-oracle test using `LAYA_COREAI_PATH` pointing to the complete
published variant folder (including `reference.json` and tokenizer),
`USE_OS_COREAI=1`, and `pytest tests/sdk/test_laya_smoke.py -m smoke`.
Local run records: `tmp/diagnostics/laya/coreai-results.json`,
`coreai-sdk.json` and `pytorch-sdk.json`. Production downloads omit the reference
fixtures. Unit tests cover lazy registration, checkpoint/tokenizer separation,
fixed-window validation, typed action retention and managed resource views.
