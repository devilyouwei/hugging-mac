# Vela3.1-fast decision-only SDK variant — 2026-11-17

The SDK exposes `vllm-sr/vela`, variant `3_1-fast`, through the existing
`DecisionMaking` capability and both `pytorch` and `coreai` runtimes.

## Artifact selection

Core AI selects only `coreai/graphs/decision/**`, `coreai/weights/model.safetensors`,
`coreai/manifest.json`, `coreai/config.json`, `coreai/LICENSE` and `coreai/NOTICE`
from `hugging-mac/vela3.1-fast`, removing the `coreai/` prefix locally. Its tokenizer
is a separate shared resource for 3.1. No task adapter or adapter graph is needed.
Original checkpoint selection likewise includes only the decision backbone and
its Python/configuration files. Weight size is 1,521,771,536 bytes.

The verification source was revision `0ed6ed301fbbf0c312f11a13ad17e3a6f8887262`;
the converted graph came from the previously validated full bundle
`tmp/conversions/vela/3_1-fast/coreai/20261117T071119Z-ef45798f`. SDK YAML follows
the repository's unpinned-source convention. This records the versions tested,
not a guarantee about future upstream revisions.

## Real-weight parity

A local managed directory was populated using the declared download selections,
with no adapter files. The SDK engine compared complete responses with the
original references in `tmp/evaluations/vela/3_1-fast/20261117T064718Z-fa9799cc`.
No tolerance changes or new graph exports were needed.

| Core AI case | Maximum absolute response error |
| --- | --- |
| Multilingual choice | 1.6445636441195233e-9 |
| Structured state, multiple questions, described noul | 9.683892929213123e-8 |
| Single option | 0 |
| Exact 1024-token input | 8.209282505999838e-7 |
| Exact 2048-token input | 4.7511286482837534e-7 |

All five passed; complete field/label/usage comparisons remain strict except
floating values (`atol=0.02, rtol=0.02`). The Core AI process did not import Torch
or Transformers. The PyTorch engine separately passed the structured multi-question
case with zero response difference. Records are in
`tmp/diagnostics/vela31-variant/coreai-results.json` and `pytorch-results.json`.

The original dynamic 32768 bound is retained. Full 32768-token inference was not
stress-tested. Only choice/noul are part of this variant: adapter-based router,
span, entity, relation and evidence tasks belong to the full model interface.
Custom true/false noul descriptions are enabled for 3.1, preserving its renderer.

## Reusable SDK smoke test

The public-capability smoke passed for `3_1-fast` (1 test, 141.65 seconds).
The final SDK variant/converter/convention/documentation suite passed all 43 tests;
web platform tests also passed. Ruff and mypy (20 Vela source files) passed.

`tests/sdk/test_vela_smoke.py` accepts `VELA_VARIANT` (default `3-fast`). It stages
only the selected variant's declared artifacts and compares PyTorch/Core AI
responses through the public SDK handle, including JSON state and a single option.

```sh
VELA_VARIANT=3_1-fast \
VELA_SOURCE_PATH=tmp/originals/vela/3_1-fast/0ed6ed301fbbf0c312f11a13ad17e3a6f8887262 \
VELA_COREAI_PATH=tmp/conversions/vela/3_1-fast/coreai/20261117T071119Z-ef45798f \
USE_OS_COREAI=1 OMP_NUM_THREADS=1 \
uv run --project packages/hugging_mac_conversion --locked pytest tests/sdk/test_vela_smoke.py -m smoke -q
```
