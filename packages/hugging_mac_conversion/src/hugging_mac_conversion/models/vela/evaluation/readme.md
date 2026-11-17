# Vela evaluation

[Model guide / 中文](../readme.md)

- `config.yaml`, `cases/decisions.json`, `runner.py`: existing decision-only diagnostics through the shared evaluation CLI.
- `full_cases.py`: complete Vela 3.1 cases, generated from the pinned router configuration and original tokenizer.
- `full.py`: original-vs-Core AI component tensors and recursive complete-response comparisons, with source/build/reference fingerprint verification.
- `sdk_validation.py`: executes packaged fixtures through the SDK; conversion validation belongs here, not in the SDK model package.

Full evaluation checks choices/sets, spans and their offsets/scores, thresholds, errors, usage, dynamic input lengths and all adapter routes. Floating fields retain atol=rtol=0.02; fields and discrete outputs must agree. Passing only task probabilities is insufficient. References use original FP32 weights and original inference functions; active-adapter loading limits memory without removing an adapter.

Machine-readable reports live under project-local `tmp/evaluations/vela/<variant>`. Dated Markdown reports live here. The diagnostic and full pipelines retain separate acceptance rules; a decision-only probe cannot become a full release.

Latest: [Vela3.1-fast full Core AI](coreai-3_1-fast-2026-11-17.md).
Earlier: [Vela3-fast dynamic decisions](coreai-dynamic-2026-11-16.md), [initial conversion](results-3-fast-2026-11-16.md).
