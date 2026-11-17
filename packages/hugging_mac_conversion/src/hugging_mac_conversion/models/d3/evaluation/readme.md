# D3 evaluation

- `cases/decisions.json`, `config.yaml`, `runner.py`: original-vs-converted text decisions through the shared evaluation CLI.
- `multimodal_cases.py`: deterministic images, videos and mixed-input cases.
- `native.py`: full native graph comparison, original preprocessing, exact 1024/1536-token cases, multi-question responses, and source/case fingerprint checks for reference reuse.
- `publication.py` (parent directory): embeds portable fixtures and original expected responses in the release.
- SDK `hugging_mac_conversion.models.d3.evaluation.sdk_validation --artifact-path <bundle>`: runs the complete published bundle against those reference cases, without original neural weights.

Probability comparisons keep atol=rtol=0.02 and require matching decisions. Public response keys and usage are checked. Graph tensor smoke checks are recorded independently. No input is silently shortened to fit an export profile.

Latest: [dynamic multimodal Core AI, 2026-11-17](coreai-dynamic-2026-11-17.md). Historical diagnostic failures remain in the previous dated reports; they do not describe the current graph.
