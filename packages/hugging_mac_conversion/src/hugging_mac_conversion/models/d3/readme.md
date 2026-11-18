# D3 conversion

The model owns its variant configuration (`model.yaml`), decision export composition (`converter.py`), vision evaluation (`vision.py`), evaluation, and publication assembly (`publication.py`). Apple exporters and artifact management remain in the shared conversion library. The SDK owns inference, original host preprocessing, and lifecycle under `hugging_mac_sdk.models.d3`; conversion owns independent copies and imports no SDK code. SDK `converter.py` and private
`utils/coreai_*.py` own the architecture exports; this package owns evaluation
and publication.

**Publication remains blocked by complete-response numeric validation; see the report below.**

The current Edge Core AI export preserves text, image, video, choice/noul/score, structured state, and multiple questions. Both sequence length and visual patch count are dynamic. Exact original BF16 weight values are computed in FP32. Original preprocessing and max_length=null remain unchanged.

Originals live in `tmp/originals/d3/edge/<revision>`, converted graphs in `tmp/conversions/d3/edge/coreai/<build>`, evaluation JSON in `tmp/evaluations/d3/edge/<run>`, and the upload bundle in `tmp/releases/d3-edge-CoreAI-dynamic`. All are project-local and ignored. Historical failed builds remain available.

See [evaluation](evaluation/readme.md) and the [dynamic conversion report](evaluation/coreai-dynamic-2026-11-17.md) for limitations, provenance, measurements, and reproduction commands.
