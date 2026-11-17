# D3 Edge dynamic Core AI evaluation — 2026-11-17

## Release decision

**Not approved for publication yet.** Dynamic text and vision graphs were exported and execute locally. Task-probability checks passed, but the full FP32 SDK response check found a confidence mismatch. No limited replacement has been uploaded to `hugging-mac/d3-edge-CoreAI`. The original source and earlier failed runs are preserved under the ignored project `tmp/` directory.

## Other Hugging Face conversions

| Repository | Inspected revision | Observations |
| --- | --- | --- |
| [ahmedandaloes/d3-edge-CoreAI](https://huggingface.co/ahmedandaloes/d3-edge-CoreAI) | `aa1716253bf76c9b63dc21f5f07dfe919a913645` | BF16 text and vision; fixed 512-token combined input; 224×224 images/video; reported maximum probability error 0.010562 on six cases. |
| [maxffarrell/d3-edge-CoreAI](https://huggingface.co/maxffarrell/d3-edge-CoreAI) | `ffa8431fe8dd5209f3ec2c8bf59afc829ec96c31` | Separate text and vision graphs; fixed 512/224×224; ten matching decisions but reported maximum probability error 0.053746, failing its 0.02 parity threshold. |
| [hugging-mac/d3-edge-CoreAI](https://huggingface.co/hugging-mac/d3-edge-CoreAI) | `61a6a13a44d8390a5cf72755a6f581238ca88740` | Previous text-only fixed-512 export; legacy iOS archive is not a validation of the new graphs. |

These are repository claims/recipes, not independently rerun third-party benchmarks. Downloaded research metadata is under `tmp/research/d3/`. Their fixed limits were not copied into this conversion. Adapted code is attributed in the model's `NOTICE`.

## Preserved contract

Original source: `vllm-sr/d3-edge`, revision `bee5e872f75b9c5d1405767bf244a6948b95c0af`, stored at `tmp/originals/d3/edge/<revision>`. `.source.json` inventories original file hashes.

The SDK reuses the pinned original request normalization, rendering, tokenizer, question validation, permutation averaging, response formatting, image processor, and video sampling. It accepts text, images, videos, structured state and multiple questions; supports choice/noul/score and the original 255 answer codes. Inference uses two Core AI graphs; no original neural weights are loaded by the SDK.

Text sequence length and vision patch count are dynamic. Original `max_length: null` is retained. The decision graph's symbolic sequence bound is 262144 from the architecture configuration; tested exact text lengths are 1024 and 1536. Neither 32K nor the architectural maximum has been stress-tested on this 8-GB host. No new 128/512/1024 cap or forced 224×224 resize is imposed. The original processor's own resizing, frame sampling and budgets still apply.

Decision inputs: int32 `input_ids[1,L]`, int32 `attention_mask[1,L]`, int32 `position_ids[4,1,L]`, float32 `visual_embeddings[1,L,1024]`, bool `visual_mask[1,L,1]`; output float32 `logits[1,255]`. Vision inputs: `pixels[N,1536]`, int32 `indices[N,4]`, float32 `weights[N,4]`, int32 `positions[N,2]`; output `features[N/4,1024]`. Temporal patch groups use the original vision attention boundaries.

## Numerical investigation

1. Earlier static CPU exports produced NaNs. Layer probes isolated the first failure to DeltaNet normalization. A small sum-based normalization exported with an observed RMS-like scale discrepancy. The algebraically equivalent `mean(x²) + epsilon/D`, followed by the original scaling, avoids this issue. This is an observed lowering/runtime failure, not a claim about Apple's internal implementation.
2. The original causal DeltaNet recurrence is retained using a dynamic `torch.while_loop`; unrelated full-attention layers remain noncausal. Native recurrence tests exercise lengths 5, 8 and 17; PyTorch tests also cross the chunk boundary at lengths 64 and 67.
3. BF16 decision export produced finite task outputs and passed the eight multimodal complete-response comparisons, but failed the stricter raw 255-logit smoke check (maximum offending absolute difference 0.09308171). It is not approved for release.
4. Promoting the original BF16 weight values exactly to FP32 passed the decision tensor check (maximum absolute difference 0.00000786781311). FP32 vision passed its tensor check (0.0002475977). No retraining or weight quantization was performed.
5. The FP32 decision + vision pair passed task-probability comparisons, but the SDK's recursive complete-response check exposed a derived confidence mismatch. Reference confidence: 0.715477; converted: 0.756597; absolute difference: 0.04111985. Tolerances remain `atol=0.02, rtol=0.02`. This failure blocks publication.
6. Small cast probes also showed that a FP32→BF16→FP32 path did not retain the expected BF16 rounding. A separate arithmetic probe observed `exp2` behaving incorrectly; these experiments remain diagnostics, not released graph changes. Explicit rounding preservation was also tested separately: the zero-safe 58-cast candidate remained finite but failed raw logits (maximum offending absolute error 0.10024112). Its failed graph was removed to recover disk; hashes, arrays and recipe are retained. A repeat of the small cast/linear probe under supported torch 2.13.0 / torchvision 0.28.0 still failed the round-trip check (maximum difference 0.05941868), so the observed issue is not confined to torch 2.14.

## Measured runs

| Candidate / check | Result |
| --- | --- |
| BF16 decision `20261116T160946Z-dca889c4` raw tensors | Failed; finite but outside tolerance |
| BF16 decision + FP32 vision, `native-dynamic-20261117` | 8/8 task and complete-response comparisons passed; does not override tensor failure |
| FP32 decision `20261116T161831Z-736d70e3` raw tensors | Passed |
| FP32 vision `vision-dynamic-fp32-20261117` raw tensors | Passed |
| FP32 text `20261116T162422Z-bcaa3196` | 8/8 task comparisons; max probability error 0.0024952412 |
| FP32 multimodal `native-dynamic-fp32-20261117` | 8/8 probability checks; max probability error 0.01384668052 |
| Same FP32 multimodal, complete-response recheck | 7/8 passed; two-images/multiple-questions confidence failed |
| SDK complete-response suite | 16/17 passed; two-images/multiple-questions confidence failed. The rebuilt isolated wheel also completed all 17 cases with the same single failure. |

The eight multimodal/long-input cases are landscape image, portrait image, two images with noul and score questions, video frames, image plus video, exact 1024 tokens, exact 1536 tokens, and structured state with multiple questions. The release fixtures additionally include eight text cases and an invalid-question response case.

References use the untouched original BF16 source. Multimodal references were reused only after matching source-inventory and deterministic case fingerprints. The first reference report's fingerprints were attached after the run and explicitly marked `fingerprint_added_after_run`; subsequent runs validate them. CPU is the tested Core AI device. ANE, GPU and new-graph iOS execution are not claimed.

## Reproduction and files

Run from the repository root using `uv`; conversion has its own package/environment.

```sh
USE_OS_COREAI=1 OMP_NUM_THREADS=1 uv run --project packages/hugging_mac_conversion --locked \
  hugging-mac-convert probe d3 --variant edge --runtime coreai --coreai-device cpu \
  --offline --revision bee5e872f75b9c5d1405767bf244a6948b95c0af

USE_OS_COREAI=1 uv run --project packages/hugging_mac_conversion --locked \
  python -m hugging_mac_conversion.models.d3.vision --source <original-directory> --output <new-vision-directory>

USE_OS_COREAI=1 uv run --project packages/hugging_mac_conversion --locked \
  python -m hugging_mac_conversion.models.d3.evaluation.native \
  --source <original-directory> --decision <decision.aimodel> --vision <vision.aimodel> \
  --output <new-evaluation-directory>
```

`evaluation/native.py` now checks all public answer values as well as probabilities and usage; the older probability-only report cannot authorize publication. `publication.py` independently repeats the complete-response check before assembling a candidate. `hugging_mac_conversion.models.d3.evaluation.sdk_validation --artifact-path <bundle>` checks the actual packaged SDK; it records per-case failures and exits nonzero on failure.

Artifacts: `tmp/conversions/d3/edge/coreai/`. Reports: `tmp/evaluations/d3/edge/`, including `complete-response-validation.json` beside each native report. The unapproved candidate bundle is at `tmp/releases/d3-edge-CoreAI-dynamic/`. Diagnostic scripts/arrays are at `tmp/diagnostics/d3/`. These paths are ignored by Git.

Environment: Python 3.12.13, torch 2.14.0, transformers 5.17.0, coreai-torch 0.4.3, coreai-core 1.0.0b3. The converter warns that validation covers torch ≤2.13; this warning is retained. Conversion unit suite: 41 passed, 3 smoke tests deselected; two additional complete-response regression tests passed. Model/SDK Ruff and mypy checks passed. These software checks do not override the numerical release failure.
