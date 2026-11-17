---
language:
- en
- zh
license: apache-2.0
base_model: vllm-sr/vela3-fast
pipeline_tag: zero-shot-classification
tags:
- coreai
- apple-silicon
- aimodel
- experimental
- decision-model
---

# Vela3-fast Core AI — experimental decision graph

English | [简体中文](README.zh-CN.md)

An Apple Core AI conversion of [vllm-sr/vela3-fast](https://huggingface.co/vllm-sr/vela3-fast) by hugging-mac.
This release contains the validated **choice / noul decision backbone**: one sample per graph call, dynamic sequence length, and a dynamic number of options.
The export retains the upstream **32,768-token** input bound; actual CPU parity tests currently cover lengths through **2,048 tokens**, including exactly 1,024.
**It is an experimental subset, not the complete Vela3-fast model with all task adapters.**

Neural inference runs through Core AI, with host-side tokenization and response formatting. The model does not generate text.

## Supported contract

- `choice`: one or more ordered options; returns their probabilities and the selected option.
- `noul`: fixed `no` / `yes` options; returns `p(yes)`.
- The input text, question, option descriptions, and special markers together must fit within the upstream **32,768-token** bound. Longer inputs raise an error; they are not truncated.
- Each graph call has batch size 1. `predict_questions(state, questions)` accepts a mapping of questions and returns `(answers, input_tokens)`, matching the upstream decision API; questions execute sequentially. The adapter preserves upstream JSON serialization and accepts mapping or list choice criteria.
- CPU execution is validated. Earlier automatic-device validation did not finish because of resource pressure on the test host; no GPU/ANE performance is claimed.
- Original FP32 decision weights, with no quantization, training, or fine-tuning.
- Router, span, relation, NER, evidence, domain/factcheck/feedback, and other upstream adapters are not included. Full `system_one` adapter routing is not implemented by this decision-only artifact. Untested capabilities are not known to be unconvertible.
- This release does not provide complete `hugging_mac_sdk` registration or production acceptance. Probabilities are relative to the supplied options, not calibrated confidence scores.

## Python usage

Validated on Apple Silicon, macOS 27.2.1, Python 3.12.13, and `coreai-core==1.0.0b3`.
A macOS version supporting the Core AI system runtime is required. The example environment uses uv and **does not require PyTorch, transformers, or the original safetensors weights**.

```bash
uvx --from huggingface-hub hf download hugging-mac/vela3-fast-CoreAI --local-dir vela3-fast-CoreAI
cd vela3-fast-CoreAI
uv sync --locked
USE_OS_COREAI=1 uv run --locked python example.py
```

```python
import asyncio
from vela_coreai import VelaCoreAI

async def main():
    # Point to the downloaded repository; the adapter explicitly uses CPU mode.
    async with VelaCoreAI.open(".") as model:
        result = await model.predict(
            "Write a Python sorting function.",
            {
                "type": "choice",
                "instructions": "What is the topic?",
                "criteria": {"coding": "Software", "cooking": "Food"},
            },
        )
        print(result)  # type, choice, probabilities

        yes_no = await model.predict(
            "Python code",
            {"type": "noul", "instructions": "Is this about software?"},
        )
        print(yes_no)  # {"type": "noul", "noul": p_yes}

asyncio.run(main())
```

`predict` prepares token IDs, marker positions, and the pooling mask before calling Core AI.
Each instance accepts one inference request at a time; await completion before making another call.
Unknown tasks, empty option sets, and inputs exceeding the token limit raise errors.

Run the standalone package checks:

```bash
USE_OS_COREAI=1 uv run --locked python smoke.py
```

This compares 12 cases with upstream reference probabilities, checks exact input-tensor equality against upstream encoding,
and verifies rejection of three invalid-request categories. It writes `package_smoke.json` and `swift_fixture.json` for the Swift example.

## Swift usage

Requires Xcode 27 / the macOS 27 SDK or newer. `verify_coreai.swift` is a complete, compilable example that
loads pretokenized inputs from `swift_fixture.json`, runs native `CoreAI` inference, and compares probabilities with Python.
**The Swift example demonstrates native model invocation; it does not include a Swift tokenizer or a complete SDK for arbitrary text inputs.**

```bash
# Run inside the model directory. Adjust DEVELOPER_DIR if Xcode is installed elsewhere.
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swiftc -parse-as-library -target arm64-apple-macos27.0 \
  verify_coreai.swift -o /tmp/verify-vela
/tmp/verify-vela "$PWD"
```

The core invocation is shown below; see `verify_coreai.swift` for complete tensor construction:

```swift
import CoreAI
import Foundation

let model = try await AIModel(
    contentsOf: modelDirectory.appendingPathComponent("model.aimodel"),
    options: .cpuOnly
)
guard let function = try model.loadFunction(named: "main") else {
    fatalError("Missing main function")
}
// inputs: [String: NDArray], following the dtypes, shapes, and encoding rules below.
var result = try await function.run(inputs: inputs)
guard let logits = result.remove("logits")?.ndArray else {
    fatalError("Missing logits")
}
// Apply stable softmax to the output logits and interpret them in input-option order.
```

For new text, reuse `tokenizer.json`, `decision.json`, and the encoding rules in `vela_coreai.py`.
Tokenizing only the input body does not produce valid model inputs. The bundled fixture is a synthetic example and contains no user data.

## Tensor interface

Entry point: `main`. No persistent recurrent/KV state. Sequence and option axes are dynamic; inputs use their actual encoded length without fixed padding.

| Name | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `input_ids` | int32 | `[1,sequence]` | Complete question/options/body sequence |
| `attention_mask` | int32 | `[1,sequence]` | 1 for valid tokens, 0 for padding |
| `option_positions` | int32 | `[1,options]` | Positions of the `[O]` markers |
| `pool_mask` | float32 | `[1,sequence]` | 1 from `[SEG_user]` through the end of the body, excluding EOS |
| `logits` (output) | float32 | `[1,options]` | Unnormalized scores for the one or more ordered options |

Upstream sequence format:

```text
[BOS] [Q] instructions [O] key_1: description_1 [O] key_2: description_2
[SEP_SCHEMA] [SEG_user] state [EOS]
```

## Validation and limitations

- **12/12** text cases matched the original model's decisions and the expected answers.
- Maximum absolute probability error was approximately **7.80e-7**. Validation uses `atol=0.02, rtol=0.02` and also requires decision agreement.
- The standalone Python package passed all 12 cases without PyTorch installed, with exact input-tensor parity against upstream.
- Native Swift CPU results are in `swift_smoke.json`; standalone Python results are in `package_smoke.json`.
- These handcrafted examples are not a general accuracy benchmark. Coverage includes a JSON object, 1/2/3 options, and exact 129/1,024/2,048-token inputs.
- iPhone/iPad, iOS AOT, automatic device selection, GPU/ANE placement, lengths above 2,048 tokens, and other upstream tasks have not been validated.
- The new dynamic artifact passed initial CPU tensor validation and subsequent task evaluation. Tool versions are recorded in `manifest.json` and `environment.json`.
- The 32K bound is retained in the export and host adapter, but end-to-end 32K inference has not been validated on this 8 GB test machine. Longer inputs require additional memory; 2,048 is a tested length, not a new hard limit.
- This revision replaces the earlier fixed 128-token, two-option export. The adapters listed above remain outside the current decision graph.

Full cases and probabilities: `validation_cases.json` and `evaluation.json`.
`SHA256SUMS` covers the published files. Original weight copies, development caches, and virtual environments are not uploaded.

## Attribution and license

Upstream: [vllm-sr/vela3-fast](https://huggingface.co/vllm-sr/vela3-fast/tree/fa60e3aa41132c60bf05a84cabfdced39cb6d4f7),
pinned to commit `fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`, under Apache-2.0.
The converted graph and input rendering adapt `vela3_encoder.py` and `ub_d3_common.py` from that revision.
See `NOTICE` for attribution and modification details, and `LICENSE` for the license.
This is a hugging-mac community conversion, not an official release from Apple or the upstream team.
