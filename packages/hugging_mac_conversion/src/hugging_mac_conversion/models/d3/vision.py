"""Dynamic vision graph with upstream patch grids and position interpolation.

The graph processes one temporal patch group at a time, matching the upstream
vision attention boundaries. It does not resize or crop media; the original
processor owns those operations. Adapted from the Apache-2.0 recipe in NOTICE.
"""

from __future__ import annotations

from typing import Any

import torch

from hugging_mac_conversion.models.d3.utils.coreai_vision import prepare_vision


def main() -> None:
    """Export the original vision weights with dynamic upstream patch grids."""
    import argparse
    import asyncio
    import gc
    import json
    from pathlib import Path

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from hugging_mac_conversion.shared.coreai_export import convert_pytorch_to_coreai

    from ...shared.artifacts import inventory
    from ...shared.coreai import predict
    from ...shared.hub import source_inventory
    from ...shared.validation import check_tensors

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((args.source / ".source.json").read_text())
    if (
        provenance["repo_id"] != "vllm-sr/d3-edge"
        or source_inventory(args.source) != provenance["files"]
    ):
        raise ValueError("Verified original D3-edge snapshot required")
    args.output.mkdir(parents=True, exist_ok=False)
    config = AutoConfig.from_pretrained(args.source).vision_config
    config._attn_implementation = "sdpa"
    visual: Any = Qwen3_5VisionModel(config)
    visual = visual.float().eval().requires_grad_(False)
    weights = {}
    for path in args.source.glob("*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not an iterable mapping
                if key.startswith("visual."):
                    weights[key.removeprefix("visual.")] = handle.get_tensor(key)
    visual.load_state_dict(weights, strict=True)
    del weights
    graph = prepare_vision(visual)
    with torch.no_grad():
        expected = graph.module(*graph.args).numpy()
    artifact = convert_pytorch_to_coreai(
        graph.module,
        args.output / "vision.aimodel",
        example_inputs=graph.args,
        dynamic_shapes=graph.dynamic_shapes,
        input_names=graph.input_names,
        output_names=[graph.output_name],
        optimize=False,
    )
    inputs = {
        name: value.numpy() for name, value in zip(graph.input_names, graph.args, strict=True)
    }
    del graph, visual
    gc.collect()
    actual = asyncio.run(predict(artifact, inputs, "cpu"))["features"]
    validation = check_tensors(actual, expected)
    report = {
        "status": "validated",
        "source_revision": provenance["revision"],
        "compute_dtype": "float32",
        "source_weights": "original values, no quantization",
        "dynamic_patch_count": True,
        "image_resizing": "unchanged upstream processor",
        "validation": validation,
        "files": inventory(args.output),
    }
    (args.output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "build.json", flush=True)


if __name__ == "__main__":
    main()
