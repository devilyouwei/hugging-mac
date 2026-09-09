"""Reproducible PP-DocLayoutV3 safetensors-to-Core-ML conversion."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult


class PPDocLayoutV3ConversionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    half: bool = True


class PPDocLayoutV3Converter(ModelConverter):
    """Export the neural network only; SDK postprocessing stays runtime-neutral."""

    converter_id = "paddlepaddle.pp-doclayout-v3"

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def supports(self, request: ConversionRequest) -> bool:
        source = request.source.path
        return (
            request.model_id == self._model_id
            and request.source_format is ArtifactFormat.SAFETENSORS
            and request.target_format is ArtifactFormat.COREML
            and source.is_dir()
            and (source / "config.json").is_file()
            and (source / "model.safetensors").is_file()
            and (source / "preprocessor_config.json").is_file()
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if not self.supports(request):
            raise UnsupportedRuntimeError(f"{self.converter_id} does not support this conversion")
        options = PPDocLayoutV3ConversionOptions.model_validate(request.options)
        output = request.output_path.expanduser().resolve()
        if output.exists() and not request.overwrite:
            raise FileExistsError(f"Conversion output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.parent / f".{output.name}.convert-{uuid4().hex}"
        try:
            await asyncio.to_thread(self._convert, request.source.path, staging, options)
            if output.exists():
                if output.is_dir():
                    shutil.rmtree(output)
                else:
                    output.unlink()
            os.replace(staging, output)
            return ConversionResult(
                path=output,
                format=ArtifactFormat.COREML,
                digest=directory_sha256(output),
                size_bytes=directory_size(output),
                converter_id=self.converter_id,
                model_id=request.model_id,
                model_revision=request.model_revision,
                variant=request.variant,
                source_digest=request.source.digest,
                options=options.model_dump(mode="json"),
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(staging)

    def _convert(
        self, source: Path, output: Path, options: PPDocLayoutV3ConversionOptions
    ) -> None:
        try:
            torch = importlib.import_module("torch")
            coremltools = importlib.import_module("coremltools")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "PP-DocLayoutV3 conversion requires the SDK layout extra", cause=error
            ) from error

        class CoreMlExportWrapper(torch.nn.Module):  # type: ignore[name-defined,misc]
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.model = model

            def forward(self, pixel_values: Any) -> tuple[Any, Any, Any, Any]:
                result = self.model(pixel_values=pixel_values, return_dict=True)
                return result.logits, result.pred_boxes, result.order_logits, result.out_masks

        class FixedPositionEmbedding(torch.nn.Module):  # type: ignore[name-defined,misc]
            def __init__(self, value: Any) -> None:
                super().__init__()
                self.register_buffer("value", value)

            def forward(self, width: Any, height: Any, device: Any, dtype: Any) -> Any:
                del width, height, device, dtype
                return self.value

        class CoreMlDeformableAttention(torch.nn.Module):  # type: ignore[name-defined,misc]
            """Numerically equivalent attention with no rank-six intermediates."""

            def __init__(self, original: Any) -> None:
                super().__init__()
                self.original = original
                self.num_heads = original.n_heads
                self.num_levels = original.n_levels
                self.num_points = original.n_points
                self.model_width = original.d_model

            def forward(
                self,
                hidden_states: Any,
                attention_mask: Any = None,
                encoder_hidden_states: Any = None,
                encoder_attention_mask: Any = None,
                position_embeddings: Any = None,
                reference_points: Any = None,
                spatial_shapes: Any = None,
                spatial_shapes_list: Any = None,
                level_start_index: Any = None,
                **kwargs: Any,
            ) -> tuple[Any, Any]:
                del encoder_attention_mask, spatial_shapes, level_start_index, kwargs
                if position_embeddings is not None:
                    hidden_states = hidden_states + position_embeddings
                batch_size, num_queries, _ = hidden_states.shape
                _, sequence_length, _ = encoder_hidden_states.shape
                head_width = self.model_width // self.num_heads
                value = self.original.value_proj(encoder_hidden_states)
                if attention_mask is not None:
                    value = value.masked_fill(~attention_mask[..., None], 0.0)
                value = value.reshape(
                    batch_size, sequence_length, self.num_heads, head_width
                )
                point_count = self.num_levels * self.num_points
                offsets = self.original.sampling_offsets(hidden_states).reshape(
                    batch_size, num_queries, self.num_heads, point_count, 2
                )
                weights = torch.nn.functional.softmax(
                    self.original.attention_weights(hidden_states).reshape(
                        batch_size, num_queries, self.num_heads, point_count
                    ),
                    dim=-1,
                )
                centers = reference_points[:, :, :1, :2].expand(
                    -1, -1, self.num_levels, -1
                )
                sizes = reference_points[:, :, :1, 2:].expand(
                    -1, -1, self.num_levels, -1
                )
                centers = centers.repeat_interleave(self.num_points, dim=2)
                sizes = sizes.repeat_interleave(self.num_points, dim=2)
                sampling_grids = 2.0 * (
                    centers.unsqueeze(2)
                    + offsets / self.num_points * sizes.unsqueeze(2) * 0.5
                ) - 1.0
                split_sizes = [height * width for height, width in spatial_shapes_list]
                values = value.split(split_sizes, dim=1)
                sampled: list[Any] = []
                for level, (height, width) in enumerate(spatial_shapes_list):
                    level_value = (
                        values[level]
                        .flatten(2)
                        .transpose(1, 2)
                        .reshape(batch_size * self.num_heads, head_width, height, width)
                    )
                    start = level * self.num_points
                    level_grid = sampling_grids[
                        :, :, :, start : start + self.num_points
                    ]
                    level_grid = level_grid.transpose(1, 2).flatten(0, 1)
                    sampled.append(
                        torch.nn.functional.grid_sample(
                            level_value,
                            level_grid,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=False,
                        )
                    )
                sampled_values = torch.stack(sampled, dim=-2).reshape(
                    batch_size * self.num_heads,
                    head_width,
                    num_queries,
                    point_count,
                )
                flattened_weights = weights.transpose(1, 2).reshape(
                    batch_size * self.num_heads, 1, num_queries, point_count
                )
                result = (sampled_values * flattened_weights).sum(-1)
                result = result.reshape(
                    batch_size, self.num_heads * head_width, num_queries
                ).transpose(1, 2)
                return self.original.output_proj(result.contiguous()), weights

        model = transformers.AutoModelForObjectDetection.from_pretrained(
            str(source), local_files_only=True, trust_remote_code=False
        ).eval()
        # Core ML only supports tensors up to rank five. The upstream forward
        # builds constants dynamically and represents deformable sampling as a
        # rank-six tensor, so specialize both operations for the fixed 800px
        # checkpoint input before tracing. The arithmetic and weights are unchanged.
        spatial_shapes = tuple(
            (800 // stride, 800 // stride) for stride in model.config.feat_strides
        )
        anchors, valid_mask = model.model.generate_anchors(spatial_shapes)
        model.model.register_buffer("anchors", anchors)
        model.model.register_buffer("valid_mask", valid_mask)
        object.__setattr__(model.model.config, "anchor_image_size", [800, 800])
        modeling = importlib.import_module(
            "transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3"
        )
        for projection_index, aifi in zip(
            model.config.encode_proj_layers, model.model.encoder.aifi, strict=True
        ):
            stride = model.config.feat_strides[projection_index]
            position = modeling.build_2d_sinusoidal_position_embedding(
                height=800 // stride,
                width=800 // stride,
                embed_dim=model.config.encoder_hidden_dim,
                temperature=model.config.positional_encoding_temperature,
            ).unsqueeze(0)
            aifi.position_embedding = FixedPositionEmbedding(position)
        for layer in model.model.decoder.layers:
            layer.encoder_attn = CoreMlDeformableAttention(layer.encoder_attn)
        wrapper = CoreMlExportWrapper(model).eval()
        example = torch.zeros(1, 3, 800, 800, dtype=torch.float32)
        output.mkdir(parents=True)
        try:
            with torch.inference_mode():
                traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
                traced = torch.jit.freeze(traced)
            precision = (
                coremltools.precision.FLOAT16
                if options.half
                else coremltools.precision.FLOAT32
            )
            converted = coremltools.convert(
                traced,
                convert_to="mlprogram",
                inputs=[
                    coremltools.TensorType(
                        name="pixel_values", shape=(1, 3, 800, 800)
                    )
                ],
                outputs=[
                    coremltools.TensorType(name="logits"),
                    coremltools.TensorType(name="pred_boxes"),
                    coremltools.TensorType(name="order_logits"),
                    coremltools.TensorType(name="out_masks"),
                ],
                compute_precision=precision,
                minimum_deployment_target=coremltools.target.macOS14,
            )
            converted.user_defined_metadata.update(
                {
                    "hugging_mac_model": self._model_id,
                    "hugging_mac_input_size": "800",
                    "hugging_mac_raw_outputs": "true",
                    "hugging_mac_source": "PaddlePaddle/PP-DocLayoutV3_safetensors",
                }
            )
            converted.save(str(output / "model.mlpackage"))
            (output / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": self._model_id,
                        "input": {"name": "pixel_values", "shape": [1, 3, 800, 800]},
                        "outputs": ["logits", "pred_boxes", "order_logits", "out_masks"],
                        "precision": "float16" if options.half else "float32",
                        "postprocessing": "hugging_mac_sdk",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as error:
            raise UnsupportedRuntimeError(
                "PP-DocLayoutV3 Core ML conversion failed", cause=error
            ) from error
