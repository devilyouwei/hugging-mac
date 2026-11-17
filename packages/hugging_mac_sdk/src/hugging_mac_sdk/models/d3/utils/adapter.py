"""Core AI neural execution with the pinned upstream D3 request/response contract.

Original rendering, image/video decoding, resizing, sampling, question validation,
permutation averaging and response formatting are reused without modification.
Torch is used for host tensor preparation, never for neural inference here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import torch

from .preprocess import position_inputs
from .upstream import load_upstream


def create_runtime(
    root: Path, decision: Any, vision: Any, *, model_name: str, permutation_average: bool = False
) -> Any:
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    upstream = load_upstream(root)
    config = AutoConfig.from_pretrained(root)

    class D3CoreAI(upstream.D3):  # type: ignore[name-defined, misc]
        def __init__(self) -> None:
            self.root = root
            self.torch = torch
            self.config = json.loads((root / "decision_config.json").read_text())
            self.codes = self.config["codes"]
            self.token_ids = self.config["token_ids"]
            self.temperature = self.config["temperature"]
            self.prompt = self.config["prompt"]
            self.attention_mode = self.config["attention_mode"]
            self.max_length = self.config.get("max_length")
            self.readout_dtype = self.config["readout_dtype"]
            self.permutation_average = permutation_average
            self.batch_size = 1
            self.model_name = model_name
            self.device = "cpu"
            self.fast = None
            self.tokenizer = AutoTokenizer.from_pretrained(root)
            self.tokenizer.padding_side = "left"
            self.pad_id = self.tokenizer.pad_token_id
            self.processor = upstream.load_processor(root)
            upstream.configure_video_processor(self.processor)
            self.image_unavailable = self.video_unavailable = None
            self._position_model: Any = SimpleNamespace(config=config)
            self._position_model.get_vision_position_ids = MethodType(
                Qwen3_5Model.get_vision_position_ids, self._position_model
            )
            vc = config.vision_config
            self._visual = SimpleNamespace(
                config=vc,
                spatial_merge_size=vc.spatial_merge_size,
                num_grid_per_side=int(vc.num_position_embeddings**0.5),
                interpolation_mode="bilinear",
                interpolation_align_corners=True,
            )

        def _vision_features(self, pixels: Any, grids: Any) -> list[Any]:
            features = []
            offset = 0
            for t, h, w in grids.tolist():
                groups = []
                for _ in range(t):
                    group = pixels[offset : offset + h * w].float()
                    offset += h * w
                    indices, weights, positions = position_inputs(
                        self._visual, torch.tensor([[1, h, w]])
                    )
                    inputs = dict(
                        zip(
                            ("pixels", "indices", "weights", "positions"),
                            (group.numpy(), indices.numpy(), weights.numpy(), positions.numpy()),
                            strict=True,
                        )
                    )
                    values = vision.run(inputs)["features"]
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Nonfinite Core AI vision features")
                    groups.append(torch.from_numpy(values.copy()))
                features.append(torch.cat(groups))
            if offset != len(pixels):
                raise ValueError("Vision patch inventory does not match the upstream grid")
            return features

        def _decision(
            self,
            ids: Any,
            mask: Any,
            types: Any = None,
            images: Any = None,
            videos: Any = None,
            features: Any = None,
        ) -> Any:
            width = ids.shape[1]
            visual = np.zeros((1, width, config.text_config.hidden_size), dtype=np.float32)
            selected = np.zeros((1, width, 1), dtype=np.bool_)
            if features:
                for kind, token in (
                    ("image", config.image_token_id),
                    ("video", config.video_token_id),
                ):
                    if kind not in features:
                        continue
                    slots = (ids[0] == token).numpy()
                    values = torch.cat(features[kind]).float().numpy()
                    if int(slots.sum()) != len(values):
                        raise ValueError("Visual token count differs from the original processor")
                    visual[0, slots] = values
                    selected[0, slots] = True
                positions, _ = Qwen3_5Model.get_rope_index(
                    self._position_model,
                    ids,
                    types,
                    image_grid_thw=images,
                    video_grid_thw=videos,
                    attention_mask=mask,
                )
                positions = torch.cat((torch.arange(width)[None, None], positions))
            else:
                positions = torch.arange(width).reshape(1, 1, width).expand(4, 1, width)
            inputs = {
                "input_ids": ids.numpy().astype(np.int32),
                "attention_mask": mask.numpy().astype(np.int32),
                "position_ids": positions.numpy().astype(np.int32),
                "visual_embeddings": visual,
                "visual_mask": selected,
            }
            logits = decision.run(inputs)["logits"]
            if not np.isfinite(logits).all():
                raise FloatingPointError("Nonfinite Core AI decision logits")
            return torch.from_numpy(logits.copy())

        def logits(self, sequences: Any, counts: Any) -> Any:
            rows = []
            for tokens, count in zip(sequences, counts, strict=True):
                ids = torch.tensor([tokens])
                row = self._decision(ids, torch.ones_like(ids))
                row[:, count:] = -torch.inf
                rows.append(row)
            return torch.cat(rows)

        def image_logits(self, texts: Any, images: Any, counts: Any, width: int) -> Any:
            rows = []
            for text, count in zip(texts, counts, strict=True):
                encoded = self.processor(
                    text=[text], images=images, padding=True, return_tensors="pt"
                )
                features = {
                    "image": self._vision_features(
                        encoded["pixel_values"], encoded["image_grid_thw"]
                    )
                }
                row = self._decision(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                    encoded["mm_token_type_ids"],
                    images=encoded["image_grid_thw"],
                    features=features,
                )
                row[:, count:] = -torch.inf
                rows.append(row)
            return torch.cat(rows)

        def media_features(self, media: Any) -> Any:
            features = {}
            for kind, pixels, grid in (
                ("image", "pixel_values", "image_grid_thw"),
                ("video", "pixel_values_videos", "video_grid_thw"),
            ):
                if pixels in media:
                    features[kind] = self._vision_features(media[pixels], media[grid])
            return features

        def video_logits(
            self, texts: Any, prepared: Any, features: Any, counts: Any, width: int
        ) -> Any:
            media = prepared.media
            image_texts = [
                self.processor.replace_image_token(media, image_idx=i)
                for i in range(len(prepared.images))
            ]
            rows = []
            for text, count in zip(texts, counts, strict=True):
                expanded = self.processor.get_text_with_replacements(
                    [text], image_texts, prepared.video_texts
                )[0][0]
                encoded = self.processor.tokenizer(
                    [expanded], padding=True, return_token_type_ids=False, return_tensors="pt"
                )
                ids = encoded["input_ids"]
                types = torch.tensor(self.processor.create_mm_token_type_ids(ids))
                row = self._decision(
                    ids,
                    encoded["attention_mask"],
                    types,
                    images=media.get("image_grid_thw"),
                    videos=media["video_grid_thw"],
                    features=features,
                )
                row[:, count:] = -torch.inf
                rows.append(row)
            return torch.cat(rows)

    return D3CoreAI()
