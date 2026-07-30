"""Core ML conversion and runtime helpers for the Audio8 audio tower."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from .audio_encoder import (
    AudioEncoderLayer,
    feature_lengths_after_convolution,
)
from .modeling import Audio8AsrModel

AUDIO_BUCKET_FRAMES = (500, 1000, 3000)
AUDIO_BUCKET_TOKENS = {500: 65, 1000: 130, 3000: 390}


class CoreMlAudioTower(nn.Module):
    """Fixed-shape traceable audio encoder + MLP tower."""

    def __init__(self, model: Audio8AsrModel, bucket_frames: int) -> None:
        super().__init__()
        if bucket_frames not in AUDIO_BUCKET_FRAMES:
            raise ValueError(f"Unsupported Audio8-ASR Core ML bucket: {bucket_frames}")
        self.encoder = model.audio_encoder
        self.mlp_tower = model.audio_mlp_tower
        self.bucket_frames = bucket_frames
        self.chunk_frames = self.encoder.config.n_window * 2
        self.chunk_count = bucket_frames // self.chunk_frames
        self.token_count = AUDIO_BUCKET_TOKENS[bucket_frames]

    def forward(
        self,
        audios: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = audios[0].transpose(0, 1)
        chunks = features.reshape(
            self.chunk_count,
            self.chunk_frames,
            self.encoder.config.num_mel_bins,
        ).transpose(1, 2)
        hidden = chunks.unsqueeze(1)
        hidden = F.gelu(self.encoder.conv2d1(hidden))
        hidden = F.gelu(self.encoder.conv2d2(hidden))
        hidden = F.gelu(self.encoder.conv2d3(hidden))
        batch, channels, frequency, time = hidden.shape
        hidden = hidden.permute(0, 3, 1, 2).contiguous()
        hidden = self.encoder.conv_out(hidden.reshape(batch, time, channels * frequency))
        position = self.encoder.positional_embedding.weight[:time]
        hidden = hidden + position.unsqueeze(0).to(hidden.dtype)
        hidden = hidden.reshape(self.token_count, self.encoder.config.d_model)

        for raw_layer in self.encoder.layers:
            layer = cast(AudioEncoderLayer, raw_layer)
            residual = hidden
            normalized = layer.self_attn_layer_norm(hidden)
            attention = layer.self_attn
            query = (
                attention.q_proj(normalized)
                .reshape(
                    self.token_count,
                    attention.num_heads,
                    attention.head_dim,
                )
                .transpose(0, 1)
                .unsqueeze(0)
            )
            key = (
                attention.k_proj(normalized)
                .reshape(
                    self.token_count,
                    attention.num_heads,
                    attention.head_dim,
                )
                .transpose(0, 1)
                .unsqueeze(0)
            )
            value = (
                attention.v_proj(normalized)
                .reshape(
                    self.token_count,
                    attention.num_heads,
                    attention.head_dim,
                )
                .transpose(0, 1)
                .unsqueeze(0)
            )
            weights = torch.matmul(query, key.transpose(2, 3)) * attention.scaling
            weights = F.softmax(weights + attention_mask, dim=-1, dtype=torch.float32)
            attended = torch.matmul(weights.to(query.dtype), value)
            attended = attended.transpose(1, 2).reshape(
                self.token_count,
                self.encoder.config.d_model,
            )
            hidden = residual + attention.out_proj(attended)
            residual = hidden
            hidden = layer.final_layer_norm(hidden)
            activation = cast(
                Callable[[torch.Tensor], torch.Tensor],
                layer.activation,
            )
            hidden = layer.fc2(activation(layer.fc1(hidden)))
            hidden = residual + hidden

        hidden = self.encoder.ln_post(hidden)
        encoder_activation = cast(
            Callable[[torch.Tensor], torch.Tensor],
            self.encoder.activation,
        )
        hidden = self.encoder.proj2(encoder_activation(self.encoder.proj1(hidden)))
        return cast(torch.Tensor, self.mlp_tower(hidden))


def select_audio_bucket(feature_frames: int) -> int:
    for bucket in AUDIO_BUCKET_FRAMES:
        if feature_frames <= bucket:
            return bucket
    raise ValueError(f"Audio feature length exceeds 30 seconds: {feature_frames}")


def coreml_attention_mask(
    *,
    bucket_frames: int,
    valid_feature_frames: int,
    np: Any,
) -> tuple[Any, int]:
    """Create the fixed additive attention mask and valid encoded length."""

    total_tokens = AUDIO_BUCKET_TOKENS[bucket_frames]
    valid_tensor = torch.tensor([valid_feature_frames], dtype=torch.long)
    valid_tokens = int(feature_lengths_after_convolution(valid_tensor)[0].item())
    mask = np.full((1, 1, total_tokens, total_tokens), -3e4, dtype=np.float32)
    window_tokens = 104
    start = 0
    while start < valid_tokens:
        end = min(start + window_tokens, valid_tokens)
        mask[0, 0, start:end, start:end] = 0.0
        start = end
    for index in range(valid_tokens, total_tokens):
        mask[0, 0, index, index] = 0.0
    return mask, valid_tokens


def adaptive_pool_hidden(hidden: Any, output_size: int, np: Any) -> Any:
    input_size = int(hidden.shape[0])
    if input_size == output_size:
        return hidden
    rows = []
    for index in range(output_size):
        start = index * input_size // output_size
        end = ((index + 1) * input_size + output_size - 1) // output_size
        rows.append(hidden[start : max(end, start + 1)].mean(axis=0))
    return np.stack(rows, axis=0)


def project_audio_hidden(
    hidden: Any,
    *,
    norm_weight: Any,
    norm_bias: Any,
    linear_weight: Any,
    linear_bias: Any,
    np: Any,
) -> Any:
    mean = hidden.mean(axis=-1, keepdims=True)
    variance = ((hidden - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (hidden - mean) / np.sqrt(variance + 1e-5)
    normalized = normalized * norm_weight + norm_bias
    return normalized @ linear_weight.T + linear_bias
