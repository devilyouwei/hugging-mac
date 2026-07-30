"""Qwen3-ASR-compatible audio encoder used by Audio8-ASR.

The encoder architecture follows the Apache-2.0 Qwen3-ASR implementation, but
is kept model-private so downloaded snapshots never need executable remote code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN


@dataclass(frozen=True, slots=True)
class AudioEncoderConfig:
    num_mel_bins: int = 128
    encoder_layers: int = 18
    encoder_attention_heads: int = 14
    encoder_ffn_dim: int = 3584
    d_model: int = 896
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    scale_embedding: bool = False
    max_source_positions: int = 1500
    n_window: int = 50
    output_dim: int = 1024
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> AudioEncoderConfig:
        defaults = cls()
        return cls(
            num_mel_bins=int(values.get("num_mel_bins", defaults.num_mel_bins)),
            encoder_layers=int(values.get("encoder_layers", defaults.encoder_layers)),
            encoder_attention_heads=int(
                values.get("encoder_attention_heads", defaults.encoder_attention_heads)
            ),
            encoder_ffn_dim=int(values.get("encoder_ffn_dim", defaults.encoder_ffn_dim)),
            d_model=int(values.get("d_model", defaults.d_model)),
            dropout=float(values.get("dropout", defaults.dropout)),
            attention_dropout=float(values.get("attention_dropout", defaults.attention_dropout)),
            activation_function=str(
                values.get("activation_function", defaults.activation_function)
            ),
            scale_embedding=bool(values.get("scale_embedding", defaults.scale_embedding)),
            max_source_positions=int(
                values.get("max_source_positions", defaults.max_source_positions)
            ),
            n_window=int(values.get("n_window", defaults.n_window)),
            output_dim=int(values.get("output_dim", defaults.output_dim)),
            n_window_infer=int(values.get("n_window_infer", defaults.n_window_infer)),
            conv_chunksize=int(values.get("conv_chunksize", defaults.conv_chunksize)),
            downsample_hidden_size=int(
                values.get("downsample_hidden_size", defaults.downsample_hidden_size)
            ),
        )


def feature_lengths_after_convolution(lengths: torch.Tensor) -> torch.Tensor:
    lengths = torch.clamp(lengths.long(), min=1)
    remainder = lengths % 100
    first = (remainder - 1) // 2 + 1
    return ((first - 1) // 2 + 1 - 1) // 2 + 1 + (lengths // 100) * 13


def block_diagonal_mask(
    hidden_states: torch.Tensor,
    cumulative_lengths: torch.Tensor,
) -> torch.Tensor:
    sequence_length = int(hidden_states.shape[0])
    mask = torch.full(
        (1, 1, sequence_length, sequence_length),
        torch.finfo(hidden_states.dtype).min,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for index in range(1, int(cumulative_lengths.numel())):
        start = int(cumulative_lengths[index - 1].item())
        end = int(cumulative_lengths[index].item())
        mask[..., start:end, start:end] = 0
    return mask


class AudioSelfAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("d_model must be divisible by encoder_attention_heads")
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cumulative_lengths: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = int(hidden_states.size(0))

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)

        query = split_heads(self.q_proj(hidden_states)).unsqueeze(0)
        key = split_heads(self.k_proj(hidden_states)).unsqueeze(0)
        value = split_heads(self.v_proj(hidden_states)).unsqueeze(0)
        weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        weights = weights + block_diagonal_mask(hidden_states, cumulative_lengths)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().reshape(sequence_length, self.embed_dim)
        return cast(torch.Tensor, self.out_proj(output))


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.self_attn = AudioSelfAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.activation = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cumulative_lengths: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, cumulative_lengths)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            limit = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-limit, max=limit)
        return hidden_states


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, length: int, channels: int, max_timescale: int = 10000) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("Position embedding channels must be even")
        self.weight: torch.Tensor
        increment = math.log(max_timescale) / (channels // 2 - 1)
        inverse_timescales = torch.exp(-increment * torch.arange(channels // 2).float())
        scaled_time = torch.outer(torch.arange(length).float(), inverse_timescales)
        self.register_buffer(
            "weight",
            torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=1),
            persistent=False,
        )


class AudioEncoder(nn.Module):
    """Convolutional downsampler plus windowed transformer audio encoder."""

    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.positional_embedding = SinusoidalPositionEmbedding(
            config.max_source_positions,
            embed_dim,
        )
        self.layers = nn.ModuleList(AudioEncoderLayer(config) for _ in range(config.encoder_layers))
        self.ln_post = nn.LayerNorm(embed_dim)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        convolution_frequency = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * convolution_frequency,
            embed_dim,
            bias=False,
        )
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.activation = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(embed_dim, config.output_dim)

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> torch.Tensor:
        feature_lengths = feature_lengths.to(device=input_features.device, dtype=torch.long)
        encoded_lengths = feature_lengths_after_convolution(feature_lengths)
        window = self.config.n_window * 2
        chunk_counts = torch.ceil(feature_lengths / window).long()
        chunk_lengths = torch.full(
            (int(chunk_counts.sum().item()),),
            window,
            dtype=torch.long,
            device=input_features.device,
        )
        tail_indexes = F.pad(chunk_counts, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_indexes] = feature_lengths % window
        chunk_lengths[chunk_lengths == 0] = window

        chunks = input_features.T.split(  # type: ignore[no-untyped-call]
            chunk_lengths.tolist(),
            dim=0,
        )
        padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2)
        lengths_after_convolution = feature_lengths_after_convolution(chunk_lengths)
        valid = nn.utils.rnn.pad_sequence(
            [
                torch.ones(int(length.item()), dtype=torch.bool, device=padded.device)
                for length in lengths_after_convolution
            ],
            batch_first=True,
        )
        padded = padded.unsqueeze(1)
        convolution_outputs = []
        for chunk in padded.split(  # type: ignore[no-untyped-call]
            self.config.conv_chunksize,
            dim=0,
        ):
            chunk = F.gelu(self.conv2d1(chunk))
            chunk = F.gelu(self.conv2d2(chunk))
            convolution_outputs.append(F.gelu(self.conv2d3(chunk)))
        embedded = torch.cat(convolution_outputs, dim=0)
        batch, channels, frequency, time = embedded.size()
        embedded = embedded.permute(0, 3, 1, 2).contiguous()
        embedded = self.conv_out(embedded.view(batch, time, channels * frequency))
        position = self.positional_embedding.weight[: embedded.shape[1]]
        embedded = embedded + position.unsqueeze(0).to(embedded.dtype)
        hidden_states = embedded[valid]

        cumulative_chunks = [0]
        attention_window = valid.shape[-1] * (
            self.config.n_window_infer // (self.config.n_window * 2)
        )
        for length in encoded_lengths:
            integer_length = int(length.item())
            cumulative_chunks.extend([attention_window] * (integer_length // attention_window))
            remainder = integer_length % attention_window
            if remainder:
                cumulative_chunks.append(remainder)
        cumulative_lengths = torch.tensor(
            cumulative_chunks,
            device=encoded_lengths.device,
        ).cumsum(-1, dtype=torch.int32)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cumulative_lengths)
        hidden_states = self.ln_post(hidden_states)
        return cast(
            torch.Tensor,
            self.proj2(self.activation(self.proj1(hidden_states))),
        )
