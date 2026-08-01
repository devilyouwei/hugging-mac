"""Local SenseVoiceSmall SANM encoder and CTC inference implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPositionEncoder(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, timesteps, depth = inputs.shape
        positions = torch.arange(
            1,
            timesteps + 1,
            device=inputs.device,
            dtype=inputs.dtype,
        )[None, :]
        increment = torch.log(
            torch.tensor(10000.0, dtype=inputs.dtype, device=inputs.device)
        ) / (depth / 2 - 1)
        inverse = torch.exp(
            torch.arange(
                depth // 2,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            * -increment
        )
        scaled = positions[:, :, None] * inverse[None, None, :]
        encoding = torch.cat((torch.sin(scaled), torch.cos(scaled)), dim=2)
        return inputs + encoding


class PositionwiseFeedForward(nn.Module):
    def __init__(self, size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.w_2(self.dropout(self.activation(self.w_1(inputs)))),
        )


class MultiHeadedAttentionSanm(nn.Module):
    def __init__(
        self,
        heads: int,
        input_size: int,
        output_size: int,
        dropout: float,
        kernel_size: int,
        shift: int = 0,
    ) -> None:
        super().__init__()
        if output_size % heads:
            raise ValueError("SenseVoice attention size must be divisible by its heads")
        self.d_k = output_size // heads
        self.h = heads
        self.linear_out = nn.Linear(output_size, output_size)
        self.linear_q_k_v = nn.Linear(input_size, output_size * 3)
        self.dropout = nn.Dropout(dropout)
        self.fsmn_block = nn.Conv1d(
            output_size,
            output_size,
            kernel_size,
            stride=1,
            padding=0,
            groups=output_size,
            bias=False,
        )
        left = (kernel_size - 1) // 2 + max(shift, 0)
        self.pad_fn = nn.ConstantPad1d((left, kernel_size - 1 - left), 0.0)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, timesteps, _ = inputs.shape
        qkv = self.linear_q_k_v(inputs)
        query, key, value = torch.split(qkv, self.h * self.d_k, dim=-1)

        def split_heads(item: torch.Tensor) -> torch.Tensor:
            return item.reshape(batch, timesteps, self.h, self.d_k).transpose(1, 2)

        query_heads = split_heads(query) * self.d_k**-0.5
        key_heads = split_heads(key)
        value_heads = split_heads(value)
        scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
        attention_mask = mask.unsqueeze(1).eq(0)
        scores = scores.masked_fill(attention_mask, -float("inf"))
        attention = torch.softmax(scores, dim=-1).masked_fill(attention_mask, 0.0)
        attended = torch.matmul(self.dropout(attention), value_heads)
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch,
            timesteps,
            self.h * self.d_k,
        )

        memory_mask = mask.reshape(batch, -1, 1)
        memory_input = value * memory_mask
        memory = self.fsmn_block(self.pad_fn(memory_input.transpose(1, 2)))
        memory = memory.transpose(1, 2) + memory_input
        memory = self.dropout(memory) * memory_mask
        return cast(torch.Tensor, self.linear_out(attended) + memory)


class FloatLayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(inputs.dtype)


class EncoderLayerSanm(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        heads: int,
        linear_units: int,
        dropout: float,
        attention_dropout: float,
        kernel_size: int,
        shift: int,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadedAttentionSanm(
            heads,
            input_size,
            output_size,
            attention_dropout,
            kernel_size,
            shift,
        )
        self.feed_forward = PositionwiseFeedForward(output_size, linear_units, dropout)
        self.norm1 = FloatLayerNorm(input_size)
        self.norm2 = FloatLayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        self.in_size = input_size
        self.size = output_size
        self.normalize_before = True
        self.concat_after = False
        self.stochastic_depth_rate = 0.0
        self.dropout_rate = dropout

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = inputs
        hidden = self.norm1(inputs)
        attended = self.dropout(self.self_attn(hidden, mask))
        hidden = residual + attended if self.in_size == self.size else attended
        residual = hidden
        hidden = residual + self.dropout(self.feed_forward(self.norm2(hidden)))
        return hidden, mask


class SenseVoiceEncoderSmall(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        output_size: int = 512,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 50,
        tp_blocks: int = 20,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        **_: Any,
    ) -> None:
        super().__init__()
        self._output_size = output_size
        self.embed = SinusoidalPositionEncoder()
        def layer(layer_input_size: int) -> EncoderLayerSanm:
            return EncoderLayerSanm(
                layer_input_size,
                output_size,
                heads=attention_heads,
                linear_units=linear_units,
                dropout=dropout_rate,
                attention_dropout=attention_dropout_rate,
                kernel_size=kernel_size,
                shift=sanm_shfit,
            )

        self.encoders0 = nn.ModuleList(
            [layer(input_size)]
        )
        self.encoders = nn.ModuleList(
            layer(output_size) for _ in range(num_blocks - 1)
        )
        self.tp_encoders = nn.ModuleList(
            layer(output_size) for _ in range(tp_blocks)
        )
        self.after_norm = FloatLayerNorm(output_size)
        self.tp_norm = FloatLayerNorm(output_size)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(features.shape[1], device=lengths.device)
        mask = (positions[None, :] < lengths[:, None])[:, None, :].to(features.dtype)
        hidden = self.embed(features * self._output_size**0.5)
        for raw_layer in self.encoders0:
            hidden, mask = cast(EncoderLayerSanm, raw_layer)(hidden, mask)
        for raw_layer in self.encoders:
            hidden, mask = cast(EncoderLayerSanm, raw_layer)(hidden, mask)
        hidden = self.after_norm(hidden)
        output_lengths = mask.squeeze(1).sum(1).int()
        for raw_layer in self.tp_encoders:
            hidden, mask = cast(EncoderLayerSanm, raw_layer)(hidden, mask)
        return cast(torch.Tensor, self.tp_norm(hidden)), output_lengths


class CtcHead(nn.Module):
    def __init__(self, input_size: int, vocabulary_size: int) -> None:
        super().__init__()
        self.ctc_lo = nn.Linear(input_size, vocabulary_size)


class SenseVoiceSmallModel(nn.Module):
    """Inference-only SenseVoiceSmall graph with compatible checkpoint names."""

    LANGUAGE_IDS: ClassVar[dict[str, int]] = {
        "auto": 0,
        "zh": 3,
        "en": 4,
        "yue": 7,
        "ja": 11,
        "ko": 12,
        "nospeech": 13,
    }
    TEXT_NORMALIZATION_IDS: ClassVar[dict[str, int]] = {
        "withitn": 14,
        "woitn": 15,
    }
    UNKNOWN_EMOTION_TOKEN_ID = 25009

    def __init__(
        self,
        *,
        input_size: int,
        vocabulary_size: int,
        encoder_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.blank_id = 0
        self.encoder = SenseVoiceEncoderSmall(input_size, **encoder_config)
        self.ctc = CtcHead(512, vocabulary_size)
        self.embed = nn.Embedding(16, input_size)

    def infer_tokens(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        *,
        language: str,
        use_itn: bool,
        ban_unknown_emotion: bool,
    ) -> torch.Tensor:
        batch = features.shape[0]
        language_id = self.LANGUAGE_IDS.get(language, 0)
        language_query = self.embed(
            torch.tensor([[language_id]], dtype=torch.long, device=features.device)
        ).repeat(batch, 1, 1)
        event_emotion_query = self.embed(
            torch.tensor([[1, 2]], dtype=torch.long, device=features.device)
        ).repeat(batch, 1, 1)
        style_id = self.TEXT_NORMALIZATION_IDS["withitn" if use_itn else "woitn"]
        style_query = self.embed(
            torch.tensor([[style_id]], dtype=torch.long, device=features.device)
        ).repeat(batch, 1, 1)
        hidden = torch.cat(
            (language_query, event_emotion_query, style_query, features),
            dim=1,
        )
        encoded, encoded_lengths = self.encoder(hidden, lengths + 4)
        logits = F.log_softmax(self.ctc.ctc_lo(encoded), dim=-1)
        if ban_unknown_emotion:
            logits[:, :, self.UNKNOWN_EMOTION_TOKEN_ID] = -float("inf")
        predicted = logits[0, : int(encoded_lengths[0].item())].argmax(dim=-1)
        predicted = torch.unique_consecutive(predicted.detach().cpu())
        return cast(torch.Tensor, predicted[predicted.ne(self.blank_id)])


def load_sensevoice_model(
    artifact: Path,
    *,
    vocabulary_size: int,
    config: Mapping[str, Any],
    device: str,
    dtype: torch.dtype,
) -> SenseVoiceSmallModel:
    model = SenseVoiceSmallModel(
        input_size=560,
        vocabulary_size=vocabulary_size,
        encoder_config=cast(Mapping[str, Any], config["encoder_conf"]),
    )
    checkpoint = torch.load(
        artifact / "model.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = _state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "SenseVoiceSmall checkpoint does not match the local implementation: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device=device, dtype=dtype)
    return model.eval()


def _state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("SenseVoiceSmall model.pt does not contain a state dictionary")
    for key in ("state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            checkpoint = nested
            break
    state: dict[str, torch.Tensor] = {}
    for raw_name, value in checkpoint.items():
        if not isinstance(raw_name, str) or not isinstance(value, torch.Tensor):
            continue
        name = raw_name.removeprefix("module.")
        state[name] = value
    if not state:
        raise TypeError("SenseVoiceSmall model.pt contains no tensor weights")
    return state
