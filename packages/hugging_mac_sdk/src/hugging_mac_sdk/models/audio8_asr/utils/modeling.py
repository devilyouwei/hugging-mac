"""Local Audio8-ASR architecture and greedy autoregressive decoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_model
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audio_encoder import AudioEncoder, AudioEncoderConfig


class ResidualMlpBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.fc1(self.norm(hidden_states))
        hidden_states = self.dropout(F.gelu(hidden_states))
        hidden_states = self.dropout(self.fc2(hidden_states))
        return residual + hidden_states


class AudioMlpTower(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        layers: int,
        intermediate_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ResidualMlpBlock(hidden_size, intermediate_size, dropout) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return cast(torch.Tensor, self.final_norm(hidden_states))


class Audio8AsrModel(nn.Module):
    """Audio encoder composed with a compact Qwen2 causal language model."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.audio_token_id = int(config["audio_token_id"])
        language_values = dict(config)
        language_values.pop("model_type", None)
        language_config = Qwen2Config(**language_values)
        self.language_model = Qwen2ForCausalLM(language_config)  # type: ignore[no-untyped-call]
        audio_config = AudioEncoderConfig.from_dict(config["qwen3_asr_audio_config"])
        self.audio_encoder = AudioEncoder(audio_config)
        audio_dim = audio_config.output_dim
        tower_hidden = int(config.get("qwen3_asr_mlp_tower_hidden_size") or audio_dim * 4)
        self.audio_mlp_tower = AudioMlpTower(
            audio_dim,
            layers=int(config.get("qwen3_asr_mlp_tower_layers", 4)),
            intermediate_size=tower_hidden,
            dropout=float(config.get("qwen3_asr_mlp_tower_dropout", 0.0)),
        )
        self.audio_projector = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, language_config.hidden_size),
        )

    def _project_audio(
        self,
        features: torch.Tensor,
        *,
        token_count: int,
    ) -> torch.Tensor:
        if features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.dim() != 2:
            raise ValueError(f"Expected [mel, frames] audio features, got {features.shape}")
        expected_mels = self.audio_encoder.config.num_mel_bins
        if features.size(0) != expected_mels and features.size(1) == expected_mels:
            features = features.transpose(0, 1)
        if features.size(0) != expected_mels:
            raise ValueError(f"Expected {expected_mels} mel bins, got {features.size(0)}")
        encoded = self.audio_encoder(
            features,
            torch.tensor(
                [features.size(1)],
                dtype=torch.long,
                device=features.device,
            ),
        )
        encoded = self.audio_mlp_tower(encoded)
        if int(encoded.size(0)) != token_count:
            pooling_input = encoded.transpose(0, 1).float().unsqueeze(0)
            if encoded.device.type == "mps" and int(encoded.size(0)) % token_count != 0:
                # MPS does not yet implement non-divisible adaptive pooling.
                # This tensor is at most a few MB and returning it to MPS keeps
                # the expensive encoder and language model on the GPU.
                pooling_input = pooling_input.cpu()
            encoded = (
                F.adaptive_avg_pool1d(
                    pooling_input,
                    output_size=token_count,
                )
                .squeeze(0)
                .transpose(0, 1)
                .to(device=encoded.device, dtype=encoded.dtype)
            )
        return cast(torch.Tensor, self.audio_projector(encoded))

    def generate_transcript(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        embeddings = self.language_model.model.embed_tokens(input_ids).clone()
        audio_positions = torch.nonzero(
            input_ids[0].eq(self.audio_token_id),
            as_tuple=False,
        ).flatten()
        if not int(audio_positions.numel()):
            raise ValueError("Prompt contains no audio placeholder tokens")
        projected = self._project_audio(
            input_features[0],
            token_count=int(audio_positions.numel()),
        )
        embeddings[0, audio_positions, :] = projected.to(embeddings.dtype)

        output = self.language_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        generated: list[torch.Tensor] = []
        current_mask = attention_mask
        for _ in range(max_new_tokens):
            token = output.logits[:, -1, :].argmax(dim=-1)
            generated.append(token)
            if bool(token.eq(eos_token_id).all()):
                break
            current_mask = torch.cat(
                (
                    current_mask,
                    torch.ones(
                        (current_mask.size(0), 1),
                        dtype=current_mask.dtype,
                        device=current_mask.device,
                    ),
                ),
                dim=1,
            )
            output = self.language_model(
                input_ids=token[:, None],
                attention_mask=current_mask,
                past_key_values=output.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        if not generated:
            return input_ids.new_empty((input_ids.size(0), 0))
        return torch.stack(generated, dim=1)


def load_audio8_asr_model(
    artifact: Path,
    *,
    device: str,
    dtype: torch.dtype,
) -> Audio8AsrModel:
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    original_dtype = torch.get_default_dtype()
    try:
        # Construct directly in the selected inference dtype. Building this
        # checkpoint in FP32 first would add roughly 650 MB of transient memory.
        torch.set_default_dtype(dtype)
        model = Audio8AsrModel(config)
    finally:
        torch.set_default_dtype(original_dtype)
    missing, unexpected = load_model(
        model,
        artifact / "model.safetensors",
        strict=False,
        device="cpu",
    )
    allowed_missing = {"language_model.lm_head.weight"}
    actual_missing = set(missing) - allowed_missing
    if actual_missing or unexpected:
        raise RuntimeError(
            "Audio8-ASR weight structure does not match the local implementation: "
            f"missing={sorted(actual_missing)}, unexpected={sorted(unexpected)}"
        )
    model.to(device=device, dtype=dtype)
    return model.eval()
