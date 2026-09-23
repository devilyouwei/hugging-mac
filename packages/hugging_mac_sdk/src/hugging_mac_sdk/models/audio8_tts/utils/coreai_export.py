"""Tensor-only export adapters; optional PyTorch imports stay inside factories."""

from __future__ import annotations

import importlib
from typing import Any


def fast_step_module(model: Any) -> Any:
    """Export one fast-AR step with explicit per-request KV tensors."""
    torch = importlib.import_module("torch")

    class FastStep(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.layers = model.fast_layers
            self.norm = model.fast_norm
            self.output = model.fast_output
            self.register_buffer("rope", model.fast_freqs_cis.float())

        def forward(self, hidden: Any, position: Any, keys: Any, values: Any) -> Any:
            new_keys, new_values = [], []
            rope = self.rope[position].reshape(1, 1, 1, -1, 2)
            for index, layer in enumerate(self.layers):
                attention = layer.attention
                x = layer.attention_norm(hidden)
                query_size = attention.n_head * attention.head_dim
                kv_size = attention.n_local_heads * attention.head_dim
                query, key, value = attention.wqkv(x).split((query_size, kv_size, kv_size), -1)
                query = query.reshape(1, 1, attention.n_head, attention.head_dim)
                key = key.reshape(1, 1, attention.n_local_heads, attention.head_dim)
                value = value.reshape(1, 1, attention.n_local_heads, attention.head_dim)
                if attention.qk_norm:
                    query, key = attention.q_norm(query), attention.k_norm(key)

                def rotate(tensor: Any) -> Any:
                    paired = tensor.float().reshape(1, 1, tensor.shape[2], -1, 2)
                    return (
                        torch.stack(
                            (
                                paired[..., 0] * rope[..., 0] - paired[..., 1] * rope[..., 1],
                                paired[..., 1] * rope[..., 0] + paired[..., 0] * rope[..., 1],
                            ),
                            -1,
                        )
                        .flatten(3)
                        .to(tensor.dtype)
                        .transpose(1, 2)
                    )

                query, key = rotate(query), rotate(key)
                value = value.transpose(1, 2)
                updated_k = keys[index].index_copy(2, position, key)
                updated_v = values[index].index_copy(2, position, value)
                new_keys.append(updated_k)
                new_values.append(updated_v)
                repeats = attention.n_head // attention.n_local_heads
                k = updated_k.repeat_interleave(repeats, dim=1)
                v = updated_v.repeat_interleave(repeats, dim=1)
                scores = query @ k.transpose(-2, -1) / attention.head_dim**0.5
                mask = torch.arange(k.shape[2], device=hidden.device) <= position[0]
                probabilities = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
                attended = (probabilities @ v).transpose(1, 2).reshape(1, 1, query_size)
                hidden = hidden + attention.wo(attended)
                hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))
            return (
                self.output(self.norm(hidden))[:, -1],
                torch.stack(new_keys),
                torch.stack(new_values),
            )

    return FastStep().eval()


def codec_decoder_module(codec: Any) -> Any:
    """Export only codec decoding, replacing complex-valued RoPE construction."""
    torch = importlib.import_module("torch")
    original = codec.quantizer.post_module

    class PostTransformer(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.layers = original.layers
            self.norm = original.norm
            self.input_proj = original.input_proj
            self.output_proj = original.output_proj
            self.look_ahead_conv = original.look_ahead_conv

        def forward(self, x: Any) -> Any:
            x = self.look_ahead_conv(self.input_proj(x.transpose(1, 2)))
            length = x.shape[1]
            positions = torch.arange(length, device=x.device)
            row, column = positions[:, None], positions[None, :]
            mask = column <= row
            if original.window_size is not None:
                mask = mask & (column >= (row - original.window_size + 1).clamp_min(0))
            frequencies = 1.0 / (
                original.rope_base
                ** (
                    torch.arange(0, original.head_dim, 2, device=x.device).float()
                    / original.head_dim
                )
            )
            phases = torch.outer(positions.float(), frequencies)
            rope = torch.stack((torch.cos(phases), torch.sin(phases)), -1).to(torch.bfloat16)
            for layer in self.layers:
                x = layer(x, rope, mask[None, None])
            return self.output_proj(self.norm(x)).transpose(1, 2)

    class CausalConv(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self, original_conv: Any) -> None:
            super().__init__()
            self.conv = original_conv.conv
            self.padding = original_conv.padding
            self.stride = original_conv.stride

        def forward(self, x: Any) -> Any:
            # ceil(length / stride) expressed using integer shape arithmetic.
            right = (-x.shape[-1]) % self.stride
            return self.conv(torch.nn.functional.pad(x, (self.padding, right))).contiguous()

    def exportable(module: Any) -> Any:
        if type(module).__name__ == "ArkttsCausalConv1d":
            return CausalConv(module)
        cloned = object.__new__(type(module))
        cloned.__dict__ = module.__dict__.copy()
        cloned._modules = {name: exportable(child) for name, child in module._modules.items()}
        return cloned

    class Decoder(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.semantic = codec.quantizer.semantic_quantizer
            self.residual = codec.quantizer.quantizer
            self.post = PostTransformer()
            self.upsample = exportable(codec.quantizer.upsample)
            self.decoder = exportable(codec.decoder)

        def forward(self, codes: Any) -> Any:
            semantic = self.semantic.from_codes(
                codes[:, :1].clamp(0, self.semantic.codebook_size - 1)
            )
            residual = self.residual.from_codes(
                codes[:, 1:].clamp(0, self.residual.codebook_size - 1)
            )
            return self.decoder(self.upsample(self.post(semantic + residual)))

    return Decoder().eval()
