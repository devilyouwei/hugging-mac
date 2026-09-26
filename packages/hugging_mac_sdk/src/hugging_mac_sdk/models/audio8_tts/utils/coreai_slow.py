"""Stateful Falcon-H1 export; framework imports are conversion-only."""

from __future__ import annotations

import importlib
from typing import Any


def slow_step_module(model: Any, capacity: int) -> Any:
    """A recurrent token step for prompt ingestion and decoding."""
    torch = importlib.import_module("torch")
    config = model.config

    class SlowStep(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = model.embeddings
            self.codebooks = model.codebook_embeddings
            self.layers = model.slow.layers
            self.norm = model.slow.final_layernorm
            self.output = model.semantic_output
            self.project = model.fast_project_in
            self.register_buffer("inv_freq", model.slow.rotary_emb.inv_freq.clone())
            n = len(self.layers)
            m = self.layers[0].mamba
            self.register_buffer(
                "keys", torch.zeros(n, 1, config.num_key_value_heads, capacity, config.head_dim)
            )
            self.register_buffer("values", torch.zeros_like(self.keys))
            self.register_buffer("conv", torch.zeros(n, 1, m.conv_dim, m.conv_kernel_size))
            self.register_buffer(
                "ssm", torch.zeros(n, 1, m.num_heads, m.head_dim, m.ssm_state_size)
            )

        def forward(self, ids: Any, position: Any) -> Any:
            hidden = self.embedding(ids[:, 0])
            offset = torch.arange(config.num_codebooks, device=ids.device) * config.codebook_size
            code = self.codebooks(ids[:, 1:, 0] + offset).sum(1)[:, None]
            semantic = (ids[:, :1, 0] >= config.semantic_begin_id) & (
                ids[:, :1, 0] <= config.semantic_end_id
            )
            hidden = (
                hidden + torch.where(semantic[..., None], code, 0.0)
            ) * model.slow.embedding_multiplier
            phase = position.float().reshape(1, 1, 1) * self.inv_freq.reshape(1, 1, -1)
            phase = torch.cat((phase, phase), -1)
            cos = phase.cos()[:, None] * model.slow.rotary_emb.attention_scaling
            sin = phase.sin()[:, None] * model.slow.rotary_emb.attention_scaling
            mask = torch.arange(capacity, device=ids.device) <= position[0]
            next_keys, next_values, next_conv, next_ssm = [], [], [], []
            for i, layer in enumerate(self.layers):
                x = layer.input_layernorm(hidden)
                m = layer.mamba
                projected = m.in_proj(x * m.ssm_in_multiplier) * m.mup_vector
                gate, conv_input, dt = projected.split(
                    (m.intermediate_size, m.conv_dim, m.num_heads), -1
                )
                conv = torch.cat((self.conv[i][..., 1:], conv_input.transpose(1, 2)), -1)
                mixed = (conv * m.conv1d.weight.squeeze(1)).sum(-1)
                if m.use_conv_bias:
                    mixed = mixed + m.conv1d.bias
                mixed = torch.nn.functional.silu(mixed)
                u, b, c = mixed.split(
                    (
                        m.intermediate_size,
                        m.n_groups * m.ssm_state_size,
                        m.n_groups * m.ssm_state_size,
                    ),
                    -1,
                )
                dt = torch.nn.functional.softplus(dt[:, 0] + m.dt_bias)
                a = -torch.exp(m.A_log.float())
                decay = torch.exp(dt * a)[..., None, None]
                b = b.reshape(1, m.n_groups, m.ssm_state_size).repeat_interleave(
                    m.num_heads // m.n_groups, 1
                )
                c = c.reshape(1, m.n_groups, m.ssm_state_size).repeat_interleave(
                    m.num_heads // m.n_groups, 1
                )
                u = u.reshape(1, m.num_heads, m.head_dim)
                state = self.ssm[i] * decay + (dt[..., None, None] * b[..., None, :]) * u[..., None]
                y = (state @ c[..., None]).squeeze(-1) + u * m.D[None, :, None]
                y = y.reshape(1, 1, m.intermediate_size)
                y = m.norm(y, gate) if m.mamba_rms_norm else y * torch.nn.functional.silu(gate)
                recurrent = m.out_proj(y) * layer.ssm_out_multiplier
                attention = layer.self_attn
                ax = x * layer.attention_in_multiplier

                def heads(t: Any, head_dim: int = attention.head_dim) -> Any:
                    return t.reshape(1, 1, -1, head_dim).transpose(1, 2)

                def rotate(t: Any) -> Any:
                    half = t.shape[-1] // 2
                    return t * cos + torch.cat((-t[..., half:], t[..., :half]), -1) * sin

                q = rotate(heads(attention.q_proj(ax)))
                k = rotate(heads(attention.k_proj(ax)) * attention.key_multiplier)
                v = heads(attention.v_proj(ax))
                keys = self.keys[i].index_copy(2, position, k)
                values = self.values[i].index_copy(2, position, v)
                kk = keys.repeat_interleave(attention.num_key_value_groups, 1)
                vv = values.repeat_interleave(attention.num_key_value_groups, 1)
                scores = (q @ kk.transpose(-1, -2)) * attention.scaling
                weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
                attended = (weights @ vv).transpose(1, 2).reshape(1, 1, -1)
                hidden = hidden + recurrent + attention.o_proj(attended) * layer.attn_out_multiplier
                hidden = hidden + layer.feed_forward(layer.pre_ff_layernorm(hidden))
                next_keys.append(keys)
                next_values.append(values)
                next_conv.append(conv)
                next_ssm.append(state)
            self.keys.copy_(torch.stack(next_keys))
            self.values.copy_(torch.stack(next_values))
            self.conv.copy_(torch.stack(next_conv))
            self.ssm.copy_(torch.stack(next_ssm))
            normalized = self.norm(hidden)
            return self.output(normalized)[:, -1], self.project(normalized)

    return SlowStep().eval().requires_grad_(False)
