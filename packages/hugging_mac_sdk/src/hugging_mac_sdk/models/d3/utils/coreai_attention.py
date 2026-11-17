"""Exportable dynamic Gated DeltaNet recurrence, retaining upstream weights/math.

The recurrence is adapted from Transformers' Qwen3.5 implementation (Apache-2.0).
A Core AI while loop replaces Python sequence-length loops. It does not change
causal linear attention or the separate noncausal full-attention layers.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F


def recurrent_delta(query: Any, key: Any, value: Any, g: Any, beta: Any) -> Any:
    dtype = query.dtype
    q, k, v, decay, rate = (x.float() for x in (query, key, value, g, beta))
    # sum(x²) = D * mean(x²): retain L2 normalization while avoiding the
    # observed Core AI CPU fusion of a sum-based pattern into RMS normalization.
    q = q * torch.rsqrt((q * q).mean(-1, keepdim=True) + 1e-6 / q.shape[-1])
    q = q / (q.shape[-1] ** 0.5)
    k = k * torch.rsqrt((k * k).mean(-1, keepdim=True) + 1e-6 / k.shape[-1])
    k = k / (k.shape[-1] ** 0.5)
    q = q / (q.shape[-1] ** 0.5)
    batch, length, heads, width = k.shape
    state = torch.zeros((batch, heads, width, v.shape[-1]), device=q.device)
    output = torch.zeros((length, batch, heads, v.shape[-1]), device=q.device)
    index = torch.zeros((), dtype=torch.int64, device=q.device)

    def condition(i: Any, state: Any, output: Any) -> Any:
        return i < length

    def body(i: Any, state: Any, output: Any) -> tuple[Any, Any, Any]:
        ix = i.reshape(1)
        qi, ki, vi = (x.index_select(1, ix).squeeze(1) for x in (q, k, v))
        gi = decay.index_select(1, ix).squeeze(1).exp()[..., None, None]
        bi = rate.index_select(1, ix).squeeze(1).unsqueeze(-1)
        state = state * gi
        memory = (state * ki.unsqueeze(-1)).sum(-2)
        delta = (vi - memory) * bi
        state = state + ki.unsqueeze(-1) * delta.unsqueeze(-2)
        current = (state * qi.unsqueeze(-1)).sum(-2)
        output = output.index_copy(0, ix, current.unsqueeze(0))
        return i + 1, state, output

    if torch.compiler.is_compiling():
        output = torch.while_loop(  # type: ignore[no-untyped-call]
            condition, body, (index, state, output)
        )[2]
    else:
        # Avoid compiling the host oracle; exported inference uses the same body.
        for _ in range(length):
            index, state, output = body(index, state, output)
    return output.permute(1, 0, 2, 3).to(dtype)


class DynamicDeltaNet(torch.nn.Module):
    def __init__(self, original: Any) -> None:
        super().__init__()
        self.original = original

    def forward(self, hidden_states: Any, attention_mask: Any = None, **kwargs: Any) -> Any:
        from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn

        m = self.original
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
        batch, length, _ = hidden_states.shape
        projected = m.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed = causal_conv1d_fn(
            projected, m.conv1d.weight.squeeze(1), m.conv1d.bias, activation=m.activation
        ).transpose(1, 2)
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(batch, length, -1, m.head_k_dim)
        k = k.reshape(batch, length, -1, m.head_k_dim)
        v = v.reshape(batch, length, -1, m.head_v_dim)
        if m.num_v_heads // m.num_k_heads > 1:
            q = q.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
            k = k.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        beta = m.in_proj_b(hidden_states).sigmoid()
        g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(hidden_states).float() + m.dt_bias)
        core = recurrent_delta(q, k, v, g, beta)
        z = m.in_proj_z(hidden_states).reshape(-1, m.head_v_dim)
        normalized = m.norm(core.reshape(-1, m.head_v_dim), z).reshape(batch, length, -1)
        return m.out_proj(normalized)


def dynamic_language_model(original: Any, *, compute_dtype: Any = None) -> Any:
    """Copy module structure, sharing immutable weights; keep the oracle unmodified."""
    if compute_dtype is not None:
        original = copy.deepcopy(original).to(compute_dtype)
    model = copy.copy(original)
    model._modules = original._modules.copy()
    layers = []
    for source in original.layers:
        layer = copy.copy(source)
        layer._modules = source._modules.copy()
        if hasattr(source, "linear_attn"):
            layer.linear_attn = DynamicDeltaNet(source.linear_attn)
        layers.append(layer)
    model.layers = torch.nn.ModuleList(layers)
    return model.eval()
