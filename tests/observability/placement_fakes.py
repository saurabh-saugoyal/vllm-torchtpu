# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared fakes for the placement tests.

Layer stand-ins subclass the real vLLM classes so ``isinstance`` dispatch in
the provider is exercised against the true types; ``__init__`` is bypassed
and every attribute the provider reads is set explicitly.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase

from vllm_torchtpu.observability.placement_types import DeviceIdentity


def _bare(cls: type, **attrs) -> torch.nn.Module:
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    for name, value in attrs.items():
        object.__setattr__(obj, name, value)
    return obj


def attention(num_heads: int, num_kv_heads: int) -> Attention:
    """Per-rank head counts, as vLLM passes them to ``Attention``."""
    return _bare(Attention, num_heads=num_heads, num_kv_heads=num_kv_heads)


def mla_attention(num_heads: int) -> MLAAttention:
    """Per-rank Q heads; vLLM sets ``num_kv_heads = 1`` (the latent KV)."""
    return _bare(MLAAttention, num_heads=num_heads, num_kv_heads=1)


def encoder_attention(num_heads: int) -> MMEncoderAttention:
    """Vision-tower attention: per-rank heads, MHA (kv == q)."""
    return _bare(MMEncoderAttention, num_heads=num_heads, num_kv_heads=num_heads)


def qkv_proj(total_heads: int, total_kv_heads: int, tp_size: int) -> QKVParallelLinear:
    replicas = tp_size // total_kv_heads if tp_size > total_kv_heads else 1
    return _bare(
        QKVParallelLinear,
        total_num_heads=total_heads,
        total_num_kv_heads=total_kv_heads,
        num_heads=total_heads // tp_size,
        num_kv_heads=max(1, total_kv_heads // tp_size),
        num_kv_head_replicas=replicas,
    )


def down_proj(intermediate: int, tp_size: int) -> RowParallelLinear:
    return _bare(
        RowParallelLinear,
        input_size=intermediate,
        input_size_per_partition=intermediate // tp_size,
    )


class GatedDeltaNet(torch.nn.Module, MambaBase):
    """Linear-attention layer as the provider sees it: ``MambaBase`` with
    per-rank state shapes (vLLM's GDN splits value heads over TP)."""

    def __init__(self, num_v_heads, tp_size, head_dim=128, conv_dim=512):
        super().__init__()
        self._shapes = (
            (conv_dim // tp_size, 4),
            (num_v_heads // tp_size, head_dim, head_dim),
        )

    def get_state_shape(self):
        return self._shapes

    @property
    def mamba_type(self):
        return SimpleNamespace(
            name="GDN"
        )  # the backend enum member, as the provider reads it

    def get_state_dtype(self):
        return (torch.bfloat16, torch.float32)


class MoeBlock(torch.nn.Module):
    """vLLM's ``Qwen2MoeSparseMoeBlock`` shape: routed experts + fused shared expert."""

    def __init__(self, global_experts, ep_rank, ep_size, tp_size, shared_intermediate):
        super().__init__()
        self.experts = routed_experts(global_experts, ep_rank, ep_size)
        self.shared_expert = torch.nn.Module()
        self.shared_expert.down_proj = down_proj(shared_intermediate, tp_size)


def routed_experts(global_experts: int, ep_rank: int, ep_size: int) -> RoutedExperts:
    per = global_experts // ep_size
    expert_map = None
    if ep_size > 1:  # vLLM: global id -> local slot, -1 if not resident
        expert_map = torch.full((global_experts,), -1, dtype=torch.int32)
        expert_map[ep_rank * per : (ep_rank + 1) * per] = torch.arange(per)
    return _bare(
        RoutedExperts,
        global_num_experts=global_experts,
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(ep_rank=ep_rank, ep_size=ep_size)
        ),
        expert_map_manager=SimpleNamespace(expert_map=expert_map),
    )


class AttentionBlock(torch.nn.Module):
    """Owner of ``attn`` and ``qkv_proj`` (vLLM's ``LlamaAttention`` shape)."""

    def __init__(self, total_heads, total_kv_heads, tp_size, encoder=False):
        super().__init__()
        self.qkv_proj = qkv_proj(total_heads, total_kv_heads, tp_size)
        if encoder:
            self.attn = encoder_attention(self.qkv_proj.num_heads)
        else:
            self.attn = attention(self.qkv_proj.num_heads, self.qkv_proj.num_kv_heads)


class DenseLayer(torch.nn.Module):
    def __init__(
        self, tp_size, heads=32, kv_heads=8, intermediate=14336, encoder=False
    ):
        super().__init__()
        self.self_attn = AttentionBlock(heads, kv_heads, tp_size, encoder=encoder)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = down_proj(intermediate, tp_size)


class FakeModel(torch.nn.Module):
    """``model.layers.<i>`` plus optional extra roots (vision tower, draft)."""

    def __init__(self, layers: list[torch.nn.Module], start_layer=0, end_layer=None):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(layers)
        self.model.start_layer = start_layer
        self.model.end_layer = len(layers) if end_layer is None else end_layer


def dense_model(tp_size, num_layers=4, **kw) -> FakeModel:
    return FakeModel([DenseLayer(tp_size, **kw) for _ in range(num_layers)])


class FakeGroup:
    def __init__(self, rank, size):
        self.rank_in_group = rank
        self.world_size = size


def fake_parallel_state(monkeypatch, tp_rank, tp_size, pp=(0, 1), pcp=None, dcp=None):
    """Point ``placement_provider.parallel_state`` at a stand-in."""
    from vllm_torchtpu.observability import placement_provider

    state = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: tp_rank,
        get_tensor_model_parallel_world_size=lambda: tp_size,
        get_pp_group=lambda: FakeGroup(*pp),
    )
    if pcp is not None:
        state.get_pcp_group = lambda: FakeGroup(*pcp)
    if dcp is not None:
        state.get_dcp_group = lambda: FakeGroup(*dcp)
    monkeypatch.setattr(placement_provider, "parallel_state", state)


def no_device(monkeypatch) -> None:
    from vllm_torchtpu.observability import placement_provider

    monkeypatch.setattr(placement_provider, "describe_device", DeviceIdentity)
