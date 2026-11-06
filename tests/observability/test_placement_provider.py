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
"""Provider: rank coordinates, device identity and per-layer sharding."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from tests.observability import placement_fakes as fakes
from vllm_torchtpu.observability import placement_provider
from vllm_torchtpu.observability.placement_provider import describe_rank_placement
from vllm_torchtpu.observability.placement_types import DeviceIdentity


@pytest.fixture
def quiet_device(monkeypatch):
    fakes.no_device(monkeypatch)


def describe(model, rank=0, dp_rank=0, node_rank=0):
    return describe_rank_placement(rank, dp_rank, node_rank, model)


def only_group(placement):
    assert len(placement.layer_groups) == 1, placement.layer_groups
    return placement.layer_groups[0]


# ---------------------------------------------------------------- coordinates


def test_rank_coordinates(monkeypatch, quiet_device):
    fakes.fake_parallel_state(
        monkeypatch, tp_rank=1, tp_size=2, pp=(1, 2), pcp=(0, 2), dcp=(1, 2)
    )
    p = describe(fakes.dense_model(tp_size=2), rank=3, dp_rank=1, node_rank=1)
    assert (p.rank, p.dp_rank, p.node_rank) == (3, 1, 1)
    assert (p.tp_rank, p.tp_size, p.pp_rank, p.pp_size) == (1, 2, 1, 2)
    assert (p.pcp_rank, p.dcp_rank) == (0, 1)
    assert p.layers == (0, 4)


def test_context_ranks_none_when_groups_missing(monkeypatch, quiet_device):
    fakes.fake_parallel_state(monkeypatch, tp_rank=0, tp_size=1)
    p = describe(fakes.dense_model(tp_size=1))
    assert p.pcp_rank is None and p.dcp_rank is None


def test_pipeline_stage_layers_from_model(monkeypatch, quiet_device):
    fakes.fake_parallel_state(monkeypatch, tp_rank=0, tp_size=1, pp=(1, 2))
    model = fakes.FakeModel([fakes.DenseLayer(1) for _ in range(16)], 16, 32)
    assert describe(model).layers == (16, 32)


def test_describe_never_raises(monkeypatch, quiet_device):
    monkeypatch.setattr(placement_provider, "parallel_state", SimpleNamespace())
    p = describe(torch.nn.Module(), rank=7)
    assert p.rank == 7 and p.error and p.layer_groups == ()


def test_device_identity(monkeypatch):
    def dev(id_, coords, core):
        return SimpleNamespace(
            id=id_, coords=coords, core_on_chip=core, device_kind="TPU v7x"
        )

    devices = [dev(0, (0, 0, 0), 0), dev(1, (0, 0, 0), 1), dev(2, (1, 0, 0), 0)]
    monkeypatch.setitem(
        sys.modules, "jax", types.SimpleNamespace(devices=lambda: devices)
    )
    from vllm_torchtpu.distributed import chip_topology, mesh_utils

    monkeypatch.setattr(mesh_utils, "_get_tpu_global_device_id", lambda: 1)
    monkeypatch.setattr(
        chip_topology,
        "get_chip_topology",
        lambda: chip_topology.ChipTopology(
            chips=[[0, 1], [2, 3]], device_ids=[0, 1, 2, 3]
        ),
    )
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "4,5,6,7")

    assert placement_provider.describe_device() == DeviceIdentity(
        device_id=1,
        device_kind="TPU v7x",
        chip_coords=(0, 0, 0),
        core_on_chip=1,
        chip_index=0,
        visible_chips=2,
        cores_per_chip=2,
        host_visible_chips=(4, 5, 6, 7),
    )


# ------------------------------------------------------------------- sharding


@pytest.mark.parametrize(
    "tp_rank,q_heads,kv_heads,intermediate",
    [
        (0, (0, 8), (0, 2), (0, 3584)),
        (1, (8, 16), (2, 4), (3584, 7168)),
        (3, (24, 32), (6, 8), (10752, 14336)),
    ],
)
def test_dense_tp4(monkeypatch, quiet_device, tp_rank, q_heads, kv_heads, intermediate):
    fakes.fake_parallel_state(monkeypatch, tp_rank=tp_rank, tp_size=4)
    group = only_group(describe(fakes.dense_model(tp_size=4), rank=tp_rank))
    assert group.root == "model" and group.layers == (0, 1, 2, 3)
    assert (group.attn.q_heads, group.attn.kv_heads) == (q_heads, kv_heads)
    assert not group.attn.kv_replicated and not group.attn.gqa_padded
    assert (group.mlp.intermediate, group.mlp.total_intermediate) == (
        intermediate,
        14336,
    )


@pytest.mark.parametrize(
    "rank,kv", [(0, (0, 1)), (3, (0, 1)), (4, (1, 2)), (7, (1, 2))]
)
def test_kv_heads_replicated_when_fewer_than_tp(monkeypatch, quiet_device, rank, kv):
    fakes.fake_parallel_state(monkeypatch, tp_rank=rank, tp_size=8)
    attn = only_group(
        describe(fakes.dense_model(8, heads=32, kv_heads=2), rank=rank)
    ).attn
    assert attn.kv_heads == kv and attn.kv_replicated
    assert attn.q_heads == (rank * 4, (rank + 1) * 4)


def test_gqa_padding_reported(monkeypatch, quiet_device):
    fakes.fake_parallel_state(monkeypatch, tp_rank=5, tp_size=8)
    layer = torch.nn.Module()
    layer.self_attn = fakes.AttentionBlock(32, 4, tp_size=8, orig_total_heads=28)
    attn = only_group(describe(fakes.FakeModel([layer]), rank=5)).attn
    assert attn.gqa_padded and attn.total_q_heads == 32 and attn.q_heads == (20, 24)


def test_mla_attention_without_qkv_proj(monkeypatch, quiet_device):
    """MLA has no QKVParallelLinear: totals derive from per-rank counts."""
    fakes.fake_parallel_state(monkeypatch, tp_rank=2, tp_size=4)
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.mla_attn = fakes.attention(num_heads=4, num_kv_heads=1)
    attn = only_group(describe(fakes.FakeModel([layer]), rank=2)).attn
    assert attn.q_heads == (8, 12) and attn.kv_heads == (2, 3)
    assert attn.total_q_heads == 16 and not attn.kv_replicated


def test_hybrid_layers_group_by_sharding(monkeypatch, quiet_device):
    """Attention on even layers, dense MLP on 0-1 only -> three groups."""
    fakes.fake_parallel_state(monkeypatch, tp_rank=1, tp_size=2)
    layers = []
    for i in range(4):
        layer = torch.nn.Module()
        if i % 2 == 0:
            layer.self_attn = fakes.AttentionBlock(16, 4, tp_size=2)
        layer.mlp = torch.nn.Module()
        if i < 2:
            layer.mlp.down_proj = fakes.down_proj(8192, 2)
        layers.append(layer)
    groups = describe(fakes.FakeModel(layers), rank=1).layer_groups
    assert [g.layers for g in groups] == [(0,), (1,), (2,)]
    assert groups[0].attn.q_heads == (8, 16) and groups[0].mlp
    assert groups[1].attn is None and groups[1].mlp.intermediate == (4096, 8192)
    assert groups[2].attn and groups[2].mlp is None


def test_linear_attention_layers_interleaved_with_full_attention(
    monkeypatch, quiet_device
):
    """Qwen3.5 / Qwen3-Next: three GDN layers then one full-attention layer,
    all with the same dense MLP -> two groups, by attention kind."""
    fakes.fake_parallel_state(monkeypatch, tp_rank=1, tp_size=4)
    layers = []
    for i in range(8):
        layer = torch.nn.Module()
        if i % 4 == 3:
            layer.self_attn = fakes.AttentionBlock(16, 4, tp_size=4)
        else:
            layer.linear_attn = fakes.GatedDeltaNet(num_v_heads=32, tp_size=4)
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = fakes.down_proj(8192, 4)
        layers.append(layer)
    groups = describe(fakes.FakeModel(layers), rank=1).layer_groups
    assert [g.layers for g in groups] == [(0, 1, 2, 4, 5, 6), (3, 7)]
    linear, full = groups
    assert linear.attn is None and full.linear_attn is None
    assert linear.linear_attn.kind == "GDN" and not linear.linear_attn.tp_replicated
    assert linear.linear_attn.state_shapes == ((128, 4), (8, 128, 128))
    assert full.attn.q_heads == (4, 8)
    assert linear.mlp == full.mlp


def test_vision_tower_and_draft_do_not_merge_with_language_model(
    monkeypatch, quiet_device
):
    fakes.fake_parallel_state(monkeypatch, tp_rank=0, tp_size=2)
    model = fakes.dense_model(tp_size=2, num_layers=2)
    model.visual = torch.nn.Module()
    model.visual.blocks = torch.nn.ModuleList(
        [fakes.DenseLayer(2, heads=16, kv_heads=16, intermediate=4096)]
    )
    model.draft = torch.nn.Module()
    model.draft.layers = torch.nn.ModuleList(
        [fakes.DenseLayer(2)]
    )  # same sharding as model
    groups = describe(model).layer_groups
    assert sorted((g.root, g.layers) for g in groups) == [
        ("draft", (0,)),
        ("model", (0, 1)),
        ("visual", (0,)),
    ]
    visual = next(g for g in groups if g.root == "visual")
    assert visual.attn.q_heads == (0, 8) and visual.attn.total_kv_heads == 16
