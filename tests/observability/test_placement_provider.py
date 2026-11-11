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

import json
import logging
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


def test_each_worker_logs_its_own_record(monkeypatch, quiet_device, caplog):
    """Logged before KV-cache sizing and compilation, so a later failure still
    leaves the rank's placement in the pod log."""
    fakes.fake_parallel_state(monkeypatch, tp_rank=1, tp_size=2)
    with caplog.at_level(logging.INFO, logger=placement_provider.logger.name):
        describe(fakes.dense_model(tp_size=2), rank=1, dp_rank=0, node_rank=0)
    lines = [
        r.getMessage() for r in caplog.records if "rank_placement " in r.getMessage()
    ]
    assert len(lines) == 1
    record = json.loads(lines[0].split("rank_placement ", 1)[1])
    assert (record["rank"], record["tp_rank"], record["layers"]) == (1, 1, [0, 4])


# ------------------------------------------------------------------- sharding


@pytest.mark.parametrize(
    "tp_rank,q_heads,kv_heads",
    [(0, (0, 8), (0, 2)), (1, (8, 16), (2, 4)), (3, (24, 32), (6, 8))],
)
def test_dense_tp4(monkeypatch, quiet_device, tp_rank, q_heads, kv_heads):
    fakes.fake_parallel_state(monkeypatch, tp_rank=tp_rank, tp_size=4)
    group = only_group(describe(fakes.dense_model(tp_size=4), rank=tp_rank))
    assert group.root == "model" and group.layers == (0, 1, 2, 3)
    assert (group.attn.q_heads, group.attn.kv_heads) == (q_heads, kv_heads)
    assert group.attn.kind == "full" and not group.attn.kv_replicated


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


def test_mla_latent_kv_is_replicated_across_tp(monkeypatch, quiet_device):
    """DeepSeek MLA: Q heads split by TP, the single latent KV head on every rank."""
    fakes.fake_parallel_state(monkeypatch, tp_rank=2, tp_size=4)
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.mla_attn = fakes.mla_attention(num_heads=4)
    attn = only_group(describe(fakes.FakeModel([layer]), rank=2)).attn
    assert attn.q_heads == (8, 12) and attn.total_q_heads == 16
    assert attn.kv_heads == (0, 1) and attn.total_kv_heads == 1 and attn.kv_replicated


def test_sliding_window_and_cross_attention_kinds(monkeypatch, quiet_device):
    """Gemma-style local/global alternation groups by kind; cross attention is
    its own kind too."""
    fakes.fake_parallel_state(monkeypatch, tp_rank=0, tp_size=2)
    layers = []
    for i in range(4):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.attn = fakes.attention(
            8, 4, sliding_window=512 if i % 2 == 0 else None
        )
        layers.append(layer)
    cross = torch.nn.Module()
    cross.cross_attn = torch.nn.Module()
    cross.cross_attn.attn = fakes.attention(
        8, 4, attn_type=fakes.AttentionType.ENCODER_DECODER
    )
    layers.append(cross)
    groups = describe(fakes.FakeModel(layers)).layer_groups
    assert [(g.attn.kind, g.layers) for g in groups] == [
        ("sliding", (0, 2)),
        ("full", (1, 3)),
        ("cross", (4,)),
    ]


def test_dcp_token_shard_on_attention(monkeypatch, quiet_device):
    fakes.fake_parallel_state(monkeypatch, tp_rank=1, tp_size=4, dcp=(1, 2))
    attn = only_group(describe(fakes.dense_model(tp_size=4), rank=1)).attn
    assert attn.kv_token_shard == (1, 2)
    fakes.fake_parallel_state(monkeypatch, tp_rank=1, tp_size=4, dcp=(0, 1))
    attn = only_group(describe(fakes.dense_model(tp_size=4), rank=1)).attn
    assert attn.kv_token_shard is None


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
        layers.append(layer)
    groups = describe(fakes.FakeModel(layers), rank=1).layer_groups
    assert [g.layers for g in groups] == [(0, 1, 2, 4, 5, 6), (3, 7)]
    linear, full = groups
    assert linear.attn is None and full.linear_attn is None
    assert linear.linear_attn.kind == "GDN" and not linear.linear_attn.tp_replicated
    assert linear.linear_attn.state_shapes == ((128, 4), (8, 128, 128))
    assert full.attn.q_heads == (4, 8)


def test_vision_tower_and_draft_do_not_merge_with_language_model(
    monkeypatch, quiet_device
):
    fakes.fake_parallel_state(monkeypatch, tp_rank=0, tp_size=2)
    model = fakes.dense_model(tp_size=2, num_layers=2)
    model.visual = torch.nn.Module()
    model.visual.blocks = torch.nn.ModuleList(
        [fakes.DenseLayer(2, heads=16, kv_heads=16, encoder=True)]
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
    assert visual.attn.q_heads == (0, 8) and visual.attn.kv_heads == (0, 8)
    assert visual.attn.total_kv_heads == 16 and not visual.attn.kv_replicated
