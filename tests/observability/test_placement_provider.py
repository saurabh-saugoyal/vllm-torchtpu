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
"""Provider: rank coordinates and device identity."""

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
    assert p.rank == 7 and p.error and p.device == DeviceIdentity()


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
