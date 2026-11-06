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

A minimal model with ``model.layers`` and pipeline bounds, and a stand-in for
vLLM's parallel state.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


class DenseLayer(torch.nn.Module):
    """One transformer block as far as the provider reads it."""

    def __init__(self, tp_size, **_):
        super().__init__()
        self.tp_size = tp_size


class FakeModel(torch.nn.Module):
    """``model.layers.<i>`` plus the pipeline stage's ``start_layer``/``end_layer``."""

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
