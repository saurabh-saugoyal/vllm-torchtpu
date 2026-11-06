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
"""Emitter: labels, gauges, log line, cross-engine collection."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import prometheus_client
import pytest
from prometheus_client import CollectorRegistry

from vllm_torchtpu.observability import placement_emitter as emitter
from vllm_torchtpu.observability.placement_types import (
    LAYER_LABELS,
    PARALLEL_CONFIG_LABELS,
    RANK_LABELS,
    AttentionSharding,
    DeviceIdentity,
    LayerGroup,
    RankPlacement,
    label_names,
)


def placement(rank, dp_rank=0, groups=None, **overrides) -> RankPlacement:
    if groups is None:
        groups = (
            LayerGroup(
                root="model",
                layers=tuple(range(32)),
                attn=AttentionSharding(
                    "full",
                    (rank * 8, rank * 8 + 8),
                    (rank * 2, rank * 2 + 2),
                    False,
                    32,
                    8,
                ),
            ),
        )
    base = dict(
        rank=rank,
        dp_rank=dp_rank,
        tp_rank=rank,
        tp_size=4,
        pp_rank=0,
        pp_size=1,
        pcp_rank=None,
        dcp_rank=None,
        node_rank=0,
        device=DeviceIdentity(rank, "TPU v6e", (rank, 0, 0), 0, rank, 4, 1, None),
        layers=(0, 32),
        layer_groups=groups,
    )
    base.update(overrides)
    return RankPlacement(**base)


def parallel_config(**kw):
    base = dict(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        world_size=4,
        nnodes=1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------- labels


def test_parallel_config_labels_exact():
    summary = emitter.parallel_summary(parallel_config(nnodes=2), [placement(0)])
    labels = emitter.parallel_config_labels(summary, 1)
    assert labels == {
        "engine": "1",
        "tensor_parallel_size": "4",
        "pipeline_parallel_size": "1",
        "data_parallel_size": "1",
        "enable_expert_parallel": "False",
        "prefill_context_parallel_size": "1",
        "decode_context_parallel_size": "1",
        "world_size": "4",
        "nnodes": "2",
        "device_kind": "TPU v6e",
        "visible_chips": "4",
        "cores_per_chip": "1",
    }
    assert list(labels) == label_names(PARALLEL_CONFIG_LABELS)


def test_placement_labels_exact():
    group = LayerGroup(
        "model",
        tuple(range(32)),
        attn=AttentionSharding("full", (8, 16), (2, 4), False, 32, 8),
    )
    p = placement(
        1,
        groups=(group,),
        pcp_rank=0,
        dcp_rank=1,
        node_rank=1,
        device=DeviceIdentity(1, "TPU v6e", (1, 0, 0), 0, 1, 4, 1, (4, 5, 6, 7)),
    )
    rank = emitter.rank_labels(p)
    assert rank == {
        "engine": "0",
        "rank": "1",
        "tp_rank": "1",
        "pp_rank": "0",
        "pcp_rank": "0",
        "dcp_rank": "1",
        "node_rank": "1",
        "device_id": "1",
        "chip_index": "1",
        "core_on_chip": "0",
        "chip_coords": "1,0,0",
        "host_visible_chips": "4-7",
        "layers": "0-31",
    }
    assert list(rank) == label_names(RANK_LABELS)
    layer = emitter.layer_labels(p, group)
    assert layer == {
        "engine": "0",
        "rank": "1",
        "module": "model",
        "layer_subset": "0-31",
        "attention_kind": "full",
        "q_heads": "8-15",
        "kv_heads": "2-3",
        "kv_replicated": "False",
    }
    assert list(layer) == label_names(LAYER_LABELS)


def test_rank_labels_without_device():
    labels = emitter.rank_labels(placement(2, groups=(), device=DeviceIdentity()))
    assert labels["rank"] == "2" and labels["chip_coords"] == ""


def test_layer_labels_blank_for_kinds_not_present():
    labels = emitter.layer_labels(placement(1), LayerGroup("model", (1, 3)))
    assert labels["attention_kind"] == "" and labels["q_heads"] == ""


def test_one_layer_row_per_group():
    groups = (
        LayerGroup(
            "model",
            (0, 2),
            attn=AttentionSharding("full", (8, 16), (0, 1), True, 16, 2),
        ),
        LayerGroup("model", (1, 3)),
        LayerGroup(
            "visual",
            (0,),
            attn=AttentionSharding("full", (0, 8), (0, 8), False, 16, 16),
        ),
    )
    p = placement(1, groups=groups)
    series = [emitter.layer_labels(p, g) for g in p.layer_groups]
    assert [(s["module"], s["layer_subset"]) for s in series] == [
        ("model", "0,2"),
        ("model", "1,3"),
        ("visual", "0"),
    ]
    assert series[0]["kv_heads"] == "0" and series[1]["q_heads"] == ""


@pytest.mark.parametrize("bounds,text", [(None, ""), ((3, 4), "3"), ((8, 16), "8-15")])
def test_format_range(bounds, text):
    assert emitter.format_range(bounds) == text


def test_format_ids():
    assert (
        emitter.format_ids((0, 1, 2, 5, 9, 10)) == "0-2,5,9-10"
        and emitter.format_ids(()) == ""
    )


def test_parallel_summary_takes_hardware_from_placements():
    s = emitter.parallel_summary(parallel_config(), [placement(0), placement(1)])
    assert (s.tensor_parallel_size, s.world_size, s.nnodes) == (4, 4, 1)
    assert (s.device_kind, s.visible_chips, s.cores_per_chip) == ("TPU v6e", 4, 1)


# ------------------------------------------------------------------- gauges


@pytest.fixture
def registry(monkeypatch):
    reg = CollectorRegistry()
    original = prometheus_client.Gauge.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("registry", reg)
        original(self, *args, **kwargs)

    monkeypatch.setattr(prometheus_client.Gauge, "__init__", init)
    monkeypatch.setattr(emitter, "_gauges", {})
    monkeypatch.setattr(emitter, "vllm_exports_parallel_config_info", lambda: False)
    return reg


def series(reg, name):
    return {
        tuple(sorted(s.labels.items())): s.value
        for m in reg.collect()
        for s in m.samples
        if s.name == name
    }


def test_emit_prometheus_one_rank_row_and_one_layer_row_per_group(registry):
    hybrid = (
        LayerGroup(
            "model",
            (0, 2),
            attn=AttentionSharding("full", (0, 8), (0, 2), False, 16, 4),
        ),
        LayerGroup("model", (1, 3)),
    )
    ps = [
        placement(0),
        placement(1),
        placement(0, dp_rank=1),
        placement(1, dp_rank=1, groups=hybrid),
    ]
    emitter.emit_prometheus(
        emitter.parallel_summary(parallel_config(data_parallel_size=2), ps), ps
    )
    parallel = series(registry, emitter.PARALLEL_CONFIG_METRIC)
    assert {dict(k)["engine"] for k in parallel} == {"0", "1"} and set(
        parallel.values()
    ) == {1.0}
    ranks = series(registry, emitter.RANK_METRIC)
    assert {(dict(k)["engine"], dict(k)["rank"]) for k in ranks} == {
        ("0", "0"),
        ("0", "1"),
        ("1", "0"),
        ("1", "1"),
    }
    layers = series(registry, emitter.LAYER_METRIC)
    assert len(layers) == 5  # 3 homogeneous ranks x 1 group + 1 hybrid rank x 2
    assert (
        sum(1 for k in layers if dict(k)["engine"] == "1" and dict(k)["rank"] == "1")
        == 2
    )


def test_emit_prometheus_idempotent(registry):
    ps = [placement(0)]
    summary = emitter.parallel_summary(parallel_config(), ps)
    emitter.emit_prometheus(summary, ps)
    emitter.emit_prometheus(summary, ps)
    assert len(series(registry, emitter.RANK_METRIC)) == 1
    assert len(series(registry, emitter.LAYER_METRIC)) == 1


def test_parallel_config_metric_skipped_when_vllm_has_it(registry, monkeypatch):
    monkeypatch.setattr(emitter, "vllm_exports_parallel_config_info", lambda: True)
    ps = [placement(0)]
    emitter.emit_prometheus(emitter.parallel_summary(parallel_config(), ps), ps)
    assert series(registry, emitter.PARALLEL_CONFIG_METRIC) == {}
    assert len(series(registry, emitter.RANK_METRIC)) == 1


def test_emit_log_is_one_json_line(caplog):
    ps = [placement(0)]
    with caplog.at_level(logging.INFO, logger=emitter.logger.name):
        emitter.emit_log(emitter.parallel_summary(parallel_config(), ps), ps)
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("rank_placement ")
    ]
    payload = json.loads(lines[0].split(" ", 1)[1])
    assert payload["parallel"]["tensor_parallel_size"] == 4
    assert payload["ranks"][0]["layer_groups"][0]["attn"]["q_heads"] == [0, 8]


# --------------------------------------------------------------- collection


class _Core:
    """Multiprocess engine-core client: one utility call fanned out to every engine."""

    def __init__(self, rows_by_engine):
        self.rows = rows_by_engine
        self.fan_out_calls = []

    async def call_utility_all_async(self, *args):
        self.fan_out_calls.append(args)
        return list(self.rows.values())


def test_collect_uses_all_engines():
    rows = {
        b"e0": [placement(0).to_dict(), placement(1).to_dict()],
        b"e1": [placement(0, 1).to_dict()],
    }
    core = _Core(rows)
    got = asyncio.run(
        emitter.collect_rank_placements(SimpleNamespace(engine_core=core))
    )
    assert [(p.dp_rank, p.rank) for p in got] == [(0, 0), (0, 1), (1, 0)]
    assert core.fan_out_calls == [("collective_rpc", "get_placement", None, (), {})]
    assert isinstance(got[0].layer_groups[0].attn, AttentionSharding)


def test_collect_falls_back_to_collective_rpc():
    async def collective_rpc(method):
        assert method == "get_placement"
        return [placement(0).to_dict(), None]

    got = asyncio.run(
        emitter.collect_rank_placements(SimpleNamespace(collective_rpc=collective_rpc))
    )
    assert [p.rank for p in got] == [0]


def test_publish_swallows_failures(caplog):
    async def collective_rpc(method):
        raise RuntimeError("engine down")

    client = SimpleNamespace(collective_rpc=collective_rpc)
    with caplog.at_level(logging.WARNING, logger=emitter.logger.name):
        asyncio.run(
            emitter.publish_placement(
                client, SimpleNamespace(parallel_config=parallel_config())
            )
        )
    assert any(
        "Could not publish TPU placement" in r.getMessage() for r in caplog.records
    )
