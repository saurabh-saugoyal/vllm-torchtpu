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
"""Emitter: the parallel-config gauge and log line."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import prometheus_client
import pytest
from prometheus_client import CollectorRegistry
from vllm.config import ParallelConfig

from vllm_torchtpu.observability import placement_emitter as emitter
from vllm_torchtpu.observability.placement_types import (
    PARALLEL_CONFIG_LABELS,
    label_names,
)


def parallel_config(**kw):
    base = dict(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        data_parallel_rank=0,
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
    summary = emitter.parallel_summary(parallel_config(nnodes=2))
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
    }
    assert list(labels) == label_names(PARALLEL_CONFIG_LABELS)


def test_managed_engines_from_core_client_or_config():
    core = SimpleNamespace(engine_ranks_managed=[3, 2])
    client = SimpleNamespace(engine_core=core)
    assert emitter.managed_engines(client, parallel_config()) == [2, 3]
    in_process = SimpleNamespace()
    assert emitter.managed_engines(
        in_process, parallel_config(data_parallel_rank=1)
    ) == [1]


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


def test_emit_prometheus_one_series_per_engine(registry):
    summary = emitter.parallel_summary(parallel_config(data_parallel_size=2))
    emitter.emit_prometheus(summary, [0, 1])
    parallel = series(registry, emitter.PARALLEL_CONFIG_METRIC)
    assert {dict(k)["engine"] for k in parallel} == {"0", "1"}
    assert set(parallel.values()) == {1.0}


def test_emit_prometheus_idempotent(registry):
    summary = emitter.parallel_summary(parallel_config())
    emitter.emit_prometheus(summary, [0])
    emitter.emit_prometheus(summary, [0])
    assert len(series(registry, emitter.PARALLEL_CONFIG_METRIC)) == 1


def test_parallel_config_metric_skipped_when_vllm_has_it(registry, monkeypatch):
    monkeypatch.setattr(emitter, "vllm_exports_parallel_config_info", lambda: True)
    emitter.emit_prometheus(emitter.parallel_summary(parallel_config()), [0])
    assert series(registry, emitter.PARALLEL_CONFIG_METRIC) == {}


def test_parallel_summary_reads_vllm_parallel_config_at_the_pin():
    """Every field is read from the real ParallelConfig, so a rename at a pin
    bump fails here rather than at server startup."""
    config = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
    summary = emitter.parallel_summary(config).to_dict()
    assert summary == {name: getattr(config, name) for name in summary}
    assert summary["tensor_parallel_size"] == 2
    assert summary["world_size"] == config.world_size


def test_emit_log_is_one_json_line(caplog):
    with caplog.at_level(logging.INFO, logger=emitter.logger.name):
        emitter.emit_log(emitter.parallel_summary(parallel_config()))
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("rank_placement ")
    ]
    payload = json.loads(lines[0].split(" ", 1)[1])
    assert payload["parallel"]["tensor_parallel_size"] == 4


# ------------------------------------------------------------------ publish


def test_publish_emits_for_every_managed_engine(registry):
    core = SimpleNamespace(engine_ranks_managed=[0, 1])
    client = SimpleNamespace(engine_core=core)
    config = SimpleNamespace(parallel_config=parallel_config(data_parallel_size=2))
    emitter.publish_placement(client, config)
    parallel = series(registry, emitter.PARALLEL_CONFIG_METRIC)
    assert {dict(k)["engine"] for k in parallel} == {"0", "1"}
