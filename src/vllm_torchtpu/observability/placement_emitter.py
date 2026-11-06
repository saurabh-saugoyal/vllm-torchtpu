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
"""API-server-side emitter: publish the parallel config as a Prometheus info gauge.

One gauge, value ``1`` with the facts in the labels (the
``vllm:cache_config_info`` idiom): ``vllm:parallel_config_info`` once per
engine. The same payload goes to the log once as a ``rank_placement`` JSON
line.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING

from prometheus_client import Gauge
from vllm.config import ParallelConfig

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.observability.placement_types import (
    PARALLEL_CONFIG_LABELS,
    ParallelSummary,
    label_names,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)

PARALLEL_CONFIG_METRIC = "vllm:parallel_config_info"

_gauges: dict[str, Gauge] = {}


def parallel_summary(parallel_config: ParallelConfig) -> ParallelSummary:
    return ParallelSummary(
        tensor_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        data_parallel_size=parallel_config.data_parallel_size,
        enable_expert_parallel=parallel_config.enable_expert_parallel,
        prefill_context_parallel_size=parallel_config.prefill_context_parallel_size,
        decode_context_parallel_size=parallel_config.decode_context_parallel_size,
        world_size=parallel_config.world_size,
        nnodes=parallel_config.nnodes,
    )


def _label(value: object) -> str:
    return "" if value is None else str(value)


def parallel_config_labels(summary: ParallelSummary, engine: int) -> dict[str, str]:
    values = {"engine": engine, **asdict(summary)}
    return {spec.name: _label(values[spec.name]) for spec in PARALLEL_CONFIG_LABELS}


def vllm_exports_parallel_config_info() -> bool:
    """True once upstream vLLM publishes its own ParallelConfig info gauge."""
    return callable(getattr(ParallelConfig, "metrics_info", None))


def _gauge(name: str, documentation: str, labelnames: list[str]) -> Gauge:
    """Create once per process: a second registration would raise."""
    if name not in _gauges:
        _gauges[name] = Gauge(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
            multiprocess_mode="mostrecent",
        )
    return _gauges[name]


def emit_prometheus(summary: ParallelSummary, engines: Sequence[int]) -> None:
    if vllm_exports_parallel_config_info():
        return
    gauge = _gauge(
        PARALLEL_CONFIG_METRIC,
        "Information of the LLMEngine ParallelConfig",
        label_names(PARALLEL_CONFIG_LABELS),
    )
    for engine in engines:
        gauge.labels(**parallel_config_labels(summary, engine)).set(1)


def emit_log(summary: ParallelSummary) -> None:
    logger.info(
        "rank_placement %s", json.dumps({"parallel": summary.to_dict()}, sort_keys=True)
    )


def managed_engines(
    engine_client: EngineClient, parallel_config: ParallelConfig
) -> list[int]:
    """Data-parallel ranks served by this API server, one per engine core.

    The multiprocess core client lists them as ``engine_ranks_managed``; an
    in-process engine serves only this node's own data-parallel rank.
    """
    core = getattr(engine_client, "engine_core", None)
    ranks = getattr(core, "engine_ranks_managed", None)
    if ranks:
        return sorted(int(rank) for rank in ranks)
    return [parallel_config.data_parallel_rank]


async def publish_placement(
    engine_client: EngineClient, vllm_config: VllmConfig
) -> None:
    """Publish once. Failures are logged, never raised: this runs in server
    startup and must not block serving."""
    try:
        parallel_config = vllm_config.parallel_config
        summary = parallel_summary(parallel_config)
        engines = managed_engines(engine_client, parallel_config)
        emit_prometheus(summary, engines)
        emit_log(summary)
        logger.info(
            "Published %s for engine(s) %s.",
            PARALLEL_CONFIG_METRIC,
            ",".join(map(str, engines)),
        )
    except Exception as error:  # noqa: BLE001 - see docstring
        logger.warning("Could not publish TPU placement: %s", error)
