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
"""API-server-side emitter: publish the parallel config and rank placements.

``vllm:parallel_config_info`` is a gauge of value ``1`` with the facts in the
labels (the ``vllm:cache_config_info`` idiom), once per engine. The same
payload, plus every rank's placement, goes to the log once as a
``rank_placement`` JSON line.
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
    RankPlacement,
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


def emit_prometheus(
    summary: ParallelSummary, placements: Sequence[RankPlacement]
) -> None:
    """One ``vllm:parallel_config_info`` series per engine that reported ranks."""
    if vllm_exports_parallel_config_info():
        return
    gauge = _gauge(
        PARALLEL_CONFIG_METRIC,
        "Information of the LLMEngine ParallelConfig",
        label_names(PARALLEL_CONFIG_LABELS),
    )
    for engine in sorted({p.dp_rank for p in placements}) or [0]:
        gauge.labels(**parallel_config_labels(summary, engine)).set(1)


def emit_log(summary: ParallelSummary, placements: Sequence[RankPlacement]) -> None:
    logger.info(
        "rank_placement %s",
        json.dumps(
            {"parallel": summary.to_dict(), "ranks": [p.to_dict() for p in placements]},
            sort_keys=True,
        ),
    )


async def collect_rank_placements(engine_client: EngineClient) -> list[RankPlacement]:
    """Every rank's placement across every engine this API server manages.

    ``EngineClient.collective_rpc`` returns only engine 0's result under data
    parallelism, so the core client's all-engines utility call is used when
    the client has one (``AsyncMPClient.call_utility_all_async``); arguments
    mirror ``AsyncMPClient.collective_rpc_async``.
    """
    rpc = ("collective_rpc", "get_placement", None, (), {})
    core = getattr(engine_client, "engine_core", None)
    if hasattr(core, "call_utility_all_async"):
        per_engine = await core.call_utility_all_async(*rpc)
    else:
        per_engine = [await engine_client.collective_rpc("get_placement")]
    return [_as_placement(row) for rows in per_engine for row in rows or () if row]


def _as_placement(row: RankPlacement | dict) -> RankPlacement:
    """collective_rpc may hand back the dataclass or its serialised dict."""
    if isinstance(row, RankPlacement):
        return row
    return RankPlacement(**row)


async def publish_placement(
    engine_client: EngineClient, vllm_config: VllmConfig
) -> None:
    """Collect and publish once. Failures are logged, never raised: this runs
    in server startup and must not block serving."""
    try:
        placements = sorted(
            await collect_rank_placements(engine_client),
            key=lambda p: (p.dp_rank, p.rank),
        )
        summary = parallel_summary(vllm_config.parallel_config)
        emit_prometheus(summary, placements)
        emit_log(summary, placements)
        logger.info(
            "Published %s and the placement of %d rank(s).",
            PARALLEL_CONFIG_METRIC,
            len(placements),
        )
    except Exception as error:  # noqa: BLE001 - see docstring
        logger.warning("Could not publish TPU placement: %s", error)
