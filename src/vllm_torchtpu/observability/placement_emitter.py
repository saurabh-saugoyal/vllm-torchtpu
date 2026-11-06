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
"""API-server-side emitter: publish rank placements as Prometheus info gauges.

Three gauges, value ``1`` with the facts in the labels (the
``vllm:cache_config_info`` idiom), one unit of cardinality each:
``vllm:parallel_config_info`` per engine, ``vllm:rank_placement_info`` per
rank and ``vllm:layer_placement_info`` per rank and group of identically
sharded layers. The same payload goes to the log once as a
``rank_placement`` JSON line.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING

from prometheus_client import Gauge
from vllm.config import ParallelConfig

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.observability.placement_types import (
    LAYER_LABELS,
    PARALLEL_CONFIG_LABELS,
    RANK_LABELS,
    LayerGroup,
    ParallelSummary,
    Range,
    RankPlacement,
    label_names,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)

PARALLEL_CONFIG_METRIC = "vllm:parallel_config_info"
RANK_METRIC = "vllm:rank_placement_info"
LAYER_METRIC = "vllm:layer_placement_info"

_gauges: dict[str, Gauge] = {}


def parallel_summary(
    parallel_config: ParallelConfig, placements: Sequence[RankPlacement]
) -> ParallelSummary:
    device = next((p.device for p in placements if p.device.device_kind), None)
    return ParallelSummary(
        tensor_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        data_parallel_size=parallel_config.data_parallel_size,
        enable_expert_parallel=parallel_config.enable_expert_parallel,
        prefill_context_parallel_size=parallel_config.prefill_context_parallel_size,
        decode_context_parallel_size=parallel_config.decode_context_parallel_size,
        world_size=parallel_config.world_size,
        nnodes=parallel_config.nnodes,
        device_kind=device.device_kind if device else None,
        visible_chips=device.visible_chips if device else None,
        cores_per_chip=device.cores_per_chip if device else None,
    )


def format_range(bounds: Range | None) -> str:
    """Half-open ``(8, 16)`` -> inclusive ``"8-15"``; ``"8"`` for one element."""
    if bounds is None:
        return ""
    start, stop = bounds
    return str(start) if stop - start == 1 else f"{start}-{stop - 1}"


def format_ids(ids: Iterable[int] | None) -> str:
    """``(0, 1, 2, 5, 9, 10)`` -> ``"0-2,5,9-10"``."""
    if not ids:
        return ""
    ordered = sorted(ids)
    runs: list[list[int]] = [[ordered[0], ordered[0]]]
    for i in ordered[1:]:
        if i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _label(value: object) -> str:
    return "" if value is None else str(value)


def rank_labels(placement: RankPlacement) -> dict[str, str]:
    """Per-process facts: coordinates, device, pipeline range. One row per rank."""
    device = placement.device
    return {
        "engine": str(placement.dp_rank),
        "rank": str(placement.rank),
        "tp_rank": _label(placement.tp_rank),
        "pp_rank": _label(placement.pp_rank),
        "pcp_rank": _label(placement.pcp_rank),
        "dcp_rank": _label(placement.dcp_rank),
        "node_rank": str(placement.node_rank),
        "device_id": _label(device.device_id),
        "chip_index": _label(device.chip_index),
        "core_on_chip": _label(device.core_on_chip),
        "chip_coords": ",".join(map(str, device.chip_coords))
        if device.chip_coords
        else "",
        "host_visible_chips": format_ids(device.host_visible_chips),
        "layers": format_range(placement.layers),
    }


def layer_labels(placement: RankPlacement, group: LayerGroup) -> dict[str, str]:
    """Per-layer-group facts: which slice of these layers the rank holds."""
    attn = group.attn
    return {
        "engine": str(placement.dp_rank),
        "rank": str(placement.rank),
        "module": group.root,
        "layer_subset": format_ids(group.layers),
        "attention_kind": attn.kind if attn else "",
        "q_heads": format_range(attn.q_heads) if attn else "",
        "kv_heads": format_range(attn.kv_heads) if attn else "",
        "kv_replicated": _label(attn.kv_replicated) if attn else "",
    }


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
    engines = sorted({p.dp_rank for p in placements}) or [0]
    if not vllm_exports_parallel_config_info():
        gauge = _gauge(
            PARALLEL_CONFIG_METRIC,
            "Information of the LLMEngine ParallelConfig",
            label_names(PARALLEL_CONFIG_LABELS),
        )
        for engine in engines:
            gauge.labels(**parallel_config_labels(summary, engine)).set(1)
    ranks = _gauge(
        RANK_METRIC,
        "Parallel-group position and TPU device of each rank; one series per rank",
        label_names(RANK_LABELS),
    )
    layers = _gauge(
        LAYER_METRIC,
        "Slice of each group of identically sharded layers a rank holds; "
        "one series per rank and layer group",
        label_names(LAYER_LABELS),
    )
    for placement in placements:
        ranks.labels(**rank_labels(placement)).set(1)
        for group in placement.layer_groups:
            layers.labels(**layer_labels(placement, group)).set(1)


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
    from vllm_torchtpu.observability.placement_types import (
        AttentionSharding,
        DeviceIdentity,
    )

    def group(g: dict) -> LayerGroup:
        return LayerGroup(
            root=g["root"],
            layers=tuple(g["layers"]),
            attn=AttentionSharding(**g["attn"]) if g["attn"] else None,
        )

    return RankPlacement(
        **{k: v for k, v in row.items() if k not in ("device", "layer_groups")},
        device=DeviceIdentity(**row["device"]),
        layer_groups=tuple(group(g) for g in row["layer_groups"]),
    )


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
        summary = parallel_summary(vllm_config.parallel_config, placements)
        emit_prometheus(summary, placements)
        emit_log(summary, placements)
        logger.info(
            "Published TPU placement for %d rank(s) as %s, %s and %s.",
            len(placements),
            PARALLEL_CONFIG_METRIC,
            RANK_METRIC,
            LAYER_METRIC,
        )
    except Exception as error:  # noqa: BLE001 - see docstring
        logger.warning("Could not publish TPU placement: %s", error)
