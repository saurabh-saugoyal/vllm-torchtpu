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
"""Typed records describing where a rank sits in the engine's parallel groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

Range = tuple[int, int]
"""Half-open ``[start, stop)`` interval."""


@dataclass(frozen=True)
class RankPlacement:
    """Everything one worker rank reports; serialisable with :func:`asdict`."""

    rank: int
    dp_rank: int
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    pcp_rank: int | None
    dcp_rank: int | None
    node_rank: int
    layers: Range | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParallelSummary:
    """Engine-wide parallel configuration."""

    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    enable_expert_parallel: bool
    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    world_size: int
    nnodes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelSpec:
    name: str
    doc: str


# Label value formats: [index] 0-based position; [count] size; [bool]
# "True"/"False"; [range] inclusive "a-b" ("a" if one element). "" means not
# applicable. Gauge value is always 1.

PARALLEL_CONFIG_LABELS: tuple[LabelSpec, ...] = (
    LabelSpec(
        "engine", "[index] vLLM engine == data-parallel rank; joins vLLM's metrics"
    ),
    LabelSpec("tensor_parallel_size", "[count] ranks splitting each layer's heads"),
    LabelSpec("pipeline_parallel_size", "[count] stages splitting the layer sequence"),
    LabelSpec("data_parallel_size", "[count] engines (model replicas)"),
    LabelSpec("enable_expert_parallel", "[bool] experts spread across TP x DP ranks"),
    LabelSpec("prefill_context_parallel_size", "[count] ranks splitting a prefill"),
    LabelSpec("decode_context_parallel_size", "[count] ranks splitting the KV cache"),
    LabelSpec("world_size", "[count] worker processes per engine = TP x PP x PCP"),
    LabelSpec("nnodes", "[count] hosts one engine spans; 1 unless --nnodes"),
)

PLACEMENT_LABELS: tuple[LabelSpec, ...] = (
    LabelSpec("engine", "[index] engine this rank belongs to"),
    LabelSpec("rank", "[index] worker process within the engine, 0..world_size-1"),
    LabelSpec("tp_rank", "[index] position in the tensor-parallel group"),
    LabelSpec("pp_rank", "[index] pipeline stage"),
    LabelSpec("pcp_rank", "[index] prefill-context group: which prompt chunk"),
    LabelSpec(
        "dcp_rank", "[index] decode-context group: which KV shard (reuses TP ranks)"
    ),
    LabelSpec("node_rank", "[index] host of the engine's slice, 0..nnodes-1"),
    LabelSpec("layers", "[range] layers held by this pipeline stage"),
)


def label_names(specs: tuple[LabelSpec, ...]) -> list[str]:
    return [spec.name for spec in specs]
