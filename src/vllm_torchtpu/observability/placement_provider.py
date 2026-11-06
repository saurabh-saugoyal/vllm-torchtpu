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
"""Worker-side provider: describe the calling rank's placement.

Runs once in each worker after the model is loaded. Reads vLLM's parallel
groups and the pipeline stage's layer bounds. Nothing is computed from model
names or config strings.
"""

from __future__ import annotations

import json

import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.observability.placement_types import Range, RankPlacement

logger = init_logger(__name__)


def describe_rank_placement(
    rank: int, data_parallel_rank: int, node_rank: int, model: torch.nn.Module
) -> RankPlacement:
    """Placement of this rank. Observability must not fail model loading, so a
    failure is reported inside the record (``error``) and logged, never raised.

    The record is logged here, by the worker, before KV-cache sizing and
    compilation: a rank that fails later, or a pod that dies before its first
    metrics scrape, has still left its placement in the log.
    """
    try:
        placement = _describe(rank, data_parallel_rank, node_rank, model)
    except Exception as error:  # noqa: BLE001 - see docstring
        logger.warning("Could not describe TPU placement: %s", error)
        placement = RankPlacement(
            rank=rank,
            dp_rank=data_parallel_rank,
            tp_rank=-1,
            tp_size=0,
            pp_rank=-1,
            pp_size=0,
            pcp_rank=None,
            dcp_rank=None,
            node_rank=node_rank,
            error=str(error),
        )
    logger.info("rank_placement %s", json.dumps(placement.to_dict(), sort_keys=True))
    return placement


def _describe(
    rank: int, data_parallel_rank: int, node_rank: int, model: torch.nn.Module
) -> RankPlacement:
    pp_group = parallel_state.get_pp_group()
    return RankPlacement(
        rank=rank,
        dp_rank=data_parallel_rank,
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        tp_size=parallel_state.get_tensor_model_parallel_world_size(),
        pp_rank=pp_group.rank_in_group,
        pp_size=pp_group.world_size,
        pcp_rank=_optional_group_rank("get_pcp_group"),
        dcp_rank=_optional_group_rank("get_dcp_group"),
        node_rank=node_rank,
        layers=_pipeline_layer_range(model),
    )


def _optional_group_rank(getter_name: str) -> int | None:
    """Rank in a context-parallel group; ``None`` when the vLLM build has none."""
    getter = getattr(parallel_state, getter_name, None)
    if getter is None:
        return None
    return getter().rank_in_group


def _pipeline_layer_range(model: torch.nn.Module) -> Range | None:
    """``(start_layer, end_layer)`` of the first module that declares them."""
    for module in model.modules():
        start = getattr(module, "start_layer", None)
        end = getattr(module, "end_layer", None)
        if isinstance(start, int) and isinstance(end, int):
            return (start, end)
    return None
