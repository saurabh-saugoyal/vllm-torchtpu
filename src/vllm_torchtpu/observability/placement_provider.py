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
groups, the pipeline stage's layer bounds and the device this process is
pinned to. Nothing is computed from model names or config strings.
"""

from __future__ import annotations

import os

import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.observability.placement_types import (
    DeviceIdentity,
    Range,
    RankPlacement,
)

logger = init_logger(__name__)


def describe_rank_placement(
    rank: int, data_parallel_rank: int, node_rank: int, model: torch.nn.Module
) -> RankPlacement:
    """Placement of this rank. Observability must not fail model loading, so a
    failure is reported inside the record (``error``) and logged, never raised.
    """
    try:
        return _describe(rank, data_parallel_rank, node_rank, model)
    except Exception as error:  # noqa: BLE001 - see docstring
        logger.warning("Could not describe TPU placement: %s", error)
        return RankPlacement(
            rank=rank,
            dp_rank=data_parallel_rank,
            tp_rank=-1,
            tp_size=0,
            pp_rank=-1,
            pp_size=0,
            pcp_rank=None,
            dcp_rank=None,
            node_rank=node_rank,
            device=DeviceIdentity(),
            error=str(error),
        )


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
        device=describe_device(),
        layers=_pipeline_layer_range(model),
    )


def _optional_group_rank(getter_name: str) -> int | None:
    """Rank in a context-parallel group; ``None`` when the vLLM build has none."""
    getter = getattr(parallel_state, getter_name, None)
    if getter is None:
        return None
    return getter().rank_in_group


def describe_device() -> DeviceIdentity:
    """The TPU core this process is pinned to, from JAX and ChipTopology."""
    import jax

    from vllm_torchtpu.distributed.chip_topology import get_chip_topology
    from vllm_torchtpu.distributed.mesh_utils import _get_tpu_global_device_id

    pinned = os.environ.get("TPU_VISIBLE_CHIPS", "")
    host_visible_chips = (
        tuple(int(c) for c in pinned.split(",") if c.strip()) if pinned else None
    )
    device_id = int(_get_tpu_global_device_id())
    device = next((d for d in jax.devices() if int(d.id) == device_id), None)
    topology = get_chip_topology()
    chip_index = None
    if topology is not None:
        chip_index = next(
            (i for i, ids in enumerate(topology.chips) if device_id in ids), None
        )
    return DeviceIdentity(
        device_id=device_id,
        device_kind=device.device_kind if device is not None else None,
        chip_coords=tuple(int(c) for c in device.coords)
        if device is not None
        else None,
        core_on_chip=int(device.core_on_chip) if device is not None else None,
        chip_index=chip_index,
        visible_chips=topology.num_chips if topology is not None else None,
        cores_per_chip=topology.cores_per_chip if topology is not None else None,
        host_visible_chips=host_visible_chips,
    )


def _pipeline_layer_range(model: torch.nn.Module) -> Range | None:
    """``(start_layer, end_layer)`` of the first module that declares them."""
    for module in model.modules():
        start = getattr(module, "start_layer", None)
        end = getattr(module, "end_layer", None)
        if isinstance(start, int) and isinstance(end, int):
            return (start, end)
    return None
