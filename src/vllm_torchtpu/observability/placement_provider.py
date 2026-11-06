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
groups, the device this process is pinned to, and the sharded layers' own
attributes (``Attention.num_heads``, ``RowParallelLinear.input_size_per_
partition``). Nothing is computed from model names or config strings.
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch
from vllm.distributed import parallel_state
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.utils import extract_layer_index

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.observability.placement_types import (
    AttentionSharding,
    DeviceIdentity,
    LayerGroup,
    MlpSharding,
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
        layer_groups=describe_layer_groups(model),
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


def describe_layer_groups(model: torch.nn.Module) -> tuple[LayerGroup, ...]:
    """Layers grouped by (root module, sharding).

    A layer is identified by its root module (the first name component, so a
    vision tower, a draft model and the language model never merge) and its
    index from vLLM's ``extract_layer_index``. Layers with equal attention /
    MLP sharding collapse into one group, so a homogeneous model yields one
    group and a hybrid model one per layer type.
    """
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    modules = dict(model.named_modules())
    per_layer: dict[tuple[str, int | None], dict[str, object]] = defaultdict(dict)
    for name, module in modules.items():
        kind, sharding = _describe_module(name, module, modules, tp_rank)
        if kind is None:
            continue
        key = (name.split(".", 1)[0], _layer_index(name))
        per_layer[key].setdefault(kind, sharding)

    groups: dict[tuple, LayerGroup] = {}
    for (root, index), parts in sorted(
        per_layer.items(), key=lambda kv: (kv[0][0], kv[0][1] is None, kv[0][1] or 0)
    ):
        candidate = LayerGroup(
            root=root,
            layers=() if index is None else (index,),
            attn=parts.get("attn"),  # type: ignore[arg-type]
            mlp=parts.get("mlp"),  # type: ignore[arg-type]
        )
        key = candidate.sharding_key()
        if key in groups:
            groups[key] = LayerGroup(
                root,
                groups[key].layers + candidate.layers,
                candidate.attn,
                candidate.mlp,
            )
        else:
            groups[key] = candidate
    return tuple(groups.values())


def _layer_index(module_name: str) -> int | None:
    try:
        return extract_layer_index(module_name)
    except ValueError:
        return None


def _describe_module(
    name: str,
    module: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    tp_rank: int,
):
    if isinstance(module, MambaBase):
        # Linear attention / SSM: no heads to shard; its state is described
        # separately.
        return None, None
    if isinstance(module, AttentionLayerBase):
        return "attn", _attention_sharding(name, module, modules, tp_rank)
    if isinstance(module, RowParallelLinear) and name.rsplit(".", 1)[-1] in (
        "down_proj",
        "fc2",
        "c_proj",
        "w2",
    ):
        return "mlp", MlpSharding(
            intermediate=(
                tp_rank * module.input_size_per_partition,
                (tp_rank + 1) * module.input_size_per_partition,
            ),
            total_intermediate=module.input_size,
        )
    return None, None


def _attention_sharding(
    name: str,
    layer: AttentionLayerBase,
    modules: dict[str, torch.nn.Module],
    tp_rank: int,
) -> AttentionSharding:
    """``Attention`` and ``MLAAttention`` carry per-rank head counts, so GQA,
    MQA, MLA and vision attention are described alike. Totals and the Qwen2
    padding marker come from the sibling ``QKVParallelLinear`` / owning
    attention module when present (GQA); MLA has one latent KV head per rank.
    """
    num_heads = int(layer.num_heads)
    num_kv_heads = int(layer.num_kv_heads)
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    owner_name = name.rsplit(".", 1)[0]
    owner = modules.get(owner_name)
    qkv = next(
        (
            m
            for n, m in modules.items()
            if n.startswith(owner_name + ".") and isinstance(m, QKVParallelLinear)
        ),
        None,
    )
    if qkv is not None:
        total_q, total_kv = int(qkv.total_num_heads), int(qkv.total_num_kv_heads)
        replicas = int(qkv.num_kv_head_replicas)
    else:
        total_q, total_kv, replicas = num_heads * tp_size, num_kv_heads * tp_size, 1
    kv_heads: Range = (
        (tp_rank // replicas, tp_rank // replicas + 1)
        if replicas > 1
        else (tp_rank * num_kv_heads, (tp_rank + 1) * num_kv_heads)
    )
    orig_total_q = int(getattr(owner, "orig_total_num_heads", total_q))
    return AttentionSharding(
        q_heads=(tp_rank * num_heads, (tp_rank + 1) * num_heads),
        kv_heads=kv_heads,
        kv_replicated=replicas > 1,
        total_q_heads=total_q,
        total_kv_heads=total_kv,
        gqa_padded=orig_total_q != total_q,
    )
