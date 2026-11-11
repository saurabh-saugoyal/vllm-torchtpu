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
partition``, the routed-experts layer's expert map). Nothing is computed from
model names or config strings.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import torch
from vllm.distributed import parallel_state
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionType

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.observability.placement_types import (
    AttentionSharding,
    DeviceIdentity,
    LayerGroup,
    LinearAttentionSharding,
    MlpSharding,
    MoeSharding,
    Range,
    RankPlacement,
    VocabSharding,
)

logger = init_logger(__name__)


def describe_rank_placement(
    rank: int,
    data_parallel_rank: int,
    node_rank: int,
    model: torch.nn.Module,
    draft_model: torch.nn.Module | None = None,
) -> RankPlacement:
    """Placement of this rank. Observability must not fail model loading, so a
    failure is reported inside the record (``error``) and logged, never raised.

    The record is logged here, by the worker, before KV-cache sizing and
    compilation: a rank that fails later, or a pod that dies before its first
    metrics scrape, has still left its placement in the log.
    """
    try:
        placement = _describe(rank, data_parallel_rank, node_rank, model, draft_model)
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
            device=DeviceIdentity(),
            error=str(error),
        )
    logger.info("rank_placement %s", json.dumps(placement.to_dict(), sort_keys=True))
    return placement


def _describe(
    rank: int,
    data_parallel_rank: int,
    node_rank: int,
    model: torch.nn.Module,
    draft_model: torch.nn.Module | None,
) -> RankPlacement:
    pp_group = parallel_state.get_pp_group()
    dcp = _optional_group("get_dcp_group")
    roots = {"": model} if draft_model is None else {"": model, "draft": draft_model}
    return RankPlacement(
        rank=rank,
        dp_rank=data_parallel_rank,
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        tp_size=parallel_state.get_tensor_model_parallel_world_size(),
        pp_rank=pp_group.rank_in_group,
        pp_size=pp_group.world_size,
        pcp_rank=_optional_group_rank("get_pcp_group"),
        dcp_rank=dcp[0] if dcp else None,
        node_rank=node_rank,
        device=describe_device(),
        layers=_pipeline_layer_range(model),
        layer_groups=describe_layer_groups(roots, dcp=dcp),
    )


def _optional_group(getter_name: str) -> tuple[int, int] | None:
    """``(rank, size)`` of a context-parallel group; ``None`` when the vLLM
    build has none."""
    getter = getattr(parallel_state, getter_name, None)
    if getter is None:
        return None
    group = getter()
    return (group.rank_in_group, group.world_size)


def _optional_group_rank(getter_name: str) -> int | None:
    group = _optional_group(getter_name)
    return group[0] if group else None


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


def describe_layer_groups(
    roots: dict[str, torch.nn.Module], dcp: tuple[int, int] | None = None
) -> tuple[LayerGroup, ...]:
    """Layers grouped by (root module, sharding).

    ``roots`` maps a name prefix to a model: ``""`` for the target model and
    ``"draft"`` for a speculative-decoding draft model. A layer is identified
    by its root module (the first name component, so a vision tower, a draft
    model, ``lm_head`` and the language model never merge) and its index from
    vLLM's ``extract_layer_index``. Layers with equal sharding collapse into
    one group, so a homogeneous model yields one group and a hybrid model one
    per layer type.
    """
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    modules = {
        f"{prefix}.{name}" if prefix else name: module
        for prefix, model in roots.items()
        for name, module in model.named_modules()
    }
    per_layer: dict[tuple[str, int | None], dict[str, object]] = defaultdict(dict)
    for name, module in modules.items():
        kind, sharding = _describe_module(name, module, modules, tp_rank, dcp)
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
            linear_attn=parts.get("linear_attn"),  # type: ignore[arg-type]
            mlp=parts.get("mlp"),  # type: ignore[arg-type]
            moe=parts.get("moe"),  # type: ignore[arg-type]
            shared_mlp=parts.get("shared_mlp"),  # type: ignore[arg-type]
            vocab=parts.get("vocab"),  # type: ignore[arg-type]
        )
        key = candidate.sharding_key()
        if key in groups:
            groups[key] = LayerGroup(
                root,
                groups[key].layers + candidate.layers,
                candidate.attn,
                candidate.linear_attn,
                candidate.mlp,
                candidate.moe,
                candidate.shared_mlp,
                candidate.vocab,
            )
        else:
            groups[key] = candidate
    return tuple(groups.values())


# vLLM's attribute names for the always-on expert(s) of a MoE block
# (Qwen2-MoE / Llama 4: ``shared_expert``; DeepSeek: ``shared_experts``).
_SHARED_EXPERT = frozenset({"shared_expert", "shared_experts"})


def _layer_index(module_name: str) -> int | None:
    """vLLM's layer index for ``model.layers.<i>.…``; ``None`` for modules
    outside a layer stack (``lm_head``, a vision merger), which
    ``extract_layer_index`` rejects."""
    if not any(part.isdigit() for part in module_name.split(".")):
        return None
    return extract_layer_index(module_name)


def _describe_module(
    name: str,
    module: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    tp_rank: int,
    dcp: tuple[int, int] | None,
):
    if isinstance(module, MambaBase):
        return "linear_attn", LinearAttentionSharding(
            kind=module.mamba_type.name,
            state_shapes=tuple(
                tuple(int(d) for d in s) for s in module.get_state_shape()
            ),
            tp_replicated=bool(module.is_kv_cache_tp_replicated),
        )
    if isinstance(module, (AttentionLayerBase, MMEncoderAttention)):
        return "attn", _attention_sharding(name, module, modules, tp_rank, dcp)
    if isinstance(module, RoutedExperts):
        return "moe", _moe_sharding(module)
    if isinstance(module, ParallelLMHead):
        shard = module.shard_indices
        return "vocab", VocabSharding(
            rows=(int(shard.org_vocab_start_index), int(shard.org_vocab_end_index)),
            total_rows=int(module.org_vocab_size),
        )
    if isinstance(module, RowParallelLinear) and name.rsplit(".", 1)[-1] in (
        "down_proj",
        "fc2",
        "c_proj",
        "w2",
    ):
        kind = "shared_mlp" if _SHARED_EXPERT.intersection(name.split(".")) else "mlp"
        return kind, MlpSharding(
            intermediate=(
                tp_rank * module.input_size_per_partition,
                (tp_rank + 1) * module.input_size_per_partition,
            ),
            total_intermediate=module.input_size,
        )
    return None, None


def _attention_sharding(
    name: str,
    layer: AttentionLayerBase | MMEncoderAttention,
    modules: dict[str, torch.nn.Module],
    tp_rank: int,
    dcp: tuple[int, int] | None,
) -> AttentionSharding:
    """``Attention``, ``MLAAttention`` and the multimodal encoder's
    ``MMEncoderAttention`` all carry per-rank head counts, so GQA, MQA, MLA and
    vision attention are described alike (linear-attention layers are
    ``MambaBase`` and dispatched before this). Totals and KV-head replication
    come from the sibling ``QKVParallelLinear`` when present. MLA's single
    latent KV head is not split by TP: every rank holds it, so it is reported
    as one replicated head.
    """
    num_heads = int(layer.num_heads)
    num_kv_heads = int(layer.num_kv_heads)
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    owner_name = name.rsplit(".", 1)[0]
    qkv = next(
        (
            m
            for n, m in modules.items()
            if n.startswith(owner_name + ".") and isinstance(m, QKVParallelLinear)
        ),
        None,
    )
    if isinstance(layer, MLAAttention):
        total_q, total_kv, replicas = num_heads * tp_size, 1, tp_size
    elif qkv is not None:
        total_q, total_kv = int(qkv.total_num_heads), int(qkv.total_num_kv_heads)
        replicas = int(qkv.num_kv_head_replicas)
    else:
        total_q, total_kv, replicas = num_heads * tp_size, num_kv_heads * tp_size, 1
    kv_heads: Range = (
        (tp_rank // replicas, tp_rank // replicas + 1)
        if replicas > 1
        else (tp_rank * num_kv_heads, (tp_rank + 1) * num_kv_heads)
    )
    return AttentionSharding(
        kind=_attention_kind(layer),
        q_heads=(tp_rank * num_heads, (tp_rank + 1) * num_heads),
        kv_heads=kv_heads,
        kv_replicated=replicas > 1,
        total_q_heads=total_q,
        total_kv_heads=total_kv,
        kv_token_shard=dcp if dcp and dcp[1] > 1 else None,
    )


def _attention_kind(layer: AttentionLayerBase | MMEncoderAttention) -> str:
    """``cross`` for encoder-decoder attention, ``sliding`` when the layer
    attends over a window (sliding-window and chunked local attention), else
    ``full``. Read from what ``Attention`` was built with; MLA and the
    multimodal encoder's attention are always full."""
    if not isinstance(layer, Attention):
        return "full"
    if layer.attn_type == AttentionType.ENCODER_DECODER:
        return "cross"
    return "sliding" if layer.sliding_window else "full"


def _moe_sharding(layer: RoutedExperts) -> MoeSharding:
    """``expert_map[global_id]`` is the local slot or ``-1``; it is ``None``
    when expert parallelism is off and every expert is resident."""
    parallel = layer.moe_config.moe_parallel_config
    total = int(layer.global_num_experts)
    expert_map = layer.expert_map_manager.expert_map
    if parallel.enable_eplb:
        local: tuple[int, ...] = ()
    elif expert_map is None:
        local = tuple(range(total))
    else:
        local = tuple(int(i) for i in (expert_map >= 0).nonzero().flatten().tolist())
    return MoeSharding(
        ep_rank=int(parallel.ep_rank),
        ep_size=int(parallel.ep_size),
        tp_rank=int(parallel.tp_rank),
        tp_size=int(parallel.tp_size),
        global_num_experts=total,
        local_experts=local,
        placement="eplb" if parallel.enable_eplb else "static",
    )
