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
"""Typed records describing where a rank runs and which slice of a model it holds.

Vocabulary: a *slice* is every chip the engine's workers can see; a *chip* sits
at ``chip_coords`` in the ICI mesh and holds ``cores_per_chip`` TensorCores; one
core is one logical ``device_id`` and one worker process (``rank``). Under
multi-process data parallelism each engine is pinned to a subset of the host's
chips (``host_visible_chips``) and the runtime numbers devices and coordinates
within that subset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

Range = tuple[int, int]
"""Half-open ``[start, stop)`` interval."""


@dataclass(frozen=True)
class DeviceIdentity:
    """The TPU core a worker process is pinned to."""

    device_id: int | None = None
    device_kind: str | None = None
    chip_coords: tuple[int, ...] | None = None
    core_on_chip: int | None = None
    chip_index: int | None = None
    visible_chips: int | None = None
    cores_per_chip: int | None = None
    host_visible_chips: tuple[int, ...] | None = None


@dataclass(frozen=True)
class AttentionSharding:
    """Heads one tensor-parallel rank computes in an attention layer."""

    q_heads: Range
    kv_heads: Range
    kv_replicated: bool
    total_q_heads: int
    total_kv_heads: int
    gqa_padded: bool = False


@dataclass(frozen=True)
class LinearAttentionSharding:
    """Recurrent state one rank holds in a linear-attention / SSM layer
    (Gated DeltaNet, Kimi Delta Attention, Mamba): ``kind`` is vLLM's
    ``mamba_type`` and ``state_shapes`` its per-rank ``get_state_shape()``."""

    kind: str
    state_shapes: tuple[tuple[int, ...], ...]
    tp_replicated: bool


@dataclass(frozen=True)
class MlpSharding:
    """Columns of the dense MLP intermediate dim one rank holds."""

    intermediate: Range
    total_intermediate: int


@dataclass(frozen=True)
class MoeSharding:
    """Routed experts resident on one rank."""

    ep_rank: int
    ep_size: int
    global_num_experts: int
    local_experts: tuple[int, ...]


@dataclass(frozen=True)
class LayerGroup:
    """Layers of one root module that shard identically on this rank."""

    root: str
    layers: tuple[int, ...]
    attn: AttentionSharding | None = None
    linear_attn: LinearAttentionSharding | None = None
    mlp: MlpSharding | None = None
    moe: MoeSharding | None = None
    shared_mlp: MlpSharding | None = None
    """The MoE block's always-on shared expert(s), fused into one dense MLP."""

    def sharding_key(self) -> tuple:
        return (
            self.root,
            self.attn,
            self.linear_attn,
            self.mlp,
            self.moe,
            self.shared_mlp,
        )


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
    device: DeviceIdentity
    layers: Range | None = None
    layer_groups: tuple[LayerGroup, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParallelSummary:
    """Engine-wide parallel configuration plus the hardware it runs on."""

    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    enable_expert_parallel: bool
    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    world_size: int
    nnodes: int
    kv_role: str | None = None
    device_kind: str | None = None
    visible_chips: int | None = None
    cores_per_chip: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelSpec:
    name: str
    doc: str


# Label value formats: [index] 0-based position; [count] size; [bool]
# "True"/"False"; [range] inclusive "a-b" ("a" if one element); [id list]
# compressed "0-2,5,9-11"; [x,y,z] comma-separated ints; [text] runtime string.
# "" means not applicable. Gauge value is always 1.

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
    LabelSpec(
        "kv_role", "[text] kv_producer | kv_consumer | kv_both under P/D; '' otherwise"
    ),
    LabelSpec("device_kind", "[text] e.g. 'TPU v5 lite', 'TPU v6e'"),
    LabelSpec("visible_chips", "[count] chips the engine's workers can see"),
    LabelSpec("cores_per_chip", "[count] TensorCores per chip: 1 on v5e/v6e, 2 on v7x"),
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
    LabelSpec(
        "ep_rank", "[index] expert-parallel group (spans TP x DP); '' without MoE"
    ),
    LabelSpec("node_rank", "[index] host of the engine's slice, 0..nnodes-1"),
    LabelSpec("device_id", "[index] logical device (core) in the runtime's numbering"),
    LabelSpec(
        "chip_index", "[index] physical chip, 0..visible_chips-1, in coords order"
    ),
    LabelSpec(
        "core_on_chip", "[index] TensorCore within the chip: 0 on v5e/v6e, 0|1 on v7x"
    ),
    LabelSpec("chip_coords", "[x,y,z] position in the visible slice's ICI mesh"),
    LabelSpec(
        "host_visible_chips", "[id list] TPU_VISIBLE_CHIPS pin; '' if whole host"
    ),
    LabelSpec("layers", "[range] layers held by this pipeline stage"),
    LabelSpec("module", "[text] root module of the layers, e.g. 'model', 'visual'"),
    LabelSpec("layer_subset", "[id list] layers of `module` sharing this sharding"),
    LabelSpec("q_heads", "[range] query heads this rank computes"),
    LabelSpec("kv_heads", "[range] KV heads this rank holds; one head when replicated"),
    LabelSpec(
        "kv_replicated", "[bool] num_kv_heads < TP: a KV head is shared by ranks"
    ),
    LabelSpec("gqa_padded", "[bool] Q heads zero-padded to divide by TP"),
    LabelSpec(
        "linear_attention", "[text] GDN | KDA | MAMBA2 ... for linear-attention layers"
    ),
    LabelSpec(
        "linear_state", "[shapes] per-rank recurrent state, e.g. '4x128x128|4x4x128'"
    ),
    LabelSpec("mlp_intermediate", "[range] dense MLP intermediate slab on this rank"),
    LabelSpec("total_experts", "[count] routed experts in the layer; '' without MoE"),
    LabelSpec(
        "local_experts",
        "[id list] routed experts resident on this rank; '' without MoE",
    ),
    LabelSpec(
        "shared_expert_intermediate",
        "[range] intermediate slab of the MoE block's fused shared expert(s)",
    ),
)


def label_names(specs: tuple[LabelSpec, ...]) -> list[str]:
    return [spec.name for spec in specs]
