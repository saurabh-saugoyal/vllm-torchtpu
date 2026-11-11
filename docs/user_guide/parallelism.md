# Parallelism & Scaling

`vllm-torchtpu` supports **6D composable parallelism** across single-host and multi-host Cloud TPU slices:

- **Tensor Parallelism (TP)**: `--tensor-parallel-size`
- **Data Parallelism (DP)**: `--data-parallel-size`
- **Pipeline Parallelism (PP)**: `--pipeline-parallel-size`
- **Expert Parallelism (EP)**: `--enable-expert-parallel`
- **Prefill Context Parallelism (PCP)**: `--prefill-context-parallel-size`
- **Decode Context Parallelism (DCP)**: `--decode-context-parallel-size`

---

## 1. Parallelism Dimensions Overview

| Dimension | CLI Flag | Primary Use Case | Mesh / Collective Behavior |
| :--- | :--- | :--- | :--- |
| **Tensor Parallelism (TP)** | `--tensor-parallel-size N` | Shard attention heads and dense MLP projections within a high-bandwidth ICI domain. | All-reduce / Reduce-scatter across `TP` ranks per layer. |
| **Data Parallelism (DP)** | `--data-parallel-size N` | Replicate attention layers across independent engines while sharing a global Expert Parallel MoE mesh (`TP * DP` experts). | Zero attention collective overhead when `TP=1`; All-to-All across `TP * DP` for MoE layers when `--enable-expert-parallel` is enabled. |
| **Expert Parallelism (EP)** | `--enable-expert-parallel` | Shard Mixture-of-Experts (MoE) routed experts across the `TP * DP` device mesh instead of slicing each expert's weights. | Uses Pallas SparseCore / GMM ragged dispatch and combine collectives across the EP mesh (`src/vllm_torchtpu/distributed/ep_mesh.py`). |
| **Pipeline Parallelism (PP)** | `--pipeline-parallel-size N` | Partition transformer layers sequentially across stages (`PP` stages). | Point-to-point activation transfers with dynamic microbatch chunking (`TPU_PP_LONE_PROMPT_CHUNKS`, `TPU_PP_EQUAL_COST_CHUNKS`). |
| **Prefill Context Parallelism (PCP)** | `--prefill-context-parallel-size N` | Split long prefill sequences across `PCP` ranks to parallelize prefill compute and reduce time-to-first-token (TTFT). | Topology-aware ring/mesh assignment (`TPU_PCP_TOPOLOGY_AWARE_MESH=1`) minimizing ICI hop count between consecutive PCP ranks. |
| **Decode Context Parallelism (DCP)** | `--decode-context-parallel-size N` | Shard the KV cache along the sequence dimension across `DCP` ranks for long-context decode. | Online LSE (log-sum-exp) attention merge across `DCP` ranks (`TPU_DCP_LSE_MERGE_A2A`). |

> [!IMPORTANT]
> **Total Device Count**: For an in-process or multi-host deployment, the total number of logical TPU devices consumed by an engine group is `DP * TP * PP * PCP`. On **TPU v7x (Ironwood)**, each physical chip exposes **2 logical TensorCore devices**, so a 4-chip host provides **8 logical devices** (for example, `TP=8`, `DP=8, TP=1`, or `DP=4, TP=2`).

---

## 2. Choosing Between `TP` and `DP + EP` for MoE Models

For large Mixture-of-Experts (MoE) models with GQA or Multi-head Latent Attention (MLA)—such as `Qwen/Qwen3.5-397B-A17B`, `deepseek-ai/DeepSeek-V4`, or `zai-org/GLM-5.2-FP8`—combining **Data Parallelism (`DP`)** with **Expert Parallelism (`--enable-expert-parallel`)** often delivers significantly higher throughput than pure Tensor Parallelism (`TP`):

- **Attention stays chip-local (`TP=1` or `TP=2`)**: Avoiding wide 8-way or 16-way attention all-reduces on every layer.
- **MoE experts shard across the full slice (`EP = TP * DP`)**: Every device holds `num_experts / (TP * DP)` experts, keeping full HBM capacity for weights and KV cache.

### Example: `DP=8, TP=1 + EP` on a Single TPU v7x Host (8 Devices)

```bash
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 10240 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --block-size 256
```

### Example: `DP=4, TP=2 + EP` Hybrid Layout

When per-device HBM is tighter or higher single-stream concurrency is needed, pair `TP=2` (across the 2 logical cores on the same physical TPU v7x chip) with `DP=4`:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --async-scheduling
```

---

## 3. Hierarchical Expert Parallelism & MoE Kernel Tuning

On multi-chip and multi-host slices, `vllm-torchtpu` provides specialized MoE execution paths controlled via environment variables:

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `USE_MOE_SPARSE_CORE` | `1` | Offloads MoE token routing/gather-reduce collectives to TPU SparseCores. |
| `USE_MOE_FUSED_EP_KERNEL` | `0` | Enables the fused Expert-Parallel Pallas kernel (`v1` or `v2` via `MOE_FUSED_EP_KERNEL_IMPL`). |
| `TPU_MOE_HIERARCHICAL_EP` | `0` | Enables two-level (intra-chip / inter-chip) hierarchical EP token dispatch on dual-core TPU v7x topologies. |
| `TPU_EXPERT_PLACEMENT` | `""` | Static expert placement: a JSON table (local path or `gs://` URI) giving, per MoE layer, which experts each EP rank holds, as a permutation that evens out their load. Applied at weight load; works with every MoE kernel. |
| `TPU_MOE_SEQUENCE_PARALLEL` | `0` | Keeps activations sequence-parallel entering MoE layers to replace all-reduce with reduce-scatter + all-gather. |
| `TPU_LATENT_PROJ_INTRA_CHIP_TP` | `0` | Shards MLA latent projections across the 2 cores of each physical TPU v7x chip even when `TP=1`. |

---

## 4. Pipeline Parallelism (`--pipeline-parallel-size`)

Pipeline Parallelism partitions model layers across `PP` stages and is composable with `TP`:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --max-model-len 8192
```

Two chunking policies, both on by default, keep pipeline bubbles small; each has its own switch. `TPU_PP_LONE_PROMPT_CHUNKS` spreads prefill work that cannot fill every stage (for example, a lone prompt) over the stages in smaller steps. `TPU_PP_EQUAL_COST_CHUNKS` sizes prefill chunks so every step takes about the same time, from step costs each stage times at startup. Shortened steps pad to extra token buckets compiled at startup.

The stages hand activations on point to point: each forward's output goes to the next stage as soon as it is done, which keeps the stages busy when prompts of different lengths are in flight. The engine keeps half as many steps again in flight as the pipeline has stages (twelve for eight stages), so stage 0 has a forward queued while the engine core schedules the next step. The hand-off needs libtpu 0.0.50 or later with torch_tpu 0.2.0 or later; the start-up self-test fails fast otherwise. `TPU_PP_HANDOFF_DEPTH` (default `2`) caps the forwards a stage queues for sending: each device holds up to that many outgoing buffers and up to 7 landing buffers for receives posted ahead of their forwards, each `--max-num-batched-tokens` x hidden size in the model dtype, and that memory is reserved before the KV cache is sized.

A pipeline stage may also span a prefill context-parallel group (`--prefill-context-parallel-size`), splitting each chunk's attention across the stage's cores (MoE models then require `--enable-expert-parallel`). On dual-core TPU v7x hosts, a two-core stage must not bind the two cores of a single physical chip: set `VLLM_TPU_DEVICE_ORDER` to the device each local rank binds (for example, `0,2,1,3,4,6,5,7` for four two-core stages on one host).

---

## 5. Context Parallelism (`PCP` and `DCP`)

For long-context workloads (32K–256K+ tokens) or Prefill/Decode disaggregated serving:

- **Prefill Context Parallelism (`--prefill-context-parallel-size`)**: Distributes prefill sequence chunks across `PCP` ranks. `TPU_PCP_TOPOLOGY_AWARE_MESH=1` (enabled by default) binds consecutive PCP ranks to adjacent physical chips on the ICI torus to minimize ring-attention hop latency.
- **Decode Context Parallelism (`--decode-context-parallel-size`)**: Distributes KV cache pages across `DCP` ranks during decode.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 65536
```

### PCP with PCP manager

Set `USE_TPU_PCP_MANAGER=1` to combine `PCP` and `DCP` on sparse MLA models. Each prefill in a step is split over the `PCP` ranks: rank `r` holds chunks `r` and `2P-1-r` of the prompt to balance causal attention. For attention, every rank gathers those prefills' KV from the `DCP`-sharded cache into a workspace. Decode requests are replicated on every `PCP` rank and run the regular `DCP` decode path.

```bash
USE_TPU_PCP_MANAGER=1 vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 4 \
  --decode-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 4 \
  --enable-expert-parallel
```

Requirements:

- A sparse MLA model.
- `--decode-context-parallel-size` equal to `--prefill-context-parallel-size` x `--tensor-parallel-size`.
- `--cp-kv-cache-interleave-size` a multiple of `4` (vLLM defaults to `1`).
- Experimental features. Do not support speculative decoding, LoRA, pipeline parallelism, data parallelism, multimodal inputs, pooling models, and KV connectors (`--kv-transfer-config`, including disaggregated serving and KV offloading) right now.
- MoE models also need `--enable-expert-parallel`.

Sizing:

- `TPU_PCP_KV_WORKSPACE_TOKENS` (default `1048576`): if the prefills in a step need more context tokens than this, the runner runs the step as several forwards. Must be at least `--max-model-len`. A prefill forward allocates a temporary workspace of this many tokens, reused across layers: about 0.6 GB for the MLA latents and 0.27 GB for the indexer K at the default. KV cache sizing does not reserve it yet, so leave room for it in `--gpu-memory-utilization`.
- `TPU_PCP_KV_CHUNK_TOKENS` (default `65536`): context tokens moved by each all-gather.

---

## 6. Observability: which rank holds what

Once the model is loaded and before the server accepts requests, `vllm-torchtpu`
publishes three static Prometheus info gauges (value `1`, the same idiom as
vLLM's `vllm:cache_config_info`) on the regular `/metrics` endpoint. They are
set once and never change for the life of the process, so a scraper keeps them
for as long as its retention allows, which makes them usable for post-mortem
analysis of a deployment that no longer exists. Each gauge has one unit of
cardinality; an empty label value means *not applicable to this row*, never
unknown.

| Gauge | One series per | Key |
| :--- | :--- | :--- |
| `vllm:parallel_config_info` | engine (data-parallel rank) | `engine` |
| `vllm:rank_placement_info` | worker rank | `engine, rank` |
| `vllm:layer_placement_info` | rank × group of identically sharded layers | `engine, rank, module, layer_subset` |

```text
vllm:parallel_config_info{engine="0", tensor_parallel_size="4", pipeline_parallel_size="1",
  data_parallel_size="1", enable_expert_parallel="False", prefill_context_parallel_size="1",
  decode_context_parallel_size="1", world_size="4", nnodes="1", device_kind="TPU v6e",
  visible_chips="4", cores_per_chip="1"} 1

vllm:rank_placement_info{engine="0", rank="1", tp_rank="1", pp_rank="0", pcp_rank="0", dcp_rank="0",
  node_rank="0", device_id="0", chip_index="0", core_on_chip="0", chip_coords="0,0,0",
  host_visible_chips="", layers="0-31"} 1

vllm:layer_placement_info{engine="0", rank="1", module="model", layer_subset="0-31",
  attention_kind="full", q_heads="8-15", kv_heads="2-3", kv_replicated="False", kv_token_shard="",
  linear_state="", linear_state_elements="", mlp_intermediate="3584-7167", vocab_rows="",
  ep_rank="", ep_size="", moe_tp_rank="", moe_tp_size="", expert_placement="", total_experts="",
  local_expert_count="", local_experts="", shared_expert_intermediate=""} 1
vllm:layer_placement_info{engine="0", rank="1", module="lm_head", layer_subset="",
  vocab_rows="37984-75967", ...} 1
```

A homogeneous model has one layer series per rank for its transformer layers
plus one for `lm_head`. A hybrid or multimodal model has one per distinct layer
type (`count by (engine, rank) (vllm:layer_placement_info)` shows how many). For
example, Qwen3.5-4B at TP=4 (three Gated-DeltaNet layers per full-attention
layer) gives each rank two transformer-layer series:

```text
vllm:layer_placement_info{engine="0", rank="1", module="model", layer_subset="0-2,4-6,…,28-30",
  attention_kind="GDN_ATTN", linear_state="3x1x2048|8x128x128", linear_state_elements="137216",
  mlp_intermediate="2304-4607", …} 1
vllm:layer_placement_info{engine="0", rank="1", module="model", layer_subset="3,7,11,…,31",
  attention_kind="full", q_heads="4-7", kv_heads="1", kv_replicated="True", mlp_intermediate="2304-4607", …} 1
```

To put chip and layers side by side, join on the rank:

```promql
vllm:layer_placement_info * on(engine, rank) group_left(device_id, chip_coords, node_rank) vllm:rank_placement_info
```

The same facts are logged: every worker writes its own record as one JSON line
prefixed `rank_placement` as soon as it has described itself in `load_model()`,
before KV-cache sizing and compilation, so a rank that fails later, or a pod
that dies before its first scrape, has still left its placement in the log; the
API server logs the aggregate once more when it publishes the gauges.

### Device identity

`device_id`, `chip_index` and `chip_coords` are in the engine's slice-wide
runtime numbering: every worker of an engine sees the same slice, and ranks map
onto it non-trivially (rank 0 is often not device 0), which is the reason to
publish it. `host_visible_chips` is the host chip the worker process is pinned
to (`TPU_VISIBLE_CHIPS`, the `/dev/vfio/N` index) and `node_rank` the host.
Under multi-process data parallelism each engine's runtime renumbers its subset
of chips from 0, so two engines on one host report the same `device_id`s and
differ in `host_visible_chips`:

| engine | rank | tp_rank | device_id | chip_index | chip_coords | host_visible_chips |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0 | 0 | 2 | 1 | 0,1,0 | 0 |
| 0 | 1 | 1 | 0 | 0 | 0,0,0 | 1 |
| 0 | 2 | 2 | 1 | 2 | 1,0,0 | 2 |
| 0 | 3 | 3 | 3 | 3 | 1,1,0 | 3 |
| 1 | 0 | 0 | 2 | 1 | 0,1,0 | 4 |
| 1 | 3 | 3 | 3 | 3 | 1,1,0 | 7 |

(Qwen2.5-1.5B, DP=2 × TP=4 on one v6e 2x4 host.)

Per-chip telemetry is keyed by the host chip. On GKE the
`kubernetes_io:container_accelerator_*` metrics carry
`accelerator_id="<slice id>-<host chip>"`, whose suffix is that index: for a
pinned worker it equals `host_visible_chips`, and for an engine that sees the
whole host the runtime numbers the chips in host order, so it equals
`device_id` (a TP=4 engine on an idle 8-chip host took `device_id` 0–3 and GKE
reported memory on `…-0` to `…-3`). Telemetry keyed by chip within the visible
slice maps to `chip_index`. To join in PromQL, lift the suffix into a label:

```promql
label_replace(kubernetes_io:container_accelerator_memory_used,
              "host_visible_chips", "$1", "accelerator_id", ".*-([0-9]+)$")
```

### `vllm:rank_placement_info` labels

| Label | Meaning |
| :--- | :--- |
| `host_visible_chips` | Host chip(s) the worker process is pinned to (`TPU_VISIBLE_CHIPS`); one chip per worker under multi-process DP (`0`…`7` on an 8-chip host), empty when the whole host is visible. The physical identity of the chip, independent of per-engine renumbering. |
| `device_id`, `chip_coords`, `core_on_chip`, `chip_index` | The logical TPU device the rank is pinned to in the engine's slice numbering, its ICI coordinates, and which TensorCore of which physical chip that is. On TPU v7x two ranks share a `chip_index`. |
| `pcp_rank`, `dcp_rank` | Position in the prefill-context-parallel group (which chunk of a long prompt's tokens this rank processes) and in the decode-context-parallel group (which shard of the KV cache it attends over; DCP reuses TP ranks). |
| `node_rank` | Host of the engine's slice this rank runs on (`--nnodes` multi-host); `0` on single-host engines. |
| `layers` | Transformer layers held by this pipeline stage (inclusive range, from the stage's own layer bounds). Authoritative for what a stage holds; `layer_subset` below partitions only the layers that contain a described module. |

### `vllm:layer_placement_info` labels

Ranges are the sizes the built modules actually hold, i.e. padded sizes when the
platform pads heads or intermediate dimensions.

| Label | Meaning |
| :--- | :--- |
| `module`, `layer_subset` | The root module (`model`, `language_model`, `visual`, `draft`, `lm_head`, ...) and the subset of its layers whose sharding this series describes, e.g. `0-27` or `1,3,5-7`. Layers with identical sharding are merged, so homogeneous models have one series per rank; a vision tower, a speculative-decoding draft model (EAGLE, MTP, DFlash) or `lm_head` never merges with the language model. `lm_head` has no layers (`layer_subset=""`). |
| `attention_kind` | `full` for softmax attention with a KV cache; `sliding` for sliding-window / chunked local attention (Gemma 2/3, GPT-OSS, Llama 4 — these layers form their own `layer_subset`); `cross` for encoder-decoder attention; or vLLM's linear-attention backend name (`GDN_ATTN` for Gated DeltaNet in Qwen3.5 / Qwen3-Next, `MAMBA2`, ...) for layers that keep a recurrent state instead. Empty for a group without attention. |
| `q_heads`, `kv_heads` | Attention heads this tensor-parallel rank computes (inclusive ranges, from `QKVParallelLinear`). `kv_replicated="True"` means `num_kv_heads < tp_size` and the KV head is held by several TP ranks; MLA models (DeepSeek, Kimi) always report their one latent KV head this way. |
| `kv_token_shard` | Under decode context parallelism the KV *tokens* are additionally split: `1/2` means this rank attends over the second half of each sequence's KV cache. Empty without DCP. |
| `linear_state`, `linear_state_elements` | For linear-attention / state-space layers the rank holds a slice of the layer's recurrent state instead of KV heads. `linear_state` is the per-rank state shapes from `get_state_shape()`, e.g. `3x1x2048|8x128x128` (conv state, SSM state: 8 of 32 value heads on this rank) — one value for identity, not a list of filterable facts; `linear_state_elements` is the per-rank element count, comparable across ranks. |
| `mlp_intermediate` | Columns of the dense MLP intermediate dimension this rank holds (its `gate_proj`/`up_proj` rows and `down_proj` columns). |
| `vocab_rows` | Rows of the vocabulary projection (`lm_head`) this rank holds; a large share of HBM for big vocabularies. |
| `ep_rank`, `ep_size`, `moe_tp_rank`, `moe_tp_size` | The MoE layer's own parallel split, read from the built layer: expert parallelism (size = TP × DP × PCP, so not a CLI flag) and the MoE tensor split, which under hierarchical MoE (`TPU_MOE_HIERARCHICAL_EP`: EP = chips, MoE-TP = cores per chip) differs from attention TP. |
| `expert_placement`, `total_experts`, `local_expert_count`, `local_experts` | `static`: the ids resident on this rank are listed, e.g. `60` / `8` / `0-7` under `--enable-expert-parallel` or `0-59` on every rank without EP (compressed id list, so uneven and round-robin placements are representable; dropped above 1024 bytes, the count stays). `eplb`: the map is rebalanced at runtime, so only the counts are published. |
| `shared_expert_intermediate` | MoE blocks with always-on shared expert(s) (Qwen1.5/2-MoE, DeepSeek, Llama 4) hold them as one fused dense MLP that is tensor-parallel-sharded like `mlp_intermediate`, and replicated across data-parallel engines. The range is this rank's slab of its intermediate width; e.g. Qwen1.5-MoE's `0-1407` of 5632 is one quarter of the 4 x 1408 shared experts described in its paper. |

Set `VLLM_TPU_EMIT_PLACEMENT=0` to disable. `vllm:parallel_config_info` is
skipped automatically on vLLM versions that publish it themselves. With
`--api-server-count > 1` every API-server process publishes the same values;
the multiprocess registry collapses them into one series per label set.
