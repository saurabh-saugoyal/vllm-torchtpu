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
| **Pipeline Parallelism (PP)** | `--pipeline-parallel-size N` | Partition transformer layers sequentially across stages (`PP` stages). | Point-to-point activation transfers with dynamic microbatch chunking (`TPU_PP_DYNAMIC_CHUNKS=1`). |
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

By default, `TPU_PP_DYNAMIC_CHUNKS=1` dynamically sizes microbatch chunks to keep pipeline bubbles minimal without recompiling static token buckets.

The stages hand activations on in lockstep by default (`TPU_PP_HANDOFF=wave`): one collective per forward on every stage, so each round lasts as long as the slowest forward in flight. `TPU_PP_HANDOFF=ring` sends each forward's output point to point to the next stage as soon as it is done, which keeps the stages busy when prompts of different lengths are in flight. Under the ring the engine keeps half as many steps again in flight as the pipeline has stages (twelve for eight stages), so stage 0 has a forward queued while the engine core schedules the next step; the wave keeps vLLM's one step per stage. It needs libtpu 0.0.50 or later with torch_tpu 0.2.0 or later; the start-up self-test fails fast otherwise. `TPU_PP_RING_DEPTH` (default `2`) caps the forwards a stage queues for sending: each device holds up to that many outgoing buffers and up to 7 landing buffers for receives posted ahead of their forwards, each `--max-num-batched-tokens` x hidden size in the model dtype, and that memory is reserved before the KV cache is sized.

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

---

## 6. Observability: which rank holds what

Once the model is loaded and before the server accepts requests, `vllm-torchtpu`
publishes two static Prometheus info gauges (value `1`, the same idiom as vLLM's
`vllm:cache_config_info`) on the regular `/metrics` endpoint. They are set once
and never change for the life of the process, so a scraper keeps them for as
long as its retention allows, which makes them usable for post-mortem analysis
of a deployment that no longer exists.

```text
# One series per engine (data-parallel rank).
vllm:parallel_config_info{engine="0", tensor_parallel_size="4", pipeline_parallel_size="1",
  data_parallel_size="1", enable_expert_parallel="False", prefill_context_parallel_size="1",
  decode_context_parallel_size="1", world_size="4", device_kind="TPU v6e",
  visible_chips="4", cores_per_chip="1"} 1

# One series per rank and layer group: where it runs and which slice of the
# model it holds. A homogeneous model has one group per rank; a hybrid model
# (e.g. alternating full/linear attention, or dense-then-MoE layers) has one
# series per distinct layer type, identified by `layer_subset`.
vllm:tpu_placement_info{engine="0", rank="1", tp_rank="1", pp_rank="0", pcp_rank="0", dcp_rank="0", node_rank="0", ep_rank="",
  device_id="1", chip_index="1", core_on_chip="0", chip_coords="1,0,0", host_visible_chips="",
  layers="0-31", module="model", layer_subset="0-31", q_heads="8-15", kv_heads="2-3", kv_replicated="False",
  gqa_padded="False", linear_attention="", linear_state="", mlp_intermediate="3584-7167",
  total_experts="", local_experts="", shared_expert_intermediate=""} 1
```

| Label | Meaning |
| :--- | :--- |
| `host_visible_chips` | Host chips the process may use (`TPU_VISIBLE_CHIPS`), e.g. `4-7` for the second engine of `DP=2` on an 8-chip host; empty when the whole host is visible. Under multi-process DP the runtime renumbers `device_id`/`chip_index`/`chip_coords` within this subset, so this is what distinguishes engines physically. |
| `device_id`, `chip_coords`, `core_on_chip`, `chip_index` | The logical TPU device the rank is pinned to, its ICI coordinates, and which TensorCore of which physical chip that is. On TPU v7x two ranks share a `chip_index`. |
| `pcp_rank`, `dcp_rank` | Position in the prefill-context-parallel group (which chunk of a long prompt's tokens this rank processes) and in the decode-context-parallel group (which shard of the KV cache it attends over; DCP reuses TP ranks). |
| `node_rank` | Host of the engine's slice this rank runs on (`--nnodes` multi-host); `0` on single-host engines. |
| `layers` | Transformer layers held by this pipeline stage (inclusive range). |
| `module`, `layer_subset` | The root module (`model`, `visual`, `draft`, ...) and the subset of its layers whose sharding this series describes, e.g. `0-27` or `1,3,5-7`. Layers with identical attention/MLP/MoE sharding are merged, so homogeneous models have one series per rank; a vision tower or draft model never merges with the language model. |
| `q_heads`, `kv_heads` | Attention heads this tensor-parallel rank computes (inclusive ranges, from `QKVParallelLinear`). `kv_replicated="True"` means `num_kv_heads < tp_size` and the single KV head is shared by several ranks. `gqa_padded="True"` means the Q-head count was zero-padded to divide by TP, so the range is in padded head indices. |
| `linear_attention`, `linear_state` | For linear-attention / state-space layers (Gated DeltaNet in Qwen3.5 and Qwen3-Next, Kimi Delta Attention, Mamba) there are no KV heads; instead the rank holds a slice of the layer's recurrent state. `linear_attention` is vLLM's `mamba_type` (`GDN`, `KDA`, `MAMBA2`, ...) and `linear_state` the per-rank state shapes from `get_state_shape()`, e.g. `256x4|8x128x128` (conv state, SSM state). In a hybrid model these layers form their own `layer_subset`, separate from the full-attention layers. |
| `mlp_intermediate` | Columns of the dense MLP intermediate dimension this rank holds (its `gate_proj`/`up_proj` rows and `down_proj` columns). |
| `total_experts`, `local_experts` | Number of routed experts in the layer and the ids resident on this rank, e.g. `60` and `0-7` under `--enable-expert-parallel` (`ep_size = tp_size x dp_size`), or `0-59` on every rank without EP. Read from the routed-experts layer's expert map. |
| `shared_expert_intermediate` | MoE blocks with always-on shared expert(s) (Qwen1.5/2-MoE, DeepSeek, Llama 4) hold them as one fused dense MLP that is tensor-parallel-sharded like `mlp_intermediate`, and replicated across data-parallel engines. The range is this rank's slab of its intermediate width; e.g. Qwen1.5-MoE's `0-1407` of 5632 is one quarter of the 4 x 1408 shared experts described in its paper. |

The same payload is also written once to the server log as a single JSON line
prefixed `tpu_placement`, for installations without a metrics scraper.

Set `VLLM_TPU_EMIT_PLACEMENT=0` to disable. `vllm:parallel_config_info` is
skipped automatically on vLLM versions that publish it themselves.
