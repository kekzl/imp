# VRAM & RAM Management: imp vs. llama.cpp vs. Ollama vs. vLLM

A comparison of memory management strategies across four LLM inference engines.

---

## 1. Architecture Overview

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Language | C++20 / CUDA | C / C++ / CUDA | Go + llama.cpp (C++) | Python + PyTorch + CUDA |
| GPU Support | CUDA `sm_120f` only (Blackwell consumer + workstation: RTX 5090, RTX PRO 5000 / 6000) | CUDA, Metal, Vulkan, ROCm, SYCL | Inherits from llama.cpp | CUDA, ROCm, TPU |
| Focus | Max single-GPU perf on Blackwell | Broad HW compatibility | Ease of use | Max serving throughput |
| Multi-GPU | No | Layer/Row/Graph split | Layer split | Tensor/Pipeline/Expert parallelism |

---

## 2. Model Loading & Host RAM

### 2.1 File Access

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Method | `mmap()` read-only (GGUF + SafeTensors) | `mmap()` + `MAP_SHARED` | Inherits from llama.cpp | `torch.load()` / Safetensors |
| Hints | `MAP_POPULATE` + `MADV_WILLNEED` + `MADV_SEQUENTIAL` (cold-cache prefault) | `mlock()` optional | — | — |
| Disableable | No | `--no-mmap` | No | N/A (no mmap) |
| Zero-copy | Yes (until upload) | Yes (CPU layers stay in mmap) | Yes | No (PyTorch copies) |

**imp / llama.cpp / Ollama** memory-map GGUF files directly into the address space — no RAM consumption until pages are actually read. Weights remain in mmap until GPU upload, after which the OS can evict the pages.

**vLLM** does not use mmap. Models are loaded via PyTorch/Safetensors and materialized directly into GPU tensors. Higher peak RAM during loading, but host RAM is released immediately afterwards.

### 2.2 Weight Upload & Dequantization

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Strategy | Host dequant → GPU upload | Direct quantized to GPU, CPU repacking | Inherits from llama.cpp | PyTorch → GPU, optional quant |
| Q4_0 | → packed nibbles + scales (2 GPU allocs) | → Q4_X_X CPU repacking (AVX-512/NEON) | Inherits | — (own quant formats) |
| Q8_0 | Raw to GPU OR host→FP16→GPU | Raw to GPU | Inherits | — |
| F32 | → FP16 on host → GPU | Direct | Inherits | Direct as BF16/FP16 |
| FP8/NVFP4 | Native GPU formats | Not supported | Not supported | FP8 via quantization frameworks |

**imp** deliberately dequantizes on the host before upload (for formats where this is cheaper than GPU dequant), and checks available VRAM before every allocation (256 MiB headroom). **vLLM** relies on PyTorch's memory allocator and a profiling run.

### 2.3 Pinned Memory

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Pool | 4 × 128 MiB `cudaMallocHost` ring (weight upload pipeline) | Output buffer only (~2 MB) | Inherits | PyTorch-managed, swap space pinned |
| Sub-allocation | Bump-pointer + size-class free lists (256B aligned) | None | — | None (PyTorch internal) |
| Fallback | Direct `cudaMallocHost` when pool exhausted | — | — | — |

**imp** has a dedicated pinned memory pool with its own sub-allocator for fast CPU↔GPU transfers. The other engines use pinned memory only sparingly.

---

## 3. VRAM Allocation

### 3.1 GPU Allocator

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Method | `cudaMallocAsync` + `cudaMemPool_t` | `cudaMalloc` + VMM pool | Inherits from llama.cpp | PyTorch CUDA Allocator |
| Stream-ordered | Yes (async alloc/free) | No | No | No (PyTorch-managed) |
| Pool strategy | Release threshold=∞, cross-stream reuse | VMM (`cuMemCreate`/`cuMemMap`) | Inherits | CUDACachingAllocator |
| Fragmentation | Minimal (pool reuse) | VMM = fragmentation-free | Inherits | PyTorch block splitting |

**imp** uses CUDA's native stream-ordered allocator — allocations and frees can overlap with kernel execution without explicit synchronization. **llama.cpp** uses Virtual Memory Management (VMM) APIs for fragmentation-free sub-allocation. **vLLM** relies on PyTorch's `CUDACachingAllocator`, which performs internal block splitting.

### 3.2 VRAM Budgeting

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Method | 80% free-after-weights | Binary search (`llama_params_fit`) | Iterative fit/alloc/commit | Profiling run + `gpu_memory_utilization` |
| Default | Auto-size: 2x minimum KV | Manual via `--gpu-layers` | Automatic | 90% total VRAM |
| Safety | `checked_cuda_malloc` + 256 MiB reserve | Virtual test allocation | Backoff 0.1 per failure | Profiling peak measures exactly |

**vLLM** has the most robust approach: a dummy forward pass measures actual peak activation consumption, and the KV cache gets exactly the remainder. **imp** conservatively reserves 80% of free VRAM after weights. **Ollama** iterates with increasing backoff until allocation succeeds.

### 3.3 Tracking

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Method | Atomic counters + `alloc_map_` | ggml graph allocator ref-counting | Go scheduler tracks per runner | PyTorch `memory_allocated()` + block manager |
| Granularity | Per pointer (size map) | Per tensor (lifetime analysis) | Per model | Per block (KV), per tensor (weights) |
| Peak tracking | `peak_allocated_` atomic | — | — | PyTorch `max_memory_allocated()` |

---

## 4. KV Cache — The Biggest Architectural Difference

### 4.1 Core Architecture

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| **Type** | **Paged blocks** | **Contiguous ring buffer** | **Ring buffer** (inherited) | **Paged blocks** |
| Inspiration | vLLM | Custom | llama.cpp | OS virtual memory |
| Block size | 16 tokens | N/A (no blocks) | N/A | 16 tokens (up to 32) |
| Layout | `[block, slot, kv_head, head_dim]` | Contiguous per layer, K/V separate | Inherited | K: `[blocks, heads, hd/x, bs, x]`, V: `[blocks, heads, hd, bs]` |

### 4.2 Allocation & Data Structures

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Free pool | LIFO stack (free list) | Head-pointer ring | Slot-based | Doubly-linked list (`FreeKVCacheBlockQueue`) |
| Alloc complexity | O(1) pop | O(1) head advance | O(n) prefix match | O(1) pop |
| Block table | `seq_blocks_[seq_id] → [block_ids]` | N/A (position = index) | N/A | `block_table[seq][logical] → physical` |
| Ref counting | Yes (per block) | No (bitset per cell) | No | Yes (per block) |
| Copy-on-write | Possible via ref count | No | No | Yes (on `ref_count > 1` → copy) |

### 4.3 Memory Waste

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| External fragmentation | **0%** (uniform blocks) | Minimal (ring buffer) | Same as llama.cpp | **0%** (uniform blocks) |
| Internal fragmentation | Max 15 tokens/seq (last block) | **60-80%** (pre-alloc for max_seq_len) | Same as llama.cpp | **< 4%** (last block only) |
| Reservation waste | 0% (on-demand alloc) | High (full seq length reserved) | Same as llama.cpp | 0% (on-demand alloc) |

The paged block architecture of **imp** and **vLLM** is dramatically more efficient for serving workloads with many parallel sequences than the ring buffer of **llama.cpp/Ollama**. For single-user inference, the difference is smaller.

### 4.4 Eviction & Cache Management

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Eviction | LRU (doubly-linked list, O(1)) | Context shifting (discard oldest tokens) | LRU at slot level | LRU via free queue position |
| Defragmentation | Not needed (paged) | `--defrag-thold` (graph nodes) | Inherited | Not needed (paged) |
| Prefix caching | Hash → block vector, block sharing via ref count | Bitset-based sequence sharing | Token prefix comparison (`inputsEqual()`) | APC: SHA-256 hash chain → block, LRU eviction |
| Rollback | Block-granular (speculative decoding) | Not directly | Not supported | Sequence length counter not advanced |

**vLLM** has the most sophisticated prefix caching (APC): hash chains over blocks guarantee that block N only matches if all blocks 0..N contain identical tokens. **imp** has a simpler hash→block-vector approach. **llama.cpp** uses bitsets per cell for basic prefix sharing.

### 4.5 KV Cache Quantization

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Formats | FP16 (default), FP8 E4M3, INT8, INT4, NVFP4, TurboQuant | FP16, Q8_0, Q4_0 | FP16, Q8_0, Q4_0 | FP16, BF16, FP8 E4M3, FP8 E5M2 |
| Prerequisite | — | Flash Attention | `OLLAMA_FLASH_ATTENTION=1` | — |
| Calibration | Online per-layer absmax (FP8 KV; warmup-aware monotonic via PR #89) | — | — | Optional per-head scales via `llm-compressor` |
| K/V separate | No (same dtype) | Yes (`-ctk`/`-ctv`) | No (global) | No (same dtype) |
| Savings | 2× (FP8), 4× (INT4), ~3× (TurboQuant ~3 bits/elem average) | 2× (Q8_0), 3× (Q4_0) | Same as llama.cpp | 2× (FP8) |

### 4.6 KV Cache Sizing

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Method | Auto: min(2x minimum, 80% free VRAM) | Manual via `-c` (context length) | Automatic (KV = ctx x heads x dim x parallel) | Profiling: remainder after weights + peak activations |
| Formula | `n_layers x max_blocks x 2 x block_bytes` | `n_layers x ctx_len x n_heads x head_dim x 2 x dtype` | Same as llama.cpp | `num_blocks = available_memory / bytes_per_block` |
| Configurable | Yes (`max_kv_blocks`) | Yes (`-c`, `-ctk`, `-ctv`) | Limited | Yes (`gpu_memory_utilization`, `kv_cache_memory_bytes`) |

---

## 5. CPU/GPU Offloading

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Layer offload | No | `--gpu-layers` (layer split CPU/GPU) | Automatic (Go scheduler) | `--cpu-offload-gb` (weights) |
| KV cache offload | No | No | No | Swap space (4 GiB default) + native KV offloading (DMA, 83.4 GB/s) |
| Tensor-level control | — | `--override-tensor` | — | — |
| Preemption | — | — | — | Yes (lowest priority → swap out → free GPU → swap in on reschedule) |

**vLLM** is the only engine with true KV cache swapping: when VRAM runs low, lower-priority requests are fully offloaded to CPU and later restored. **llama.cpp/Ollama** can offload model layers to CPU, but the KV cache always stays on the target device.

---

## 6. Multi-GPU

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Tensor parallelism | No | `--split-mode row` (limited) | No | Full (NCCL AllReduce per layer) |
| Pipeline parallelism | No | `--split-mode layer` | Layer split | Yes (uneven splits possible) |
| Graph parallelism | No | `--split-mode graph` (NCCL, 3-4x speedup) | No | — |
| Expert parallelism | No | No | No | Yes (all-to-all for MoE) |
| Multi-node | No | No | No | Yes (Ray + NCCL) |
| Config | — | `--tensor-split 0.6,0.4` | `OLLAMA_SCHED_SPREAD` | `-tp N -pp M` |

**vLLM** is built for multi-GPU/multi-node serving. **llama.cpp** has caught up with `--split-mode graph` (NCCL), but only for CUDA. **imp** is intentionally optimized for a single Blackwell GPU (RTX 5090 / RTX PRO 5000 / 6000).

---

## 7. Multi-Model Management

| | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| Concurrent models | No | No | Yes (`OLLAMA_MAX_LOADED_MODELS`, default 3x GPUs) | No (1 model per engine instance) |
| Eviction | — | — | Idle timer (5 min default) → LRU eviction | — |
| VRAM recovery | — | — | Polling until VRAM free (5s timeout) | — |
| Request queue | — | — | FIFO (max 512, `OLLAMA_MAX_QUEUE`) | AsyncIO request queue with scheduling policies |

**Ollama** is the only engine that actively manages multiple models in VRAM, with automatic loading/unloading based on usage.

---

## 8. Special Features

| Feature | **imp** | **llama.cpp** | **Ollama** | **vLLM** |
|---|---|---|---|---|
| CUDA Graphs | Decode captured (incl. NVFP4 prequant MoE end-to-end via `cache_moe_native_nvfp4`) | Delayed activation | Inherited | Decode captured (stable shapes) |
| PDL | `cudaFuncAttributeProgrammaticStreamSerialization` (152+ edges in conditional runner) | No | No | No |
| Green Contexts | SM partitioning for prefill/decode (CUDA 13.2 Green Contexts API) | No | No | No |
| Chunked prefill | No | No | No | Yes (bounded by `max_num_batched_tokens`) |
| Disaggregated prefill | No | No | No | Yes (separate prefill/decode instances) |
| SSM / GDN state | FP32 persistent, per-seq GPU pool (Mamba2, Gated DeltaNet) | `llama_memory_recurrent` | Inherited | Limited |
| Speculative decoding | N-gram opt-in (EAGLE / self-spec / DFlash / TurboDraft all evaluated and dropped — single-5090 decode is bandwidth-bound) | Yes | No | Draft/Eagle/Medusa/MLPSpeculator |
| Native function calling | ChatML / Llama3 / Gemma-4 (`<\|tool_call>`) / Qwen3.6 XML; longest-match special-token pre-split in tokenizer | Limited | Inherits | Yes |

---

## 9. Summary

### imp — Maximum Single-GPU Performance on Blackwell
- **Strengths**: NVFP4 prequant SafeTensors as primary path (200+ tok/s on most MoE), paged KV cache (near 0% waste), stream-ordered CUDA allocation, Blackwell-native features (PDL, Green Contexts, `mma.sync` block-scaled FP4 / FP8 / FP16 MMA on the 5th-gen tensor cores), 4×128 MiB pinned ring for weight upload, native function calling with longest-match special-token pre-split tokenizer.
- **Weaknesses**: No CPU offload, no multi-GPU, no multi-model hot-swap (separate processes per model), no KV swapping. `sm_120f` only — no compatibility matrix.
- **Ideal for**: Dedicated single-GPU inference on RTX 5090 / RTX PRO 5000 / 6000 Blackwell with maximum latency optimization.

### llama.cpp — Universal Compatibility
- **Strengths**: Multi-backend (CUDA/Metal/Vulkan/CPU), CPU offloading, broad quant formats (GGML Q4/Q5/Q6/Q8), mmap zero-copy, graph-based tensor allocator, multi-GPU (layer/row/graph split)
- **Weaknesses**: Ring buffer KV cache (60-80% waste on multi-sequence), no KV swapping, no chunked prefill
- **Ideal for**: Single-user inference on heterogeneous hardware, models that don't fully fit in VRAM

### Ollama — Easiest to Use
- **Strengths**: Multi-model VRAM management, automatic layer offloading, iterative fit allocation, REST API, model registry
- **Weaknesses**: Inherits all llama.cpp KV cache limitations, no tensor parallelism, KV quant global only
- **Ideal for**: Local-first deployment, quickly trying different models, desktop usage

### vLLM — Maximum Serving Throughput
- **Strengths**: PagedAttention (<4% waste), APC with hash-chain prefix caching, KV cache swapping/offloading (83.4 GB/s DMA), tensor/pipeline/expert parallelism, chunked prefill, disaggregated prefill, multi-node (Ray), CUDA Graphs
- **Weaknesses**: Python overhead, CUDA/ROCm/TPU only, high base VRAM usage (PyTorch), no GGML quant formats, no CPU offload of layers
- **Ideal for**: Production serving with high throughput, multi-GPU/multi-node deployments, many parallel requests

---

## 10. Decision Matrix

| Scenario | Recommendation |
|---|---|
| Single Blackwell GPU (RTX 5090 / PRO 5000 / 6000), latency-critical, NVFP4 model | **imp** |
| Datacenter Hopper / Blackwell B200/B300 | **vLLM** or **TensorRT-LLM** (imp doesn't target these) |
| Older consumer GPU (RTX 4090, 3090), large model, partial CPU offload | **llama.cpp** |
| Desktop, trying different models | **Ollama** |
| Production API, 100+ concurrent users | **vLLM** |
| Multi-node cluster, maximum throughput | **vLLM** |
| Edge device, Apple Silicon | **llama.cpp** (Metal) |
| Hybrid SSM+Attention or GDN+MoE models (Mamba2, Qwen3.5/3.6) | **imp** (Blackwell) or **llama.cpp** (other HW) |
