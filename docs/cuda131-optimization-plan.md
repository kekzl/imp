# Next-Gen CUDA 13.1 Optimization Plan for imp

## Context

imp currently achieves 80-94% of llama.cpp decode throughput on dense models (Q8_0/Q6_K) and 287% on MoE (Nemotron) on RTX 5090 (sm_120). The engine already uses CUDA Graphs, PDL, Green Contexts, stream-ordered allocators, WMMA 16x16x16 attention, fused QKV/gate+up GEMV, and dp4a quantized paths. The gap to llama.cpp on dense models is primarily in the decode GEMV kernels (memory-bandwidth-bound) and prefill GEMM (cuBLAS auto-tuning vs hand-tuned). This plan targets closing that gap and pushing beyond by exploiting advanced CUDA 13.1 / Blackwell features.

## Analysis: What Actually Matters on RTX 5090

### RTX 5090 Profile
- **Memory bandwidth**: ~1.8 TB/s (GDDR7) — NOT HBM, so bandwidth is precious
- **Shared memory**: 100 KB/SM, 99 KB opt-in max
- **Tensor cores**: FP16 WMMA, FP8 tensor ops, NVFP4
- **Consumer GPU**: No NVLink, no multi-GPU, single-card only

### Bottleneck Analysis by Phase

**Decode (single-token, memory-bound)**:
- Each layer = RMSNorm + QKV GEMV + RoPE + KV write + PagedAttn + Wo GEMV + RMSNorm + gate+up GEMV + SwiGLU + down GEMV
- ~90% of decode time is GEMV (weight loading from GDDR7)
- Kernel launch overhead matters: 30-50 layers x ~10 kernels = 300-500 launches/token
- **Primary bottleneck: memory bandwidth utilization in GEMV**

**Prefill (batch tokens, compute-bound)**:
- GEMM (cuBLAS) + Flash Attention
- cuBLAS is already well-tuned for large M
- **Primary bottleneck: attention for long sequences, GEMM for short**

### Feature Impact Assessment

| Feature | Impact for LLM Decode | Impact for LLM Prefill | Verdict |
|---------|----------------------|----------------------|---------|
| **TMA** | Low (GEMV is simple row loads) | Medium (tiled GEMM loads) | Phase 2 — prefill attention only |
| **wgmma** | None (GEMV doesn't use MMA) | High (larger tiles = higher utilization) | Phase 2 — prefill attention |
| **Clusters + DSMEM** | Medium (share KV across heads) | Medium (share tiles across blocks) | Phase 3 — attention only |
| **Persistent GEMV** | **HIGH** (eliminates launch overhead) | N/A | **Phase 1 — biggest win** |
| **NVRTC JIT** | Medium (shape-specialized kernels) | Medium (tile specialization) | Phase 4 — incremental |
| **mbarrier pipelines** | Medium (overlap compute/load) | High (multi-stage GEMM) | Phase 2 — attention |
| **cp.async bulk** | Low (GEMV loads are coalesced) | Medium (larger async copies) | Phase 2 — with TMA |

## Phase 0: Decode GEMV Bandwidth Optimization (Highest ROI)

**Why**: Decode is 100% memory-bandwidth-bound. The gap vs llama.cpp is exactly here. Before adding exotic features, maximize bandwidth utilization in existing GEMV kernels.

**Files**: `src/compute/gemm.cu`

### 0.1 Multi-row GEMV (2-4 rows per warp)

Current: 1 warp = 1 output row. Each warp reads the entire input vector independently.
Problem: 8 warps in a block all read the same input vector x — 8x redundant L2 traffic.

**Fix**: Each warp computes 2-4 output rows, amortizing the input vector read.

```cpp
// Current: 1 row per warp (32 threads)
// New: 4 rows per warp, input vector cached in registers
template <int ROWS_PER_WARP>
__global__ void gemv_fp16_multirow_kernel(
    const half* __restrict__ W,  // [M, K]
    const half* __restrict__ x,  // [K]
    half* __restrict__ y,        // [M]
    int M, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int base_row = (blockIdx.x * (blockDim.x / 32) + warp_id) * ROWS_PER_WARP;

    float sums[ROWS_PER_WARP] = {0.f};

    // Iterate over K dimension, load x once, use for all rows
    for (int k = lane * 8; k < K; k += 32 * 8) {
        float4 xv = reinterpret_cast<const float4*>(x)[k / 8];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            if (base_row + r < M) {
                float4 wv = reinterpret_cast<const float4*>(W + (int64_t)(base_row + r) * K)[k / 8];
                // dot product accumulate into sums[r]
            }
        }
    }
    // Warp reduce + store each row
}
```

Expected improvement: 15-25% decode GEMV speedup from reduced L2 traffic.

### 0.2 Shared-memory Input Caching for Quantized GEMV

For Q6_K/Q8_0/Q4_0 GEMV: load the Q8_1-quantized input vector into shared memory once per block, then all warps read from shared instead of L2.

Currently, each warp independently loads q8_1 blocks from global. With 8 warps per block, that's 8x redundant reads of the same input.

```cpp
// At block start:
extern __shared__ block_q8_1 shared_input[];
// Cooperative load: all threads load input into shared
for (int i = tid; i < n_q8_blocks; i += blockDim.x) {
    shared_input[i] = input_q8[i];
}
__syncthreads();
// All warps now read from shared_input instead of global
```

Expected improvement: 10-20% for Q6_K/Q8_0 decode GEMV (eliminates L1/L2 thrashing).

### 0.3 Register-tiled Q6_K Dequant

Current Q6_K GEMV: each warp processes one row, dequantizes inline. The dequant arithmetic (bit-shifting, scale multiply) is interleaved with accumulation, causing instruction pipeline stalls.

Optimization: Separate dequant into registers first (full Q6_K superblock -> 256 FP16 values in registers), then dot product against cached input. Hides dequant latency behind memory loads.

**Files**: `src/compute/gemm.cu` (Q6_K GEMV kernels), `src/compute/gemm_q6k.cu`

## Phase 1: Persistent Decode Kernel (Eliminates Launch Overhead)

**Why**: A 30-layer decode iteration launches ~300 kernels. Even with CUDA Graphs + PDL, each kernel has grid launch + teardown overhead. A persistent kernel stays resident and processes the entire decode forward pass as a work queue.

**Files**: `src/compute/persistent_decode.cu` (new), `src/graph/executor.cu`, `src/runtime/engine.cpp`

### Design

```
┌─────────────────────────────────────────────────────┐
│              Persistent Decode Kernel                │
│                                                      │
│  Grid: N blocks (= number of SMs on RTX 5090)       │
│  Each block: 256 threads                             │
│                                                      │
│  while (true) {                                      │
│    op = work_queue[op_idx]  // device-side queue      │
│    switch (op.type) {                                │
│      case RMSNORM:   rmsnorm_device(op.args);        │
│      case GEMV_Q6K:  gemv_q6k_device(op.args);       │
│      case ROPE:      rope_device(op.args);           │
│      case KV_WRITE:  write_kv_device(op.args);       │
│      case PAGED_ATN: paged_attn_device(op.args);     │
│      case SWIGLU:    swiglu_device(op.args);         │
│      case SAMPLE:    argmax_device(op.args);         │
│      case EXIT:      return;                         │
│    }                                                 │
│    grid.sync();  // cooperative groups barrier        │
│    op_idx++;                                         │
│  }                                                   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Grid = all SMs**: Use cooperative launch (`cudaLaunchCooperativeKernel`) so `grid.sync()` works
2. **Work queue in device memory**: Pre-built by host before launch, contains op descriptors with pointers
3. **GEMV distribution**: For a GEMV [M, K], blocks split M rows across the grid (block i handles rows `[i*chunk, (i+1)*chunk)`)
4. **Attention**: Blocks split across KV-heads (same as current paged attention)
5. **RMSNorm**: Blocks split rows, grid.sync() after reduction
6. **Memory reuse**: Same shared workspace buffers as current executor, just accessed from persistent context

### Integration with CUDA Graphs

The persistent kernel IS the graph — a single kernel node replaces 300+ nodes. Graph capture becomes trivial: capture one `cudaLaunchCooperativeKernel`, replay it with updated work queue pointers.

### Expected Impact

- Eliminates ~300 kernel launches per decode step (~15-30 us saved)
- Eliminates inter-kernel synchronization overhead
- For small models (Phi-4-Mini): ~5-8% decode improvement (launch overhead is significant fraction)
- For large models: ~2-4% (GEMV time dominates)

### Tradeoffs

- Cooperative launch requires all SMs → incompatible with Green Contexts (can't partition SMs)
- Occupancy limited to 1 block/SM (all blocks must be resident)
- Complex device-side dispatch logic
- Debugging is harder (one mega-kernel vs discrete kernels)
- Fallback: keep current discrete kernel path for debug/prefill/Green Context modes

## Phase 2: TMA + wgmma Attention (Prefill Performance)

**Why**: Current Blackwell attention uses WMMA 16x16x16, which underutilizes the tensor core pipeline. wgmma provides 64x64x16 or larger tiles with 4x higher throughput per instruction. TMA provides hardware-managed tiled memory access, freeing warps from load duties.

**Files**: `src/compute/attention_blackwell.cu` (modify or add new path), `src/compute/attention_dispatch.cu`

### 2.1 TMA for Q/K/V Tile Loading

Replace manual per-thread global loads with TMA bulk copy:

```cpp
// Host-side: create TMA descriptors for Q, K, V tensors
CUtensorMap tma_desc_Q, tma_desc_K, tma_desc_V;
// Each describes a 2D tile: [tile_rows, head_dim] within [seq_len, n_heads * head_dim]

// Device-side: async TMA load into shared memory
__shared__ __align__(128) half Q_smem[Br][HD];
__shared__ __align__(128) half K_smem[Bc][HD];

// Only one thread issues TMA
if (threadIdx.x == 0) {
    uint64_t* mbar = &barriers[stage];
    mbarrier_arrive_expect_tx(mbar, Br * HD * sizeof(half));
    cp_async_bulk_tensor_2d_global_to_shared(
        &Q_smem[0][0], &tma_desc_Q, tile_row, head_offset, mbar);
}
mbarrier_wait(mbar, phase);  // all threads wait
```

**Benefit**: Frees all 256 threads from load work → they can compute while TMA fills next tile. Current kernel wastes ~30% of thread cycles on global loads.

### 2.2 wgmma for S = Q @ K^T and O += P @ V

Replace WMMA 16x16x16 with wgmma (warpgroup MMA):

```cpp
// wgmma: 4 warps (128 threads) = 1 warpgroup
// Instruction: wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
// Input: A in registers (from shared via ldmatrix), B in shared memory
// Output: 64x64 accumulator in registers across warpgroup

// Phase 1: S = Q @ K^T [Br=64, Bc=64, HD=128]
// 8 wgmma iterations (HD/16 = 8)
for (int k = 0; k < hd_chunks; k++) {
    wgmma::load_a(a_desc, Q_smem, tile_row, k * 16);
    wgmma::mma_async(acc, a_desc, K_smem_desc + k * 16, acc);
}
wgmma::wait();  // fence
```

**Benefit**: wgmma processes 64x64x16 per instruction (vs 16x16x16 for WMMA) — 16x more output elements. 2 warpgroups (8 warps) can process 2 output tiles concurrently.

### 2.3 Multi-stage Pipeline with mbarrier

3-stage pipeline: while computing S for current K-tile, load next K-tile via TMA, and convert P->half for previous tile.

```
Stage 0: [COMPUTE S_j] [LOAD K_{j+1}] [CONVERT P_{j-1}]
Stage 1: [COMPUTE S_{j+1}] [LOAD K_{j+2}] [COMPUTE O += P_j @ V_j]
Stage 2: ...
```

Each stage uses separate mbarrier for synchronization. TMA loads are fully async and don't consume any warp cycles.

### Shared Memory Budget (99 KB limit)

```
Q_smem [64 × 128] half    = 16 KB
K_smem[2] [64 × 128] half = 32 KB (double-buffered)
V_smem [64 × 128] half    = 16 KB
S_smem [64 × 64] float    = 16 KB
P_smem [64 × 64] half     =  8 KB
mbarrier [6 × 8B]          =  48 B
──────────────────────────────
Total                      = 88 KB ← fits in 99 KB
```

Leaves 11 KB headroom for scale factors, softmax state, etc.

### Expected Impact

- Prefill: 30-50% speedup for long sequences (512+ tokens) where attention dominates
- Decode paged attention: Modest improvement (bandwidth-bound, not compute-bound)
- pp tok/s gap vs llama.cpp should close or reverse

### Tradeoffs

- wgmma requires sm_90+ (Hopper/Blackwell) — keep WMMA fallback for older GPUs
- TMA descriptor setup is host-side overhead (amortized across layers)
- wgmma register pressure is high — may reduce occupancy
- Complexity: wgmma uses PTX inline assembly (no C++ WMMA wrapper)

## Phase 3: Thread Block Clusters + Distributed Shared Memory (Attention)

**Why**: For GQA with n_q_per_kv=8, 8 thread blocks each need the same K/V data. With clusters, these blocks form a cluster and share K/V via distributed shared memory (DSMEM), loading from global memory only once.

**Files**: `src/compute/attention_paged.cu` (decode), `src/compute/attention_blackwell.cu` (prefill)

### Design

```
Cluster of 4 blocks (4 Q-heads sharing 1 KV-head):
  Block 0: loads K-tile into its shared memory
  Block 1: loads V-tile into its shared memory
  Block 2-3: compute-only (no global loads)

  All 4 blocks access each other's shared memory via DSMEM:
    K_data = cluster.map_shared_rank(K_smem, rank=0)  // block 0's smem
    V_data = cluster.map_shared_rank(V_smem, rank=1)  // block 1's smem
```

```cpp
// Launch with cluster attribute
cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {4, 1, 1};  // 4 blocks per cluster

cudaLaunchConfig_t config = {};
config.gridDim = {n_batch * n_kv_heads, 1, 1};
config.blockDim = {256, 1, 1};
config.attrs = attrs;
config.numAttrs = 1;
cudaLaunchKernelEx(&config, kernel, args...);
```

### Expected Impact

- 2-4x reduction in K/V global memory reads for GQA models (ratio 4-8)
- Biggest win on bandwidth-constrained RTX 5090 (GDDR7)
- Decode paged attention: 10-20% speedup for GQA models with long context

### Tradeoffs

- Cluster size limited by SM count and occupancy
- RTX 5090 has ~170 SMs — cluster of 8 uses 8 SMs per KV-head
- Only beneficial for GQA (not MHA where ratio=1)
- Requires careful smem layout alignment (128-byte for DSMEM access)

## Phase 4: NVRTC JIT Backend

**Why**: Static compilation means one binary for all model shapes. JIT allows compile-time constants for head_dim, n_kv_heads, d_model — enabling the compiler to unroll loops, eliminate branches, and optimize register allocation per-model.

**Files**: `src/runtime/jit.cu` (new), `src/runtime/jit.h` (new), `src/compute/gemm.cu` (template extraction)

### Design

```cpp
struct JITKey {
    int hidden_size, head_dim, n_heads, n_kv_heads;
    int quant_type;  // Q6_K, Q8_0, Q4_0
    int sm_version;  // 90, 100, 120

    bool operator==(const JITKey& o) const { /* field-wise */ }
};

class JITCache {
    std::unordered_map<JITKey, CUmodule> cache_;
    std::mutex mu_;

public:
    CUfunction get_or_compile(const JITKey& key, const char* kernel_name) {
        auto it = cache_.find(key);
        if (it != cache_.end()) return get_func(it->second, kernel_name);

        // Generate CUDA source with compile-time constants
        std::string src = generate_kernel_source(key);

        // NVRTC compile
        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, src.c_str(), "jit_kernel.cu", 0, nullptr, nullptr);
        const char* opts[] = {
            "--gpu-architecture=sm_120",
            "-O3", "--use_fast_math",
            "-std=c++20"
        };
        nvrtcCompileProgram(prog, 4, opts);

        // Get PTX, load module
        size_t ptx_size;
        nvrtcGetPTXSize(prog, &ptx_size);
        std::vector<char> ptx(ptx_size);
        nvrtcGetPTX(prog, ptx.data());
        nvrtcDestroyProgram(&prog);

        CUmodule mod;
        cuModuleLoadData(&mod, ptx.data());
        cache_[key] = mod;
        return get_func(mod, kernel_name);
    }
};
```

### Kernels to JIT-Specialize

1. **GEMV**: Unroll K-dimension loop when K is known (e.g., K=3072 for Phi-4-Mini)
2. **RMSNorm**: d_model as constexpr → perfect vectorization
3. **Paged Attention**: head_dim as constexpr → unrolled dot products
4. **SwiGLU/Activation**: d_ff as constexpr → vectorized without remainder handling

### Kernel Cache Strategy

- Cache key: `(hidden_size, head_dim, n_heads, n_kv_heads, quant_type, sm_version)`
- Disk cache: Save compiled PTX to `~/.cache/imp/jit/` for instant reload
- Lazy compilation: JIT on first use, ~200ms per kernel (amortized over generation)
- Fallback: Always keep static kernels as fallback

### Expected Impact

- 5-10% decode improvement from perfect loop unrolling and register allocation
- Eliminates all remainder handling / branch overhead
- Biggest win on smaller models where overhead fraction is highest

### Tradeoffs

- First-run compilation latency (~1-2s for full kernel set)
- Binary cache management (disk space, invalidation)
- Debugging JIT kernels requires PTX-level analysis
- Adds NVRTC as build dependency

## Phase 5: Persistent Token Loop (Multi-Token Decode)

**Why**: Extend the Phase 1 persistent kernel to run the entire autoregressive loop on-device, eliminating host-GPU synchronization per token entirely.

This is an evolution of the existing `CudaGraphConditionalRunner` but implemented as a true persistent kernel instead of a CUDA graph with conditional nodes.

**Files**: `src/compute/persistent_decode.cu`, `src/runtime/engine.cpp`

### Design

```cpp
// Host prepares work plan for up to max_tokens decode steps
struct PersistentDecodePlan {
    int max_steps;
    int eos_token_id;
    int vocab_size;
    // Per-layer op descriptors (pre-built)
    OpDescriptor ops[MAX_LAYERS * OPS_PER_LAYER];
    // Ring buffer for output tokens (mapped pinned memory)
    int32_t* output_ring;  // host-visible via cudaHostAllocMapped
    volatile int32_t* steps_completed;  // atomic counter
};

// Device-side: entire generation loop
__global__ void persistent_token_loop(PersistentDecodePlan plan) {
    for (int step = 0; step < plan.max_steps; step++) {
        // Execute all layer ops (same as Phase 1)
        for (int op = 0; op < plan.n_ops; op++) {
            execute_op(plan.ops[op]);
            grid.sync();
        }

        // Argmax sampling (single block does it)
        if (blockIdx.x == 0) {
            int token = device_argmax(logits, plan.vocab_size);
            plan.output_ring[step] = token;
            atomicAdd((int*)plan.steps_completed, 1);
            // Update token_ids for next step
            plan.current_token = token;
            if (token == plan.eos_token_id) {
                plan.early_exit = true;
            }
        }
        grid.sync();
        if (plan.early_exit) return;

        // Update KV cache position (device-side)
        plan.position++;
    }
}
```

### Host-Side Token Delivery

Host polls `steps_completed` (mapped pinned memory) and reads tokens from ring buffer. Same pattern as existing `CudaGraphConditionalRunner` but with the WSL2 sync-then-deliver workaround (cudaStreamSynchronize before reading mapped memory).

### Expected Impact

- Eliminates ALL host-GPU round trips during decode
- ~1-3% latency improvement (host-side scheduling overhead removed)
- Simplifies engine step() logic for the common single-sequence case

## Implementation Priority & Sequencing

```
Phase 0  ──→  Phase 1  ──→  Phase 2  ──→  Phase 3  ──→  Phase 4
  │              │              │              │              │
  │              │              │              │              │
  ▼              ▼              ▼              ▼              ▼
GEMV BW       Persistent     TMA+wgmma     Clusters       JIT
optimization  decode kernel  attention     + DSMEM        backend
                                           (attention)
  ~2 days       ~3 days       ~5 days       ~3 days       ~4 days
```

**Phase 0** is prerequisite for Phase 1 (optimized GEMV device functions reused in persistent kernel).
**Phase 2** and **Phase 3** can be developed in parallel after Phase 1.
**Phase 4** is independent and can start anytime.

## Architecture Diagram

```
                    ┌──────────────────────────────────┐
                    │           imp Engine              │
                    │                                    │
                    │  ┌────────────┐  ┌─────────────┐  │
                    │  │ Scheduler  │  │ KV Cache Mgr│  │
                    │  │ (cont.     │  │ (paged,     │  │
                    │  │  batch)    │  │  LRU evict) │  │
                    │  └─────┬──────┘  └──────┬──────┘  │
                    │        │                │          │
                    │        ▼                ▼          │
                    │  ┌─────────────────────────────┐  │
                    │  │      Graph Executor          │  │
                    │  │                               │  │
                    │  │  Prefill: cuBLAS GEMM         │  │
                    │  │    + TMA/wgmma attention ←NEW │  │
                    │  │                               │  │
                    │  │  Decode: 2 paths               │  │
                    │  │    Path A: Persistent kernel   │  │
                    │  │      (cooperative launch,     │  │
                    │  │       device-side dispatch,   │  │
                    │  │       grid.sync barriers)     │  │
                    │  │    Path B: Discrete kernels    │  │
                    │  │      (CUDA Graph + PDL,       │  │
                    │  │       Green Contexts)         │  │
                    │  └──────────┬──────────────────┘  │
                    │             │                      │
                    │             ▼                      │
                    │  ┌──────────────────────────────┐  │
                    │  │     Kernel Library            │  │
                    │  │                               │  │
                    │  │  GEMV: multi-row, smem input  │  │
                    │  │    Q6K, Q8_0, Q4_0 + dp4a    │  │
                    │  │  Attention: TMA+wgmma (pf)    │  │
                    │  │    Cluster+DSMEM (decode)     │  │
                    │  │  RMSNorm, RoPE, SwiGLU, etc. │  │
                    │  │                               │  │
                    │  │  ┌──────────┐                 │  │
                    │  │  │ JIT Cache│ (NVRTC, PTX)    │  │
                    │  │  │ per-model│                 │  │
                    │  │  └──────────┘                 │  │
                    │  └──────────────────────────────┘  │
                    └──────────────────────────────────┘
```

## Memory Layout (Decode, Single Token)

```
GPU VRAM (32 GB RTX 5090):
┌──────────────────────────────────────────────────────┐
│ Model Weights (quantized)                    ~8-16 GB│
│   Q6_K: 6.5 bits/param, Q8_0: 8.5 bits/param       │
├──────────────────────────────────────────────────────┤
│ KV Cache (paged, 16 tokens/block)            ~2-12 GB│
│   Layout: [block_id, slot, kv_head, head_dim]        │
│   FP16: 2B per element                               │
├──────────────────────────────────────────────────────┤
│ Workspace (persistent)                       ~50-200 MB│
│   hidden[1, d_model], residual[1, d_model]           │
│   q[1, n_heads*hd], k[1, nkv*hd], v[1, nkv*hd]     │
│   attn_out[1, n_heads*hd], proj_out[1, d_model]     │
│   logits[1, vocab_size] (FP32)                       │
│   q8_1 scratch, dequant scratch                      │
├──────────────────────────────────────────────────────┤
│ cuBLAS workspace                             64 MB   │
├──────────────────────────────────────────────────────┤
│ CUDA Graph + Persistent Kernel context       ~10 MB  │
└──────────────────────────────────────────────────────┘
```

## Verification Plan

### Phase 0 (GEMV Optimization)
1. Build: `cmake --build build -j$(nproc)`
2. Run existing tests: `./build/imp-tests --gtest_filter="*Gemm*:*GEMV*"`
3. Benchmark decode against baseline: `./build/imp-bench` on Phi-4-Mini Q8_0
4. Compare tg tok/s before/after — target: >220 tok/s (up from 208)
5. Verify correctness: `IMP_DEBUG_FORWARD=1 ./build/imp-cli --model phi-4-mini.gguf --prompt "test" --max-tokens 10`

### Phase 1 (Persistent Kernel)
1. New test: `tests/test_persistent_decode.cu` — verify single-layer persistent dispatch matches discrete kernel output
2. End-to-end: compare token output of persistent vs discrete path for 128 tokens
3. Benchmark: decode tok/s improvement, especially on small models (Phi-4-Mini)
4. Fallback: verify discrete path still works when persistent is disabled

### Phase 2 (TMA + wgmma Attention)
1. Run: `./build/imp-tests --gtest_filter="*Attention*"`
2. Verify prefill output matches baseline (compare logits for first token)
3. Benchmark: `./build/imp-bench` pp tok/s on 512-token prompt
4. Verify on multiple head_dims: 64, 96, 128

### Phase 3 (Clusters + DSMEM)
1. Run: `./build/imp-tests --gtest_filter="*PagedAttention*"`
2. Verify decode output matches non-cluster path
3. Benchmark: decode with long context (1K+ tokens cached)

### Phase 4 (JIT)
1. Verify JIT kernels produce identical output to static kernels
2. Measure compilation time (should be <2s for full kernel set)
3. Verify disk cache works (second load should be instant)
4. Benchmark: decode tok/s with JIT-specialized kernels

## Expected Cumulative Speedup

| Phase | Dense Decode Improvement | Dense Prefill Improvement |
|-------|------------------------|--------------------------|
| Phase 0 (GEMV BW) | +15-25% | — |
| Phase 1 (Persistent) | +5-8% | — |
| Phase 2 (TMA+wgmma) | — | +30-50% |
| Phase 3 (Clusters) | +10-20% (GQA) | +5-10% |
| Phase 4 (JIT) | +5-10% | +5-10% |
| **Cumulative** | **+35-63%** | **+40-70%** |

Target: surpass llama.cpp on both prefill and decode for all model sizes.
