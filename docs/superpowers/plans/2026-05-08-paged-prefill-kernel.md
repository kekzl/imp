# Paged Prefill — Chunked Attention Correctness Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make chunked prefill produce correct logits by reading past-chunk K/V from the paged cache, then re-enable `prefill_chunk_size=512` default for full-attention models with FP16/FP8 KV.

**Architecture:** Gather past chunks' K/V from paged cache into a contiguous flat buffer (FP8 dequant on-the-fly), concat current chunk's K/V, dispatch a generalized rectangular `attention_cublas_prefill` with `q_offset` for offset-aware causal masking. Per-arch `resolve_prefill_chunk_size()` keeps Gemma-4/hybrid/sub-byte-KV at single-chunk.

**Tech Stack:** C++20, CUDA 13.2 (sm_120a), cuBLAS, GTest, CMake/FetchContent, Docker (`make build`/`make test-gpu`).

**Reference spec:** `docs/superpowers/specs/2026-05-08-paged-prefill-kernel-design.md`

**Build & test commands:**
- `make build` — Docker image (use this if container is fresh)
- `make test-gpu` — full GTest suite (~30s)
- `make test-unit` — CPU-only filtered tests (~5s)
- `make verify-fast` — build + filtered tests + perf gate + smoke prompt (~90s)
- `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)` — host build alternative
- After host build, individual test binaries are at `build/tests/test-*`

---

## Task 1: Add `prefill_offset` to `InferenceState` and engine plumbing

**Files:**
- Modify: `src/graph/executor.h:137` (struct `InferenceState`)
- Modify: `src/runtime/engine.cpp:1838` (in `step_prefill_one`)

**Goal:** Add the field that downstream code will read; set it in the engine but don't yet branch on it.

- [ ] **Step 1: Read current `InferenceState::is_prefill` location**

```bash
sed -n '135,140p' src/graph/executor.h
```

Expected output around line 137:
```
    // Mode
    bool is_prefill = true;
```

- [ ] **Step 2: Add `prefill_offset` field**

In `src/graph/executor.h`, change:
```cpp
    // Mode
    bool is_prefill = true;
```
to:
```cpp
    // Mode
    bool is_prefill = true;
    // Absolute position of state.positions[0] within the full sequence.
    // 0 means single-chunk prefill or first chunk of a chunked prefill.
    // > 0 means a follow-up chunk: tokens [0, prefill_offset) are already in the KV cache.
    int prefill_offset = 0;
```

- [ ] **Step 3: Set the field in `step_prefill_one`**

In `src/runtime/engine.cpp`, find the line `state.is_prefill = true;` (around line 1838) and add immediately after:

```cpp
    state.prefill_offset = offset;  // absolute pos of state.positions[0]
```

- [ ] **Step 4: Build host-side**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -5`
Expected: clean build, no errors.

- [ ] **Step 5: Run full test suite to confirm no regression**

Run: `make test-gpu 2>&1 | tail -10`
Expected: same totals as before (769 ran, 747 passed, 22 skipped, 0 failed). The new field is unread, so behavior is byte-identical.

- [ ] **Step 6: Commit**

```bash
git add src/graph/executor.h src/runtime/engine.cpp
git commit -m "$(cat <<'EOF'
feat(state): add prefill_offset to InferenceState

Plumbing for chunked-prefill correctness fix (L2 roadmap). Engine sets
prefill_offset = offset in step_prefill_one. No reader yet — behavior
is byte-identical until run_attention dispatch lands in a later commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `paged_kv_gather` kernels with unit tests (FP16 + FP8→FP16)

**Files:**
- Create: `src/compute/kv_gather.h`
- Create: `src/compute/kv_gather.cu`
- Modify: `CMakeLists.txt` (add `src/compute/kv_gather.cu` to library sources)
- Create: `tests/test_kv_gather.cu`
- Modify: `tests/CMakeLists.txt` (register test)

**Goal:** Two device kernels that read past-chunk K/V from the paged cache (`[num_blocks, block_size, nkv, hd]` slot-first layout) into a contiguous flat buffer.

- [ ] **Step 1: Verify cache layout assumption**

Run: `grep -n "kv_block_stride\|kv_slot_stride" src/compute/attention_paged.cu | head -5`
Expected:
```
const int kv_block_stride = block_size * n_kv_heads * head_dim;
const int kv_slot_stride = n_kv_heads * head_dim;
```
Confirms slot-first (`[block, slot, kv_head, hd]`) layout. The gather kernels mirror this indexing.

- [ ] **Step 2: Write the header `src/compute/kv_gather.h`**

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace imp {

// Gather contiguous K (or V) of past tokens [0, n_past) from the paged FP16
// cache into a flat buffer.
//
// dst:         FP16 [n_past, nkv, hd] contiguous (caller-allocated)
// src:         FP16 paged base pointer (KVCache::k_ptr / v_ptr)
//              Layout: [num_blocks, block_size, nkv, hd]
// block_table: device pointer, [ceil(n_past/block_size)] int32
// n_past:      number of tokens to gather (positions 0..n_past-1)
// block_size:  KV cache block_size (e.g. 16)
// nkv:         number of KV heads
// hd:          head_dim
void paged_kv_gather_fp16(half* dst, const half* src, const int* block_table,
                          int n_past, int block_size, int nkv, int hd, cudaStream_t stream);

// FP8 E4M3 paged → FP16 flat with per-tensor scalar dequant: dst[i] = (half)((float)src[i] * kv_scale).
// Same indexing as paged_kv_gather_fp16; src is FP8 E4M3 (1 byte / elem).
void paged_kv_gather_fp8_to_fp16(half* dst, const __nv_fp8_e4m3* src, const int* block_table,
                                 float kv_scale, int n_past, int block_size, int nkv, int hd,
                                 cudaStream_t stream);

}  // namespace imp
```

- [ ] **Step 3: Write the kernels in `src/compute/kv_gather.cu`**

```cpp
#include "compute/kv_gather.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace imp {

// Each thread handles one (token, kv_head, head_dim_elem) triple.
// Grid: (ceil(n_past / TOKENS_PER_BLOCK), nkv).
// Block: TOKENS_PER_BLOCK * (hd / VEC) threads (tunable; here 256 threads).
//
// We use a flat 1D thread index inside the block over (token_in_block, hd_elem)
// to keep the kernel simple and let the compiler vectorize the half loads.
//
// NOTE: __ldcs is a streaming hint — KV bytes don't pollute L2, matching
// paged_attention_decode_fp8 / decode_int8 / decode behavior.

static constexpr int TOKENS_PER_BLOCK = 8;

__global__ void paged_kv_gather_fp16_kernel(half* __restrict__ dst, const half* __restrict__ src,
                                            const int* __restrict__ block_table, int n_past,
                                            int block_size, int nkv, int hd) {
    const int block_group = blockIdx.x;     // group of TOKENS_PER_BLOCK tokens
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int threads_per_token = blockDim.x / TOKENS_PER_BLOCK;
    const int token_in_block = tid / threads_per_token;
    const int d_lane = tid % threads_per_token;

    const int pos = block_group * TOKENS_PER_BLOCK + token_in_block;
    if (pos >= n_past)
        return;

    const int blk_idx = pos / block_size;
    const int slot = pos % block_size;
    const int phys_block = block_table[blk_idx];

    const int kv_block_stride = block_size * nkv * hd;
    const int kv_slot_stride = nkv * hd;

    const half* src_row = src + (size_t)phys_block * kv_block_stride
                              + (size_t)slot * kv_slot_stride
                              + (size_t)kv_head * hd;
    half* dst_row = dst + (size_t)pos * nkv * hd + (size_t)kv_head * hd;

    for (int d = d_lane; d < hd; d += threads_per_token) {
        // Streaming load (skip L1, evict-first from L2) so KV bytes don't pollute
        // L2 for the FFN GEMM that follows. Same hint as paged_attention_decode.
        unsigned short raw = __ldcs(reinterpret_cast<const unsigned short*>(src_row + d));
        dst_row[d] = __ushort_as_half(raw);
    }
}

__global__ void paged_kv_gather_fp8_to_fp16_kernel(half* __restrict__ dst,
                                                    const __nv_fp8_e4m3* __restrict__ src,
                                                    const int* __restrict__ block_table,
                                                    float kv_scale, int n_past, int block_size,
                                                    int nkv, int hd) {
    const int block_group = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int threads_per_token = blockDim.x / TOKENS_PER_BLOCK;
    const int token_in_block = tid / threads_per_token;
    const int d_lane = tid % threads_per_token;

    const int pos = block_group * TOKENS_PER_BLOCK + token_in_block;
    if (pos >= n_past)
        return;

    const int blk_idx = pos / block_size;
    const int slot = pos % block_size;
    const int phys_block = block_table[blk_idx];

    const int kv_block_stride = block_size * nkv * hd;
    const int kv_slot_stride = nkv * hd;

    const __nv_fp8_e4m3* src_row = src + (size_t)phys_block * kv_block_stride
                                       + (size_t)slot * kv_slot_stride
                                       + (size_t)kv_head * hd;
    half* dst_row = dst + (size_t)pos * nkv * hd + (size_t)kv_head * hd;

    for (int d = d_lane; d < hd; d += threads_per_token) {
        // Streaming load via __ldcs on uint8 (FP8 is 1 byte).
        unsigned char raw = __ldcs(reinterpret_cast<const unsigned char*>(src_row + d));
        __nv_fp8_e4m3 fp8;
        memcpy(&fp8, &raw, 1);
        float f = static_cast<float>(fp8);
        dst_row[d] = __float2half(f * kv_scale);
    }
}

void paged_kv_gather_fp16(half* dst, const half* src, const int* block_table, int n_past,
                          int block_size, int nkv, int hd, cudaStream_t stream) {
    if (n_past <= 0 || nkv <= 0 || hd <= 0)
        return;
    int n_block_groups = (n_past + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    dim3 grid(n_block_groups, nkv);
    int threads = 256;  // 8 tokens × 32 lanes; works for hd up to 256 with stride-32, OK for hd=512 too
    paged_kv_gather_fp16_kernel<<<grid, threads, 0, stream>>>(dst, src, block_table, n_past,
                                                              block_size, nkv, hd);
}

void paged_kv_gather_fp8_to_fp16(half* dst, const __nv_fp8_e4m3* src, const int* block_table,
                                 float kv_scale, int n_past, int block_size, int nkv, int hd,
                                 cudaStream_t stream) {
    if (n_past <= 0 || nkv <= 0 || hd <= 0)
        return;
    int n_block_groups = (n_past + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    dim3 grid(n_block_groups, nkv);
    int threads = 256;
    paged_kv_gather_fp8_to_fp16_kernel<<<grid, threads, 0, stream>>>(dst, src, block_table, kv_scale,
                                                                      n_past, block_size, nkv, hd);
}

}  // namespace imp
```

- [ ] **Step 4: Register `kv_gather.cu` in CMake**

In `CMakeLists.txt`, find the list of compute sources (look for `src/compute/attention_paged.cu` near where library targets are defined):

Run: `grep -n "src/compute/attention_paged.cu" CMakeLists.txt`

Add `src/compute/kv_gather.cu` to the same list (in alphabetical order if the file uses one).

- [ ] **Step 5: Write the failing test `tests/test_kv_gather.cu`**

```cpp
#include "compute/kv_gather.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <vector>
#include <random>

namespace imp {

// Build a synthetic paged FP16 cache with deterministic values:
//   src[block, slot, kv_head, d] = float(block * 1000 + slot * 10 + kv_head + d * 0.01)
// Then verify gather to flat layout reads back via block_table.
TEST(KVGatherTest, FP16_PagedToFlat_RoundTrip) {
    const int num_blocks = 8;
    const int block_size = 16;
    const int nkv = 4;
    const int hd = 64;
    const int n_past = 100;  // 100 tokens → 7 full blocks + partial

    // Permuted block_table: maps logical block_idx → physical block.
    std::vector<int> h_bt = {3, 1, 0, 5, 2, 7, 4, 6};

    size_t total_elems = (size_t)num_blocks * block_size * nkv * hd;
    std::vector<half> h_src(total_elems);
    for (int b = 0; b < num_blocks; b++) {
        for (int s = 0; s < block_size; s++) {
            for (int k = 0; k < nkv; k++) {
                for (int d = 0; d < hd; d++) {
                    float v = (float)b + 0.001f * (float)s + 0.0001f * (float)k + 0.00001f * (float)d;
                    size_t idx = ((size_t)b * block_size + s) * nkv * hd + (size_t)k * hd + d;
                    h_src[idx] = __float2half(v);
                }
            }
        }
    }

    half* d_src;
    int* d_bt;
    half* d_dst;
    cudaMalloc(&d_src, total_elems * sizeof(half));
    cudaMalloc(&d_bt, h_bt.size() * sizeof(int));
    cudaMalloc(&d_dst, (size_t)n_past * nkv * hd * sizeof(half));
    cudaMemcpy(d_src, h_src.data(), total_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice);

    paged_kv_gather_fp16(d_dst, d_src, d_bt, n_past, block_size, nkv, hd, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_dst((size_t)n_past * nkv * hd);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Verify: dst[pos, kv_head, d] should equal src[block_table[pos/bs], pos%bs, kv_head, d].
    for (int pos = 0; pos < n_past; pos++) {
        int phys_block = h_bt[pos / block_size];
        int slot = pos % block_size;
        for (int k = 0; k < nkv; k++) {
            for (int d = 0; d < hd; d++) {
                size_t src_idx = ((size_t)phys_block * block_size + slot) * nkv * hd + (size_t)k * hd + d;
                size_t dst_idx = (size_t)pos * nkv * hd + (size_t)k * hd + d;
                ASSERT_EQ(__half_as_ushort(h_dst[dst_idx]), __half_as_ushort(h_src[src_idx]))
                    << "pos=" << pos << " k=" << k << " d=" << d;
            }
        }
    }

    cudaFree(d_src);
    cudaFree(d_bt);
    cudaFree(d_dst);
}

TEST(KVGatherTest, FP16_PartialLastBlock) {
    // n_past = block_size + 1 → last block has exactly 1 valid slot.
    const int num_blocks = 4;
    const int block_size = 16;
    const int nkv = 2;
    const int hd = 32;
    const int n_past = 17;

    std::vector<int> h_bt = {2, 0};  // need 2 blocks for 17 tokens
    size_t total_elems = (size_t)num_blocks * block_size * nkv * hd;
    std::vector<half> h_src(total_elems, __float2half(0.f));
    // Mark slot 0 of physical block 0 with a sentinel
    for (int k = 0; k < nkv; k++)
        for (int d = 0; d < hd; d++)
            h_src[((size_t)0 * block_size + 0) * nkv * hd + (size_t)k * hd + d] = __float2half(42.0f);

    half* d_src; int* d_bt; half* d_dst;
    cudaMalloc(&d_src, total_elems * sizeof(half));
    cudaMalloc(&d_bt, h_bt.size() * sizeof(int));
    cudaMalloc(&d_dst, (size_t)n_past * nkv * hd * sizeof(half));
    cudaMemcpy(d_src, h_src.data(), total_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice);

    paged_kv_gather_fp16(d_dst, d_src, d_bt, n_past, block_size, nkv, hd, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_dst((size_t)n_past * nkv * hd);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Token 16 (the partial-last-block token) should be the sentinel 42.0.
    for (int k = 0; k < nkv; k++) {
        for (int d = 0; d < hd; d++) {
            size_t dst_idx = (size_t)16 * nkv * hd + (size_t)k * hd + d;
            EXPECT_NEAR(__half2float(h_dst[dst_idx]), 42.0f, 0.001f);
        }
    }

    cudaFree(d_src); cudaFree(d_bt); cudaFree(d_dst);
}

TEST(KVGatherTest, FP8_PagedToFlat_DequantMatchesReference) {
    const int num_blocks = 4;
    const int block_size = 16;
    const int nkv = 2;
    const int hd = 32;
    const int n_past = 32;
    const float kv_scale = 0.25f;

    std::vector<int> h_bt = {1, 3};
    size_t total_elems = (size_t)num_blocks * block_size * nkv * hd;

    // Synthesize FP8 values via float→fp8 conversion.
    std::vector<__nv_fp8_e4m3> h_src(total_elems);
    for (size_t i = 0; i < total_elems; i++) {
        float v = (float)((i % 17) - 8);  // small range, representable in FP8
        h_src[i] = __nv_fp8_e4m3(v);
    }

    __nv_fp8_e4m3* d_src; int* d_bt; half* d_dst;
    cudaMalloc(&d_src, total_elems * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_bt, h_bt.size() * sizeof(int));
    cudaMalloc(&d_dst, (size_t)n_past * nkv * hd * sizeof(half));
    cudaMemcpy(d_src, h_src.data(), total_elems * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice);

    paged_kv_gather_fp8_to_fp16(d_dst, d_src, d_bt, kv_scale, n_past, block_size, nkv, hd, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_dst((size_t)n_past * nkv * hd);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(half), cudaMemcpyDeviceToHost);

    for (int pos = 0; pos < n_past; pos++) {
        int phys_block = h_bt[pos / block_size];
        int slot = pos % block_size;
        for (int k = 0; k < nkv; k++) {
            for (int d = 0; d < hd; d++) {
                size_t src_idx = ((size_t)phys_block * block_size + slot) * nkv * hd + (size_t)k * hd + d;
                size_t dst_idx = (size_t)pos * nkv * hd + (size_t)k * hd + d;
                float expected = static_cast<float>(h_src[src_idx]) * kv_scale;
                EXPECT_NEAR(__half2float(h_dst[dst_idx]), expected, 0.005f);  // FP16 round-off
            }
        }
    }

    cudaFree(d_src); cudaFree(d_bt); cudaFree(d_dst);
}

}  // namespace imp
```

- [ ] **Step 6: Register the test in `tests/CMakeLists.txt`**

Run: `grep -n "test_kv_cache\|test_attention_paged" tests/CMakeLists.txt | head -5`

Add `tests/test_kv_gather.cu` to the appropriate `add_executable` (likely `test-kv` or `test-compute`). If unsure, run `make test-gpu` and pick the binary that contains the most KV/attention-related tests.

- [ ] **Step 7: Build and run the failing test**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -5`
Expected: build succeeds.

Run: `build/tests/test-kv --gtest_filter="KVGatherTest.*"` (or whichever binary you registered into)
Expected: 3 tests PASS. The kernels are deterministic; first run should pass on correct implementation.

If it fails (e.g. due to the Step-3 confused `__ldcs` block on a wrong-typed pointer): simplify the FP16 inner loop to a plain `dst_row[d] = src_row[d];` and re-run. The streaming hint is a perf optimization, not correctness-critical.

- [ ] **Step 8: Commit**

```bash
git add src/compute/kv_gather.h src/compute/kv_gather.cu tests/test_kv_gather.cu CMakeLists.txt tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(compute): paged_kv_gather kernels (FP16 + FP8 to FP16)

Read past chunks K/V from paged cache (slot-first layout) into
contiguous flat buffers. FP8 path dequants on the fly with a
per-tensor scalar (matches paged_attention_decode_fp8 semantics).
3 unit tests cover round-trip equivalence, partial-last-block edge
case, and FP8 dequant accuracy vs CPU reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Generalize `causal_softmax_inplace_kernel` to `(q_len, kv_len, q_offset)`

**Files:**
- Modify: `src/compute/attention_cublas.cu` (kernels at lines 41 and 100)
- Modify: callers within `attention_cublas.cu` (`attention_cublas_prefill` body)

**Goal:** Refactor the two softmax kernels to take rectangular dimensions + offset. Square callers wrap with `q_len=kv_len=seq_len, q_offset=0`. Behavior must stay byte-identical for existing tests.

- [ ] **Step 1: Read current kernel signatures**

Run: `sed -n '41,90p' src/compute/attention_cublas.cu`

Confirm the FP32 kernel takes `(float* S, int seq_len, bool causal)`.

Run: `sed -n '100,170p' src/compute/attention_cublas.cu`

Confirm the FP16 kernel takes `(half* S, int seq_len, bool causal)`.

- [ ] **Step 2: Generalize FP32 kernel**

In `src/compute/attention_cublas.cu`, replace the FP32 softmax kernel signature and body. Find:

```cpp
__global__ void causal_softmax_fp32_inplace_kernel(float* __restrict__ S, int seq_len, bool causal) {
    int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;
    float* row_ptr = S + (static_cast<int64_t>(head) * seq_len + row) * seq_len;

    float max_val = -FLT_MAX;
    for (int j = tid; j < seq_len; j += blockDim.x)
        max_val = fmaxf(max_val, (causal && j > row) ? -FLT_MAX : row_ptr[j]);
```

Replace with:

```cpp
__global__ void causal_softmax_fp32_inplace_kernel(float* __restrict__ S, int q_len, int kv_len,
                                                    int q_offset, bool causal) {
    int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;
    float* row_ptr = S + (static_cast<int64_t>(head) * q_len + row) * kv_len;
    int abs_row = q_offset + row;

    float max_val = -FLT_MAX;
    for (int j = tid; j < kv_len; j += blockDim.x)
        max_val = fmaxf(max_val, (causal && j > abs_row) ? -FLT_MAX : row_ptr[j]);
```

Then update the rest of the kernel body — replace every `seq_len` with `kv_len` and every `j > row` with `j > abs_row`. Specifically the two loops at the bottom:

```cpp
    float sum_val = 0.0f;
    for (int j = tid; j < kv_len; j += blockDim.x)
        sum_val += (causal && j > abs_row) ? 0.0f : expf(row_ptr[j] - max_val);
```

and:

```cpp
    for (int j = tid; j < kv_len; j += blockDim.x)
        row_ptr[j] = (causal && j > abs_row) ? 0.0f : expf(row_ptr[j] - max_val) * inv_sum;
```

- [ ] **Step 3: Generalize FP16 kernel the same way**

In `causal_softmax_inplace_kernel` (FP16 variant, around line 100): identical mechanical changes. Signature becomes `(half* S, int q_len, int kv_len, int q_offset, bool causal)`. Replace `seq_len` → `kv_len` for column iteration / row stride, and add `int abs_row = q_offset + row;` for the mask predicate.

- [ ] **Step 4: Update `attention_cublas_prefill` callers (square use of new signatures)**

In `attention_cublas_prefill` body, find the kernel launches around lines 305 and 314:

```cpp
    causal_softmax_fp32_inplace_kernel<<<grid, threads, 0, stream>>>(S_f32, seq_len, causal);
```

Update to:

```cpp
    causal_softmax_fp32_inplace_kernel<<<grid, threads, 0, stream>>>(S_f32, seq_len, seq_len, /*q_offset=*/0, causal);
```

And the FP16 launch:

```cpp
    causal_softmax_inplace_kernel<<<grid, threads, 0, stream>>>(S_base, seq_len, causal);
```

Update to:

```cpp
    causal_softmax_inplace_kernel<<<grid, threads, 0, stream>>>(S_base, seq_len, seq_len, /*q_offset=*/0, causal);
```

If the GQA batched path also calls these kernels, update those launches too. Run `grep -n "causal_softmax" src/compute/attention_cublas.cu` to see all sites.

- [ ] **Step 5: Build**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -5`
Expected: clean build.

- [ ] **Step 6: Run full test suite — square path must stay byte-equivalent**

Run: `make test-gpu 2>&1 | tail -10`
Expected: same totals as before Task 1 (769 ran, 747 passed, 22 skipped, 0 failed). Critical: any drift here means the refactor broke the square path.

- [ ] **Step 7: Commit**

```bash
git add src/compute/attention_cublas.cu
git commit -m "$(cat <<'EOF'
refactor(attention): generalize causal_softmax_inplace_kernel(s) to rectangular

Signature becomes (S, q_len, kv_len, q_offset, causal). Square path
wraps with q_len=kv_len=seq_len, q_offset=0 — byte-equivalent. Mask
becomes j > q_offset + row, ignored when causal=false. Prereq for
chunked-prefill rectangular attention.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Generalize `attention_cublas_prefill` with `q_offset` parameter

**Files:**
- Modify: `src/compute/attention_cublas.h:26`
- Modify: `src/compute/attention_cublas.cu:231` (function body)
- Modify: callers in `src/graph/executor_attention.cu:747` and `src/graph/executor_attention.cu:842`

**Goal:** Make `attention_cublas_prefill` accept rectangular `(Q[q_len], K[kv_len], V[kv_len])` with `q_offset` for offset-aware causal mask. Square callers stay byte-equivalent (`q_offset=0`).

- [ ] **Step 1: Update function signature in header**

In `src/compute/attention_cublas.h`, find:

```cpp
void attention_cublas_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, Tensor& S,
                              int n_heads, int n_kv_heads, int head_dim, float scale, bool causal,
                              float softcap, cudaStream_t stream);
```

Replace with:

```cpp
// Prefill attention via cuBLAS materialized QK^T + softmax + PV.
//
// Q: [q_len, n_heads * head_dim] FP16
// K: [kv_len, n_kv_heads * head_dim] FP16
// V: [kv_len, n_kv_heads * head_dim] FP16
// O: [q_len, n_heads * head_dim] FP16
// S: workspace, sized for [n_heads * q_len * kv_len] in FP16 or FP32 (FP32 picked when buffer fits).
//
// q_offset is the absolute position of Q[0] in the full sequence. When causal=true,
// Q[i] (abs pos = q_offset + i) is masked against K[j] for j > q_offset + i.
// q_offset=0 reproduces the historic square path exactly.
void attention_cublas_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, Tensor& S,
                              int n_heads, int n_kv_heads, int head_dim, float scale, bool causal,
                              float softcap, int q_offset, cudaStream_t stream);
```

- [ ] **Step 2: Update function body in `attention_cublas.cu`**

In `src/compute/attention_cublas.cu`, find the function definition around line 231 and update both the signature and body. The mechanical changes:

1. Change signature to take `int q_offset` before `cudaStream_t stream`.
2. Read `q_len` and `kv_len`:
```cpp
    int q_len = static_cast<int>(Q.shape[0]);
    int kv_len = static_cast<int>(K.shape[0]);
    if (q_len == 0)
        return;
```
   Replace existing `int seq_len = ...; if (seq_len == 0) return;`.

3. Replace `strideS = seq_len * seq_len` with `strideS = q_len * kv_len`.

4. In QK^T `cublasGemmStridedBatchedEx`: change `M=seq_len, N=seq_len, K=head_dim` to `M=kv_len, N=q_len, K=head_dim`. Specifically the call at around line 284:

```cpp
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, head_dim, &alpha_f,
```

becomes:

```cpp
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, head_dim, &alpha_f,
```

5. In softcap kernel launch, replace `seq_len * seq_len` with `q_len * kv_len`:

```cpp
            int64_t total = static_cast<int64_t>(n_heads) * q_len * kv_len;
```

6. In softmax kernel launches, change `seq_len` to `(q_len, kv_len, q_offset)`:

```cpp
            dim3 grid(q_len, n_heads);
            if (use_fp32_s) {
                causal_softmax_fp32_inplace_kernel<<<grid, threads, 0, stream>>>(
                    S_f32, q_len, kv_len, q_offset, causal);
                // FP32 → FP16 conversion uses q_len*kv_len total
                int64_t total = static_cast<int64_t>(n_heads) * q_len * kv_len;
                ...
            } else {
                causal_softmax_inplace_kernel<<<grid, threads, 0, stream>>>(
                    S_base, q_len, kv_len, q_offset, causal);
            }
```

7. In PV `cublasGemmStridedBatchedEx`: change `M=head_dim, N=seq_len, K=seq_len` to `M=head_dim, N=q_len, K=kv_len`:

```cpp
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_len, seq_len, &one_f,
```

becomes:

```cpp
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, q_len, kv_len, &one_f,
```

8. Update FP32-vs-FP16 S-buffer heuristic:

```cpp
    int64_t s_fp32_elems = static_cast<int64_t>(n_heads) * q_len * kv_len;
```

9. Strides:
- `ld_q = n_heads * head_dim` (unchanged, Q has q_len rows)
- `ld_k = n_kv_heads * head_dim` (unchanged)
- `ld_s = kv_len` (S row length = kv_len, was seq_len)
- `ld_o = n_heads * head_dim` (unchanged)

10. **GQA path**: the same mechanical changes apply to the GQA branch (`gqa_ratio > 1`). The pointer-array variant uses `M=kv_len, N=q_len, K=head_dim` for QK^T and `M=head_dim, N=q_len, K=kv_len` for PV. Replace every `seq_len * seq_len` and bare `seq_len` (when used as M or N of the rectangular GEMM) with the appropriate `q_len`/`kv_len`.

- [ ] **Step 3: Update both callers in `executor_attention.cu`**

Caller 1 at `src/graph/executor_attention.cu:747`:

```cpp
            attention_cublas_prefill(qv, kk, vv, ao, attn_scores_, nh, nkv, hd, scale, /*causal=*/true,
                                     cfg.attn_logit_softcap, stream);
```

becomes:

```cpp
            attention_cublas_prefill(qv, kk, vv, ao, attn_scores_, nh, nkv, hd, scale, /*causal=*/true,
                                     cfg.attn_logit_softcap, /*q_offset=*/0, stream);
```

Caller 2 at `src/graph/executor_attention.cu:842` (debug `force_cublas_decode` path):

```cpp
            attention_cublas_prefill(qv, k_cont, v_cont, ao, s_view, nh, nkv, hd, scale, /*causal=*/false,
                                     cfg.attn_logit_softcap, stream);
```

becomes:

```cpp
            attention_cublas_prefill(qv, k_cont, v_cont, ao, s_view, nh, nkv, hd, scale, /*causal=*/false,
                                     cfg.attn_logit_softcap, /*q_offset=*/0, stream);
```

- [ ] **Step 4: Build**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -5`
Expected: clean build.

- [ ] **Step 5: Full test suite**

Run: `make test-gpu 2>&1 | tail -10`
Expected: same totals (769 ran, 747 passed, 22 skipped, 0 failed). Square path is byte-equivalent because rectangular dims = square dims when `q_len=kv_len=seq_len, q_offset=0`.

- [ ] **Step 6: Commit**

```bash
git add src/compute/attention_cublas.h src/compute/attention_cublas.cu src/graph/executor_attention.cu
git commit -m "$(cat <<'EOF'
refactor(attention): attention_cublas_prefill takes q_offset for rectangular Q vs K

cuBLAS handles asymmetric M/N natively — split seq_len into q_len/kv_len
in QK^T and PV calls. q_offset feeds the offset-aware causal mask in the
generalized softmax kernels. Square path (q_offset=0, q_len=kv_len)
produces byte-identical output. Both existing callers updated with
explicit q_offset=0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: New unit tests for the rectangular attention path

**Files:**
- Create: `tests/test_attention_chunked.cu`
- Modify: `tests/CMakeLists.txt`

**Goal:** Verify the rectangular path is correct independently of any model. Synthetic Q/K/V where the attended K position can be read from the output.

- [ ] **Step 1: Write the test file**

```cpp
#include "compute/attention_cublas.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace imp {

// Helper: allocate FP16 device tensor [d0, d1] (or [d0, d1, d2] for S).
static Tensor make_fp16_tensor_2d(int d0, int d1) {
    int64_t shape[2] = {d0, d1};
    half* p = nullptr;
    cudaMalloc(&p, (size_t)d0 * d1 * sizeof(half));
    return Tensor(p, QType::F16, 2, shape, /*owns=*/true);
}

static Tensor make_fp16_tensor_3d(int d0, int d1, int d2) {
    int64_t shape[3] = {d0, d1, d2};
    half* p = nullptr;
    cudaMalloc(&p, (size_t)d0 * d1 * d2 * sizeof(half));
    return Tensor(p, QType::F16, 3, shape, /*owns=*/true);
}

static void fill_fp16_random(half* d_ptr, size_t n, uint32_t seed) {
    std::vector<half> h(n);
    std::srand(seed);
    for (size_t i = 0; i < n; i++) {
        h[i] = __float2half(((float)std::rand() / RAND_MAX) * 0.1f - 0.05f);
    }
    cudaMemcpy(d_ptr, h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

TEST(AttentionChunkedTest, RectangularEqualsSquareAtZeroOffset) {
    const int seq = 64, nh = 4, nkv = 4, hd = 32;
    const float scale = 1.0f / std::sqrt((float)hd);

    Tensor Q = make_fp16_tensor_2d(seq, nh * hd);
    Tensor K = make_fp16_tensor_2d(seq, nkv * hd);
    Tensor V = make_fp16_tensor_2d(seq, nkv * hd);
    Tensor O = make_fp16_tensor_2d(seq, nh * hd);
    Tensor S = make_fp16_tensor_3d(nh, seq, seq);

    fill_fp16_random((half*)Q.data, seq * nh * hd, 1);
    fill_fp16_random((half*)K.data, seq * nkv * hd, 2);
    fill_fp16_random((half*)V.data, seq * nkv * hd, 3);

    attention_cublas_prefill(Q, K, V, O, S, nh, nkv, hd, scale, /*causal=*/true,
                             /*softcap=*/0.0f, /*q_offset=*/0, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_o((size_t)seq * nh * hd);
    cudaMemcpy(h_o.data(), O.data, h_o.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Sanity: no NaN, magnitudes plausible (not all zero).
    float sum_abs = 0.0f;
    for (size_t i = 0; i < h_o.size(); i++) {
        float v = __half2float(h_o[i]);
        ASSERT_FALSE(std::isnan(v));
        sum_abs += std::fabs(v);
    }
    EXPECT_GT(sum_abs, 0.0f);

    cudaFree(Q.data); cudaFree(K.data); cudaFree(V.data);
    cudaFree(O.data); cudaFree(S.data);
}

// Synthesized Q/K to verify the offset-aware causal mask. K is one-hot at column 0
// (only position 0 has nonzero K, all others are zero), so attention scores are
// nonzero only when Q attends to position 0. With q_offset=128 and q_len=64,
// Q[i]'s absolute position is 128 + i — should attend to position 0 (causal: 0 <= 128+i).
TEST(AttentionChunkedTest, OffsetAwareCausalMask) {
    const int q_len = 64, kv_len = 192, q_offset = 128, nh = 1, nkv = 1, hd = 16;
    const float scale = 1.0f;  // simplify mask test

    Tensor Q = make_fp16_tensor_2d(q_len, nh * hd);
    Tensor K = make_fp16_tensor_2d(kv_len, nkv * hd);
    Tensor V = make_fp16_tensor_2d(kv_len, nkv * hd);
    Tensor O = make_fp16_tensor_2d(q_len, nh * hd);
    // S sized for square max — pick max(q_len, kv_len)^2 so the FP32-fits heuristic resolves.
    Tensor S = make_fp16_tensor_3d(nh, kv_len, kv_len);

    // Q: all ones in dim 0, zero elsewhere
    std::vector<half> h_q(q_len * nh * hd, __float2half(0.f));
    for (int i = 0; i < q_len; i++) h_q[i * nh * hd + 0] = __float2half(1.f);

    // K: only K[0][0] = 1, rest zero. So Q[i] · K[j] = 1 iff j==0, else 0.
    std::vector<half> h_k(kv_len * nkv * hd, __float2half(0.f));
    h_k[0 * nkv * hd + 0] = __float2half(1.f);

    // V: V[j][0] = j as a probe. After softmax over the visible K positions,
    // P will have weight 1.0 on j=0 (only nonzero score), so O[:, 0] = V[0][0] = 0.
    // We set V[0][0] = 7.0 to make this distinctive.
    std::vector<half> h_v(kv_len * nkv * hd, __float2half(0.f));
    h_v[0 * nkv * hd + 0] = __float2half(7.f);

    cudaMemcpy(Q.data, h_q.data(), h_q.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(K.data, h_k.data(), h_k.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(V.data, h_v.data(), h_v.size() * sizeof(half), cudaMemcpyHostToDevice);

    attention_cublas_prefill(Q, K, V, O, S, nh, nkv, hd, scale, /*causal=*/true,
                             /*softcap=*/0.0f, /*q_offset=*/q_offset, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_o(q_len * nh * hd);
    cudaMemcpy(h_o.data(), O.data, h_o.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Expectation: every output row's component 0 equals 7.0 (because Q saw V[0]).
    for (int i = 0; i < q_len; i++) {
        float val = __half2float(h_o[i * nh * hd + 0]);
        EXPECT_NEAR(val, 7.0f, 0.05f) << "row " << i;
    }

    cudaFree(Q.data); cudaFree(K.data); cudaFree(V.data);
    cudaFree(O.data); cudaFree(S.data);
}

TEST(AttentionChunkedTest, GQA_Ratio4) {
    const int q_len = 32, kv_len = 64, nh = 16, nkv = 4, hd = 32;
    const float scale = 1.0f / std::sqrt((float)hd);

    Tensor Q = make_fp16_tensor_2d(q_len, nh * hd);
    Tensor K = make_fp16_tensor_2d(kv_len, nkv * hd);
    Tensor V = make_fp16_tensor_2d(kv_len, nkv * hd);
    Tensor O = make_fp16_tensor_2d(q_len, nh * hd);
    Tensor S = make_fp16_tensor_3d(nh, kv_len, kv_len);

    fill_fp16_random((half*)Q.data, q_len * nh * hd, 7);
    fill_fp16_random((half*)K.data, kv_len * nkv * hd, 8);
    fill_fp16_random((half*)V.data, kv_len * nkv * hd, 9);

    attention_cublas_prefill(Q, K, V, O, S, nh, nkv, hd, scale, /*causal=*/true,
                             /*softcap=*/0.0f, /*q_offset=*/16, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_o((size_t)q_len * nh * hd);
    cudaMemcpy(h_o.data(), O.data, h_o.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < h_o.size(); i++) {
        ASSERT_FALSE(std::isnan(__half2float(h_o[i])));
    }

    cudaFree(Q.data); cudaFree(K.data); cudaFree(V.data);
    cudaFree(O.data); cudaFree(S.data);
}

}  // namespace imp
```

- [ ] **Step 2: Register in `tests/CMakeLists.txt`**

Add `tests/test_attention_chunked.cu` to the same target as `test_attention_paged` or `test_attention_cublas` (whichever exists).

- [ ] **Step 3: Build and run**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -5`
Run: `build/tests/test-attention --gtest_filter="AttentionChunkedTest.*" -v`
Expected: 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_attention_chunked.cu tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
test(attention): rectangular attention_cublas_prefill correctness

Three unit tests: square-equiv at zero offset, offset-aware causal mask
via one-hot K probe, GQA ratio=4 with non-zero offset. Validates the
Task-4 refactor independently of any model.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Chunked-prefill dispatch in `run_attention`

**Files:**
- Modify: `src/graph/executor_attention.cu:688` (`if (state.is_prefill)` branch)

**Goal:** Detect `state.prefill_offset > 0`, gather past KV, build full K/V, dispatch rectangular `attention_cublas_prefill` with `q_offset`. Fall through to existing path for `q_offset == 0`.

- [ ] **Step 1: Add the include for `kv_gather.h`**

In `src/graph/executor_attention.cu` near the top, add:

```cpp
#include "compute/kv_gather.h"
```

(Place near `#include "compute/attention_paged.h"`.)

- [ ] **Step 2: Insert chunked-prefill dispatch**

Find the existing line:

```cpp
    if (state.is_prefill) {
        bool sliding_active = (layer_sliding_window > 0 && n > layer_sliding_window);
```

(at `src/graph/executor_attention.cu:688-689`).

Insert immediately after the `bool sliding_active` line:

```cpp
        // Chunked prefill: when prefill_offset > 0, queries from this chunk must
        // attend to past chunks K/V already in the paged cache. Gather past
        // [0, prefill_offset) KV → contiguous, append current chunk, then run
        // rectangular attention_cublas_prefill with q_offset.
        const int q_offset = state.prefill_offset;
        if (q_offset > 0) {
            KVCache* cache = state.kv_cache;
            QType kvt = cache->qtype();
            // Defense-in-depth: engine resolves out-of-scope models to chunk_size=0,
            // so this code only runs for FP16 / FP8 KV without SWA / dual-head_dim.
            if ((kvt != QType::F16 && kvt != QType::FP8_E4M3) || sliding_active || per_layer_shapes) {
                IMP_LOG_ERROR(
                    "chunked_prefill: unsupported config (kv=%d swa=%d per_layer=%d) at L%d — "
                    "engine should have prevented this",
                    (int)kvt, (int)sliding_active, (int)per_layer_shapes, layer);
                std::abort();
            }

            int kv_layer = get_kv_layer(kv_layer_map_, layer);
            int kv_bs = cache->block_size();
            int ctx_len = q_offset + n;
            size_t full_bytes = (size_t)ctx_len * nkv * hd * sizeof(half);

            half* k_full = nullptr;
            half* v_full = nullptr;
            cudaMallocAsync(&k_full, full_bytes, stream);
            cudaMallocAsync(&v_full, full_bytes, stream);

            // Gather past KV [0, q_offset) directly into k_full[0..q_offset], v_full[0..q_offset].
            if (kvt == QType::F16) {
                paged_kv_gather_fp16(k_full, static_cast<const half*>(cache->k_ptr(kv_layer, 0)),
                                     state.block_tables, q_offset, kv_bs, nkv, hd, stream);
                paged_kv_gather_fp16(v_full, static_cast<const half*>(cache->v_ptr(kv_layer, 0)),
                                     state.block_tables, q_offset, kv_bs, nkv, hd, stream);
            } else {  // FP8_E4M3
                float kv_scale = (!kv_scales_.empty() && kv_layer < (int)kv_scales_.size())
                                     ? kv_scales_[kv_layer] : 1.0f;
                paged_kv_gather_fp8_to_fp16(
                    k_full, static_cast<const __nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
                    state.block_tables, kv_scale, q_offset, kv_bs, nkv, hd, stream);
                paged_kv_gather_fp8_to_fp16(
                    v_full, static_cast<const __nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)),
                    state.block_tables, kv_scale, q_offset, kv_bs, nkv, hd, stream);
            }

            // Append current chunk's K/V at offset q_offset.
            cudaMemcpyAsync(k_full + (size_t)q_offset * nkv * hd, kk.data,
                            (size_t)n * nkv * hd * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(v_full + (size_t)q_offset * nkv * hd, vv.data,
                            (size_t)n * nkv * hd * sizeof(half), cudaMemcpyDeviceToDevice, stream);

            int64_t kv_full_shape[2] = {(int64_t)ctx_len, (int64_t)(nkv * hd)};
            Tensor k_full_t(k_full, QType::F16, 2, kv_full_shape, /*owns=*/false);
            Tensor v_full_t(v_full, QType::F16, 2, kv_full_shape, /*owns=*/false);

            attention_cublas_prefill(qv, k_full_t, v_full_t, ao, attn_scores_, nh, nkv, hd, scale,
                                     /*causal=*/true, cfg.attn_logit_softcap, q_offset, stream);

            cudaFreeAsync(k_full, stream);
            cudaFreeAsync(v_full, stream);

            // Persist current chunk's K/V (same as non-chunked path)
            write_kv_cache(layer, state, stream);
            return;
        }

        // Existing single-chunk / first-chunk path follows unchanged.
```

The `return;` is critical: it skips the existing dispatch branch (cuBLAS / FMHA / naive). The existing code below stays as-is.

- [ ] **Step 3: Verify the existing fall-through is intact**

Run: `sed -n '755,775p' src/graph/executor_attention.cu`
Expected: still ends with `write_kv_cache(layer, state, stream);` after the existing prefill dispatch.

- [ ] **Step 4: Build**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -5`
Expected: clean build.

- [ ] **Step 5: Full test suite — `q_offset=0` paths must stay byte-equivalent**

Run: `make test-gpu 2>&1 | tail -10`
Expected: same totals (769 ran, 747 passed, 22 skipped, 0 failed). The new code only triggers for `q_offset > 0`, which no current test exercises — so existing behavior is unchanged.

- [ ] **Step 6: Smoke test the new dispatch path manually**

Build the CLI: `cmake --build build -j$(nproc) --target imp-cli 2>&1 | tail -3`

Run a single-chunk smoke (should be unchanged):
```bash
./build/tools/imp-cli/imp-cli -m models/Qwen3-4B-Instruct-2507-Q8_0.gguf \
    --prefill-chunk-size 0 -p "What is the capital of France?" -n 8
```
Expected: outputs "Paris" or similar coherent answer.

Run with explicit chunking enabled (NEW behavior — exercises the gather path):
```bash
./build/tools/imp-cli/imp-cli -m models/Qwen3-4B-Instruct-2507-Q8_0.gguf \
    --prefill-chunk-size 64 -p "Write a short story about a cat with at least 100 words. Begin." -n 16
```
Expected: coherent text continuation. If output is degenerate ("own own own" or NaN-style garbage), the gather path has a bug — debug before continuing.

- [ ] **Step 7: Commit**

```bash
git add src/graph/executor_attention.cu
git commit -m "$(cat <<'EOF'
feat(graph): chunked-prefill dispatch in run_attention

When state.prefill_offset > 0, gather past KV [0, q_offset) from paged
cache into a contiguous flat buffer (FP8 dequants on the fly), append
current chunk's K/V, dispatch attention_cublas_prefill with q_offset.
Defense-in-depth check rejects unsupported configs (engine clamps
chunk_size to 0 for those archs anyway).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Per-arch `resolve_prefill_chunk_size` resolver

**Files:**
- Modify: `src/runtime/engine.h:62` (sentinel doc) and add method declarations
- Modify: `src/runtime/engine.cpp:1642` (`step_prefill`) to use resolver
- Modify: `include/imp/config.h:53` (sentinel doc)

**Goal:** Sentinel `-1` triggers per-arch default. Hardcode `0` for Gemma-4 / hybrid / sub-byte-KV; `512` for in-scope. Explicit `>0` for unsupported arch → log WARN, clamp to `0`.

- [ ] **Step 1: Update `imp/config.h` sentinel doc**

In `include/imp/config.h:53`, find:

```cpp
    int prefill_chunk_size;  // Max tokens per prefill chunk (0 = no chunking)
```

Replace with:

```cpp
    int prefill_chunk_size;  // Max tokens per prefill chunk.
                             //   -1 = use per-arch default (recommended)
                             //   0  = explicit single-chunk (force)
                             //   >0 = explicit chunk size (rejected with WARN if arch unsupported)
```

- [ ] **Step 2: Add resolver to `engine.h`**

In `src/runtime/engine.h`, find the `Engine` class private section and add:

```cpp
    // Whether the model arch + KV dtype combination supports chunked prefill.
    // Returns true for full-attention models (Qwen3, Llama, Mistral) with FP16
    // or FP8 KV cache. Returns false for Gemma-4 (SWA + dual head_dim), hybrid
    // models (GDN/Mamba2), and sub-byte KV dtypes.
    bool supports_chunked_prefill_() const;

    // Resolves config_.prefill_chunk_size considering arch + KV dtype.
    //   sentinel -1 → per-arch default (512 if supported, 0 otherwise)
    //   explicit 0  → 0 (force single-chunk, always respected)
    //   explicit >0 → that value if supported, else 0 with WARN
    int resolve_prefill_chunk_size_() const;
```

(Place near other `private:` helper methods.)

- [ ] **Step 3: Implement resolver in `engine.cpp`**

In `src/runtime/engine.cpp`, add the function bodies. Place them near other private helpers (search for `Engine::step_prefill_one` and place above it):

```cpp
bool Engine::supports_chunked_prefill_() const {
    if (!model_)
        return false;
    const auto& cfg = model_->config();
    // Out-of-scope archs: SWA / dual-head_dim / hybrid (GDN / Mamba2).
    if (cfg.arch == ModelArch::GEMMA4) return false;
    if (cfg.arch == ModelArch::QWEN35) return false;
    if (cfg.arch == ModelArch::QWEN35_MOE) return false;
    if (cfg.arch == ModelArch::QWEN36_MOE) return false;
    if (cfg.arch == ModelArch::NEMOTRON_H_MOE) return false;
    // Out-of-scope KV dtypes: only FP16 + FP8 are wired through paged_kv_gather.
    if (kv_cache_raw_) {
        QType kvt = kv_cache_raw_->qtype();
        if (kvt != QType::F16 && kvt != QType::FP8_E4M3)
            return false;
    }
    return true;
}

int Engine::resolve_prefill_chunk_size_() const {
    int explicit_val = config_.prefill_chunk_size;
    if (explicit_val < 0) {
        return supports_chunked_prefill_() ? 512 : 0;
    }
    if (explicit_val == 0)
        return 0;
    // explicit_val > 0
    if (!supports_chunked_prefill_()) {
        IMP_LOG_WARN(
            "prefill_chunk_size=%d ignored: arch=%d / kv_dtype=%d not in chunked-prefill scope; using 0",
            explicit_val, (int)model_->config().arch,
            kv_cache_raw_ ? (int)kv_cache_raw_->qtype() : -1);
        return 0;
    }
    return explicit_val;
}
```

- [ ] **Step 4: Use the resolver in `step_prefill`**

In `src/runtime/engine.cpp:1643`, find:

```cpp
    int effective_chunk = config_.prefill_chunk_size > 0 ? config_.prefill_chunk_size
                                                         : executor_->max_tokens();
```

Replace with:

```cpp
    int resolved = resolve_prefill_chunk_size_();
    int effective_chunk = (resolved > 0) ? resolved : executor_->max_tokens();
```

The downstream `if (effective_chunk > executor_->max_tokens())` clamp at line 1651 is preserved as-is — it still bounds the resolver's output.

- [ ] **Step 5: Update default in `Config` constructor**

The default value of `prefill_chunk_size` in `imp_config_default()` (look for it in `include/imp/config.h` companion `.cpp` or `src/api/config.cpp`):

Run: `grep -rn "prefill_chunk_size = 0\|prefill_chunk_size=0\|\.prefill_chunk_size = " src/ include/ | head -5`

Wherever the default is set to `0`, change to `-1` (sentinel = "use per-arch default"). The CLI / server flag wiring should pass through user-explicit values intact.

- [ ] **Step 6: Build**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -5`
Expected: clean build.

- [ ] **Step 7: Test that explicit user value still wins for in-scope arch**

Smoke:
```bash
./build/tools/imp-cli/imp-cli -m models/Qwen3-4B-Instruct-2507-Q8_0.gguf \
    --prefill-chunk-size 256 -p "Capital of France?" -n 4
```
Expected: coherent output ("Paris"). Internally chunks at 256.

Smoke (default sentinel):
```bash
./build/tools/imp-cli/imp-cli -m models/Qwen3-4B-Instruct-2507-Q8_0.gguf \
    -p "Capital of France?" -n 4
```
Expected: coherent. Engine resolves to 512 default for Qwen3-4B + FP16 KV.

Smoke (Gemma-4 should ignore explicit):
```bash
./build/tools/imp-cli/imp-cli -m /home/kekz/models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
    --prefill-chunk-size 512 -p "Hello" -n 4
```
Expected: coherent output + WARN line in stderr saying `prefill_chunk_size=512 ignored: arch=11 ...`.

- [ ] **Step 8: Full test suite**

Run: `make test-gpu 2>&1 | tail -10`
Expected: same totals — no test currently exercises the resolver, but no test should break either.

- [ ] **Step 9: Commit**

```bash
git add src/runtime/engine.h src/runtime/engine.cpp include/imp/config.h src/api/config.cpp
# (the last path may differ — adjust based on the grep in Step 5)
git commit -m "$(cat <<'EOF'
feat(engine): resolve_prefill_chunk_size with per-arch default + sentinel

Sentinel -1 = "use per-arch default" (512 for full-attention + FP16/FP8 KV,
0 otherwise). Explicit 0 = force single-chunk. Explicit >0 for unsupported
arch → WARN + clamp to 0. Default Config::prefill_chunk_size flips from 0
to -1 to enable chunking by default for in-scope models.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: E2E logits-equality test battery

**Files:**
- Create: `tests/test_chunked_prefill.cu`
- Modify: `tests/CMakeLists.txt`

**Goal:** Verify chunked vs single-chunk produce equivalent logits across the in-scope models. Tests skip if model files absent (consistent with existing e2e tests in `tests/`).

- [ ] **Step 1: Find an existing e2e test pattern to mirror**

Run: `grep -l "GTEST_SKIP\|requires.*model.*file\|gguf" tests/*.cu | head -5`

Pick the closest pattern (likely something like `tests/test_e2e_models.cu` or `tests/test_engine_e2e.cu`). Read ~30 lines to confirm the engine bring-up boilerplate.

- [ ] **Step 2: Write the test file**

```cpp
#include "imp/api.h"
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

namespace imp_test {

// Skip if model file is not present in the test environment.
static bool model_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Run a forward pass with given chunk_size, capture last-token logits via greedy sampling.
// Returns the first 8 generated token IDs (greedy, temp=0). Empty vector on failure.
static std::vector<int> generate_greedy(const std::string& model_path, const std::string& prompt,
                                         int chunk_size, int n_predict, bool use_fp8_kv = false) {
    imp_config cfg;
    imp_config_default(&cfg);
    cfg.prefill_chunk_size = chunk_size;
    if (use_fp8_kv) {
        cfg.kv_cache_dtype = IMP_QTYPE_FP8_E4M3;  // adjust to actual constant name
    }

    imp_engine* eng = nullptr;
    if (imp_engine_create(model_path.c_str(), &cfg, &eng) != IMP_OK)
        return {};

    imp_request req;
    imp_request_default(&req);
    req.prompt = prompt.c_str();
    req.max_tokens = n_predict;
    req.temperature = 0.0f;  // greedy
    req.top_k = 1;

    int64_t req_id = imp_engine_submit(eng, &req);
    std::vector<int> tokens;
    while (true) {
        imp_token tok;
        int32_t status = imp_engine_poll(eng, req_id, &tok);
        if (status == IMP_FINISHED) break;
        if (status == IMP_HAS_TOKEN) tokens.push_back(tok.token_id);
        if ((int)tokens.size() >= n_predict) break;
    }

    imp_engine_destroy(eng);
    return tokens;
}

class ChunkedPrefillTest : public ::testing::Test {
protected:
    static constexpr const char* QWEN3_4B = "models/Qwen3-4B-Instruct-2507-Q8_0.gguf";
    static constexpr const char* LLAMA_3B = "models/Llama-3.2-3B-Instruct-Q8_0.gguf";

    // ~2049-token prompt: 256 lines of varied content forces ≥4 chunks at chunk=512.
    static std::string long_prompt() {
        std::string p = "Summarize the following list:\n";
        for (int i = 0; i < 256; i++) {
            p += "Item " + std::to_string(i) + ": ";
            for (int w = 0; w < 6; w++) p += "word" + std::to_string(w) + " ";
            p += "\n";
        }
        return p;
    }
};

TEST_F(ChunkedPrefillTest, Qwen3_4B_Q8_0_FP16_KV_LogitsEqual) {
    if (!model_exists(QWEN3_4B)) GTEST_SKIP() << "model not present";
    auto single = generate_greedy(QWEN3_4B, long_prompt(), /*chunk=*/0, 8);
    auto chunked512 = generate_greedy(QWEN3_4B, long_prompt(), /*chunk=*/512, 8);
    auto chunked128 = generate_greedy(QWEN3_4B, long_prompt(), /*chunk=*/128, 8);
    auto chunked64 = generate_greedy(QWEN3_4B, long_prompt(), /*chunk=*/64, 8);

    ASSERT_EQ(single.size(), 8u);
    ASSERT_EQ(chunked512.size(), 8u);
    ASSERT_EQ(chunked128.size(), 8u);
    ASSERT_EQ(chunked64.size(), 8u);

    // Greedy generation: token-for-token equality.
    EXPECT_EQ(single, chunked512);
    EXPECT_EQ(single, chunked128);
    EXPECT_EQ(single, chunked64);
}

TEST_F(ChunkedPrefillTest, Qwen3_4B_Q8_0_FP8_KV_LogitsEqual) {
    if (!model_exists(QWEN3_4B)) GTEST_SKIP();
    auto single = generate_greedy(QWEN3_4B, long_prompt(), 0, 8, /*fp8=*/true);
    auto chunked = generate_greedy(QWEN3_4B, long_prompt(), 512, 8, /*fp8=*/true);
    ASSERT_EQ(single.size(), 8u);
    ASSERT_EQ(chunked.size(), 8u);
    // FP8 KV introduces small noise; allow first-token mismatch but require at least 6/8 match.
    int matches = 0;
    for (int i = 0; i < 8; i++) if (single[i] == chunked[i]) matches++;
    EXPECT_GE(matches, 6) << "only " << matches << "/8 tokens matched";
}

TEST_F(ChunkedPrefillTest, Llama_3_2_3B_Chunk_64_LogitsEqual) {
    if (!model_exists(LLAMA_3B)) GTEST_SKIP();
    auto single = generate_greedy(LLAMA_3B, long_prompt(), 0, 8);
    auto chunked = generate_greedy(LLAMA_3B, long_prompt(), 64, 8);  // non-block-aligned (block=16)
    ASSERT_EQ(single.size(), 8u);
    EXPECT_EQ(single, chunked);
}

TEST_F(ChunkedPrefillTest, Qwen3_4B_ChunkLargerThanPrompt) {
    if (!model_exists(QWEN3_4B)) GTEST_SKIP();
    std::string short_p = "What is 2+2?";
    auto single = generate_greedy(QWEN3_4B, short_p, 0, 4);
    auto chunked = generate_greedy(QWEN3_4B, short_p, 4096, 4);  // chunk >> prompt
    EXPECT_EQ(single, chunked);
}

TEST_F(ChunkedPrefillTest, Qwen3_4B_GenerationCoherent) {
    if (!model_exists(QWEN3_4B)) GTEST_SKIP();
    auto out = generate_greedy(QWEN3_4B, long_prompt(), 512, 32);
    ASSERT_EQ(out.size(), 32u);
    // Coherence proxy: not all the same token (no degeneration loop)
    int unique = 0;
    std::vector<int> seen;
    for (int t : out) {
        bool found = false;
        for (int s : seen) if (s == t) { found = true; break; }
        if (!found) { seen.push_back(t); unique++; }
    }
    EXPECT_GE(unique, 4) << "generation collapsed to repetition";
}

}  // namespace imp_test
```

(Note: the exact API names — `imp_engine_create`, `imp_request_default`, `imp_engine_poll`, etc. — must match the actual C API in `include/imp/`. Run `grep -n "imp_engine_create\|imp_request_default\|imp_engine_submit\|imp_engine_poll" include/imp/api.h` and adjust the boilerplate to match the real signatures.)

- [ ] **Step 3: Register in `tests/CMakeLists.txt`**

Add `tests/test_chunked_prefill.cu` to the e2e test target (likely `test-e2e` or `test-models`).

- [ ] **Step 4: Build and run**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -5`
Run: `build/tests/test-e2e --gtest_filter="ChunkedPrefillTest.*" -v`
Expected: 5 tests PASS (assuming Qwen3-4B and Llama-3.2-3B GGUF files are mounted into the test env).

If model files aren't present → tests SKIP cleanly, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add tests/test_chunked_prefill.cu tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
test(prefill): chunked-vs-single logits-equality battery

Five e2e tests covering Qwen3-4B Q8_0 (FP16 + FP8 KV), Llama-3.2-3B
non-block-aligned chunk, chunk > prompt boundary, and generation
coherence. Greedy-sampling token-for-token equality for FP16 KV;
6/8 match tolerance for FP8 KV (small drift from FP8 dequant noise).
Skips cleanly when model files absent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Pin `verify-fast` to chunk=0 + add `perf_baseline_chunked.json`

**Files:**
- Modify: `scripts/verify-fast.sh` (or wherever `make verify-fast` invokes the smoke)
- Create: `tests/perf_baseline_chunked.json`
- Modify: `Makefile` if needed for new bench target

**Goal:** Existing baselines stay apples-to-apples (chunk=0). New baseline file measures chunk=512 performance for in-scope models with looser regression gates.

- [ ] **Step 1: Find the verify-fast invocation**

Run: `grep -rn "verify-fast\|smoke.*test\|capital.*France" Makefile scripts/ | head -10`

Identify the script that runs the smoke test. The smoke is typically `imp-cli -p "What is the capital of France?" ...`.

- [ ] **Step 2: Add `--prefill-chunk-size 0` to the smoke**

In the identified script (likely `scripts/verify-fast.sh` or inline in `Makefile`), find the `imp-cli` invocation and add `--prefill-chunk-size 0` to keep the baseline untouched.

Example (adjust to actual script):
```bash
./build/tools/imp-cli/imp-cli -m "$MODEL" -p "What is the capital of France?" -n 8 \
    --prefill-chunk-size 0
```

- [ ] **Step 3: Generate the chunked baseline**

Run a fresh perf measurement with chunk=512 active for in-scope models:

```bash
# Qwen3-4B Q8_0
./build/tools/imp-bench/imp-bench -m models/Qwen3-4B-Instruct-2507-Q8_0.gguf \
    --prefill-chunk-size 512 -pp 512 -tg 256 --runs 3 > /tmp/q4b.txt

# Qwen3-8B Q8_0
./build/tools/imp-bench/imp-bench -m models/Qwen3-8B-Q8_0.gguf \
    --prefill-chunk-size 512 -pp 512 -tg 256 --runs 3 > /tmp/q8b.txt

# Llama-3.2-3B Q8_0
./build/tools/imp-bench/imp-bench -m models/Llama-3.2-3B-Instruct-Q8_0.gguf \
    --prefill-chunk-size 512 -pp 512 -tg 256 --runs 3 > /tmp/l3b.txt
```

Extract the `tg256` and `pp512` values from each output and write `tests/perf_baseline_chunked.json`:

```json
{
  "version": 1,
  "comment": "Chunked-prefill perf baseline (chunk=512). Looser gates than perf_baseline.json (5% decode / 8% prefill) to account for gather + rect-attn overhead per chunk.",
  "regression_thresholds": {
    "decode": 0.05,
    "prefill": 0.08
  },
  "models": {
    "Qwen3-4B-Q8_0": {
      "tg256": <measured>,
      "pp512": <measured>
    },
    "Qwen3-8B-Q8_0": {
      "tg256": <measured>,
      "pp512": <measured>
    },
    "Llama-3.2-3B-Q8_0": {
      "tg256": <measured>,
      "pp512": <measured>
    }
  }
}
```

Replace `<measured>` with actual numbers from the bench runs. Round to 1 decimal place.

- [ ] **Step 4: Optional — add a `make verify-chunked` target**

If the existing `verify` script reads `tests/perf_baseline.json`, add a parallel `make verify-chunked` target that reads `tests/perf_baseline_chunked.json`. This keeps the two baselines independently gated.

In `Makefile`, near the existing `verify-fast` rule, add:

```make
verify-chunked: build
	@echo "Verifying chunked-prefill perf baseline..."
	@scripts/verify-perf.sh tests/perf_baseline_chunked.json
```

(Adjust the script invocation to match how `verify-fast` actually calls into perf gating today.)

- [ ] **Step 5: Run both verify targets**

```bash
make verify-fast
```
Expected: baselines from `perf_baseline.json` (single-chunk) match within 3%/5%.

```bash
make verify-chunked
```
Expected: baselines from `perf_baseline_chunked.json` (chunk=512) match within 5%/8%.

If `verify-chunked` regresses on first run, the perf overhead is too high — investigate (likely the per-chunk `cudaMallocAsync` is the culprit; consider pre-allocating a workspace as a follow-up).

- [ ] **Step 6: Commit**

```bash
git add scripts/verify-fast.sh tests/perf_baseline_chunked.json Makefile
git commit -m "$(cat <<'EOF'
bench: pin verify-fast to chunk=0; add perf_baseline_chunked.json

Existing perf_baseline.json measurements remain valid by pinning
verify-fast smoke to --prefill-chunk-size 0. New perf_baseline_chunked.json
captures Qwen3-4B/8B Q8_0 + Llama-3.2-3B Q8_0 with chunk=512 active and
looser 5%/8% gates to absorb gather + rect-attn per-chunk overhead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Roadmap close + CHANGELOG

**Files:**
- Modify: `docs/roadmap.md` (Known limitations section, "Chunked prefill" subsection)
- Modify: `CHANGELOG.md`

**Goal:** Move L2 from open to closed. Document the new default + scope.

- [ ] **Step 1: Update `docs/roadmap.md`**

In `docs/roadmap.md`, find the section starting with `### Chunked prefill: missing past-KV in attention (paged-prefill kernel pending)` (around line 15). Replace it with:

```markdown
### Chunked prefill scope (full-attention + FP16/FP8 KV)

Default `prefill_chunk_size = 512` for full-attention models (Qwen3, Llama, Mistral) with FP16 or FP8 KV cache. Past chunks' K/V are read from the paged cache via `paged_kv_gather_*` and concatenated with the current chunk before a rectangular `attention_cublas_prefill` with `q_offset`-aware causal masking. PR #114 mitigation (default `prefill_chunk_size = 0`) is replaced by `Engine::resolve_prefill_chunk_size_()` which clamps to 0 for out-of-scope archs.

**Out-of-scope** — stay at `prefill_chunk_size = 0` via per-arch default; explicit `--prefill-chunk-size N` is logged + clamped to 0:

- Gemma-4 (SWA + dual head_dim 256/512)
- Hybrid models with non-attention layers (Qwen3.5/3.6 GDN, Nemotron-H Mamba2)
- Sub-byte KV cache dtypes (INT4, NVFP4, TurboQuant variants)

Each excluded class is a separate larger work item (paged-prefill kernel with SWA-aware mask / dual-head_dim support / sub-byte dequant during gather).
```

- [ ] **Step 2: Update `CHANGELOG.md`**

Add an entry near the top:

```markdown
## [Unreleased] - 2026-05-08

### Fixed

- **Chunked prefill correctness**: prefill chunks ≥2 now correctly read past chunks' K/V from the paged cache. New `paged_kv_gather_*` kernels + rectangular `attention_cublas_prefill(q_offset)`. Previously, `prefill_chunk_size > 0` produced silently-wrong logits for full-attention models and full degeneration for SWA models like Gemma-4.

### Added

- `Engine::resolve_prefill_chunk_size_()` with sentinel `-1` = "use per-arch default" (512 for full-attention + FP16/FP8 KV, 0 otherwise). Default `Config::prefill_chunk_size` flips from `0` to `-1`.
- `tests/perf_baseline_chunked.json` — perf baseline for chunked default with looser 5%/8% gates.
- New unit tests: `test_kv_gather`, `test_attention_chunked`, `test_chunked_prefill`.

### Changed

- `attention_cublas_prefill` signature now takes `int q_offset` (0 = square path, byte-equivalent to prior behavior).
- `causal_softmax_inplace_kernel` and `causal_softmax_fp32_inplace_kernel` generalized to `(S, q_len, kv_len, q_offset, causal)`.
- `make verify-fast` smoke now pins `--prefill-chunk-size 0` to keep `perf_baseline.json` apples-to-apples.
```

- [ ] **Step 3: Verify formatting**

Run: `head -40 CHANGELOG.md docs/roadmap.md`
Expected: clean Markdown.

- [ ] **Step 4: Commit**

```bash
git add docs/roadmap.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(roadmap): close L2 paged-prefill; document chunked default + scope

Chunked prefill now correct by default for full-attention + FP16/FP8 KV.
Out-of-scope archs (Gemma-4 / hybrid / sub-byte KV) stay single-chunk via
Engine::resolve_prefill_chunk_size_().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist

After completing all tasks, run this checklist:

- [ ] All 10 tasks complete, every step checked.
- [ ] `make test-gpu` green (769+ passes, 0 failures).
- [ ] `make verify-fast` green (within 3%/5% of `perf_baseline.json`).
- [ ] `make verify-chunked` green (within 5%/8% of `perf_baseline_chunked.json`).
- [ ] Three smoke prompts (Qwen3-4B chunk=0, chunk=64; Gemma-4 with explicit `--prefill-chunk-size 512` produces WARN) all behave as expected.
- [ ] `git log --oneline | head -10` shows 10 commits with clear messages following Conventional Commits.
- [ ] No `TODO`, `FIXME`, or `XXX` introduced in source.

---

## Acceptance summary

This plan implements the spec at `docs/superpowers/specs/2026-05-08-paged-prefill-kernel-design.md`.

After all tasks complete:
- Chunked prefill produces correct logits for in-scope models (full-attention + FP16/FP8 KV).
- Default `prefill_chunk_size = 512` is active for in-scope models; out-of-scope archs auto-clamped to 0.
- All existing tests stay green; 11 new unit/e2e tests cover the new code paths.
- Two perf baselines (single-chunk + chunked) gate independently.
- Roadmap L2 closed.
