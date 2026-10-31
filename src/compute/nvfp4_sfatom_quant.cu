// NVFP4 SfAtom layout, weight scale conversion and activation quantizers for the block-scaled NVFP4 GEMMs
// (own tile and grouped kernels, cuBLASLt block-scaled on refusal). Moved
// verbatim from gemm_cutlass_sm120.cu. Weight conversion (once at init): packed FP4 [N,K/2] borrowed,
// micro_scales linear [N,K/16] -> SfAtom UE4M3, tensor_scale stays the GEMM alpha. Activations per prefill
// call: FP16 [M,K] -> NVFP4 packed [M,K/2] + SfAtom UE4M3 scales. No CUTLASS include.

#include "compute/gemm_cutlass_sm120.h"
#include "compute/gpt_oss_glu.cuh"
#include "model/model_config.h"
#include "quant/nvfp4_quant.h"
#include "quant/fp8_utils.cuh"
#include "core/cuda_static_reset.h"
#include "core/cuda_errors.h"
#include "core/logging.h"
#include "core/device_info.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <type_traits>

#include "core/process_diag.h"

namespace imp {

// SfAtom layout (hardware-independent), K-major SFVecSize=16: Shape ((32,4),(16,4)),
// Stride ((16,4),(0,1)). Each atom covers 128 rows x 4 scale-groups (=64 K elements);
// atom size = 128*4 = 512 bytes. tile_to_shape tiles atoms over (rows,K) with
// Step<_2,_1>: K-dim tiles are inner (faster-changing), row tiles outer.

static constexpr int kSFVecSize = 16;
static constexpr int kAtomRows = 128;                          // 32 * 4
static constexpr int kAtomKGroups = 4;                         // 4 scale groups per atom
static constexpr int kAtomKElems = kSFVecSize * kAtomKGroups;  // 64
static constexpr int kAtomSize = kAtomRows * kAtomKGroups;     // 512

// Compute SfAtom offset for logical scale factor at (row, k_group).
__device__ __host__ __forceinline__ int sfatom_offset(int row, int k_group, int n_k_tiles) {
    int tile_row = row / kAtomRows;
    int tile_k = k_group / kAtomKGroups;
    int row_local = row % kAtomRows;
    int k_local = k_group % kAtomKGroups;

    int n0 = row_local % 32;  // within 32-row sub-block
    int n1 = row_local / 32;  // which of 4 sub-blocks

    int atom_offset = (n0 * 16) + (n1 * 4) + k_local;
    int tile_base = ((tile_row * n_k_tiles) + tile_k) * kAtomSize;
    return tile_base + atom_offset;
}

size_t cutlass_nvfp4_sf_size(int rows, int K) {
    int n_row_tiles = (rows + kAtomRows - 1) / kAtomRows;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;
    return static_cast<size_t>(n_row_tiles) * n_k_tiles * kAtomSize;
}

// ---------------------------------------------------------------------------
// GPU kernels for weight conversion
// ---------------------------------------------------------------------------

// Converts micro_scales from linear to SfAtom layout (no tensor_scale absorption;
// deferred to the GEMM epilogue alpha for precision). Source [N,K/16] FP8 E4M3
// (signed, always positive for scales); dest SfAtom UE4M3 (unsigned micro_scale only,
// not combined).
__global__ void convert_scales_sfatom_kernel(const uint8_t* __restrict__ src_ms,  // [N, K/16] linear
                                             uint8_t* __restrict__ dst_sf,        // SfAtom layout
                                             int N, int K, int n_k_tiles) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int K_groups = K / kSFVecSize;
    int total = N * K_groups;
    if (idx >= total)
        return;

    int n = idx / K_groups;
    int k_group = idx % K_groups;

    // Reads signed E4M3 micro-scale, drops the sign (always positive for scales),
    // re-encodes as UE4M3 via the shared float<->E4M3 helper: UE4M3 is bit-identical to
    // positive E4M3, so float_to_fp8_e4m3 on a positive argument yields the UE4M3 byte
    // directly (sign bit=0).
    float combined = fabsf(fp8_e4m3_to_float_fast(src_ms[idx]));
    dst_sf[sfatom_offset(n, k_group, n_k_tiles)] = float_to_fp8_e4m3(combined);
}

// MoE variant: one launch converts SF for all `ne` experts. Source stride N*K_groups
// bytes/expert; dest stride cutlass_nvfp4_sf_size(N,K) bytes/expert. blockIdx.y selects
// the expert; inner work identical to the single-tensor kernel.
__global__ void convert_scales_sfatom_moe_kernel(const uint8_t* __restrict__ src_ms,
                                                 uint8_t* __restrict__ dst_sf, int N, int K, int n_k_tiles,
                                                 size_t native_stride_per_expert,
                                                 size_t sfatom_stride_per_expert,
                                                 const int32_t* __restrict__ expert_offsets) {
    int K_groups = K / kSFVecSize;
    int total = N * K_groups;
    int e = blockIdx.y;
    if (expert_offsets && expert_offsets[e + 1] == expert_offsets[e])
        return;

    const uint8_t* src_e = src_ms + (static_cast<size_t>(e) * native_stride_per_expert);
    uint8_t* dst_e = dst_sf + (static_cast<size_t>(e) * sfatom_stride_per_expert);
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        int n = idx / K_groups;
        int k_group = idx % K_groups;
        float combined = fabsf(fp8_e4m3_to_float_fast(src_e[idx]));
        dst_e[sfatom_offset(n, k_group, n_k_tiles)] = float_to_fp8_e4m3(combined);
    }
}

// ---------------------------------------------------------------------------
// Activation quantization: FP16 [M, K] → NVFP4 packed + SfAtom UE4M3 scales
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t quantize_abs_to_fp4(float abs_val) {
    // Branchless: count of midpoint thresholds exceeded gives the E2M1 code.
    uint8_t code = (abs_val >= 0.25f) + (abs_val >= 0.75f) + (abs_val >= 1.25f) + (abs_val >= 1.75f) +
                   (abs_val >= 2.5f) + (abs_val >= 3.5f) + (abs_val >= 5.0f);
    return code;  // 0..7
}

// HW FP4 conversion: two scaled FP32 → packed byte (low=v0, high=v1).
// IEEE RNE rounding; sm_120+ only.
__device__ __forceinline__ uint8_t pack_fp4_pair_hw(float v0, float v1) {
#if __CUDA_ARCH__ >= 1200
    uint32_t out;
    asm volatile(
        "{ .reg .b8 b;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "  cvt.u32.u8 %0, b; }\n"
        : "=r"(out)
        : "f"(v0), "f"(v1));
    return static_cast<uint8_t>(out);
#else
    uint8_t sign0 = (v0 < 0.0f) ? 1u : 0u;
    uint8_t sign1 = (v1 < 0.0f) ? 1u : 0u;
    uint8_t c0 = (sign0 << 3) | quantize_abs_to_fp4(fabsf(v0));
    uint8_t c1 = (sign1 << 3) | quantize_abs_to_fp4(fabsf(v1));
    return (c1 << 4) | c0;
#endif
}

// Given 16 precomputed values + absmax, encodes UE4M3 scale and packs FP4 bytes
// (reusable for fused paths like SwiGLU+quantize). UE4M3 scale saturates at 448: a
// micro-block with absmax > 448*6=2688 gets a scale clipped silently
// (float_to_fp8_e4m3 clamps and returns, #1544). g_nvfp4_scale_clipped flags this at
// engine shutdown via nvfp4_report_scale_clipping(). A device printf here costs more
// than it looks: it gave five quantizer kernels an 8-byte local stack frame (make
// kernel-resources) because the call must be set up whether or not taken; a flag
// store leaves kernels at stack=0 for one float compare.
__device__ unsigned int g_nvfp4_scale_clipped = 0;

__device__ __forceinline__ void quantize_micro_block_nvfp4_from_vals(const float vals[kSFVecSize],
                                                                     float local_absmax,
                                                                     uint8_t* packed_out_row, int k_group,
                                                                     uint8_t* sfa_target) {
    // Encode UE4M3 scale (positive — `float_to_fp8_e4m3` handles clamp + rounding
    // and returns sign=0 for non-negative input, which is a valid UE4M3 byte).
    if (local_absmax > 2688.0f)
        g_nvfp4_scale_clipped = 1u;

    float scale_f = local_absmax / 6.0f;
    uint8_t ue4m3 = float_to_fp8_e4m3(scale_f);

    // Reconstruct actual scale from UE4M3 for consistent quantization. If the
    // scale rounds to zero, fall back to the smallest denorm (2^-9) to avoid
    // division by zero — matches the >=2^-9 clamp used elsewhere in imp.
    float actual_scale = fp8_e4m3_to_float_fast(ue4m3);
    if (actual_scale == 0.0f)
        actual_scale = 1.0f / 512.0f;
    float inv_scale = 1.0f / actual_scale;

    *sfa_target = ue4m3;

    uint8_t* packed_at = packed_out_row + (static_cast<int64_t>(k_group) * (kSFVecSize / 2));
#pragma unroll
    for (int i = 0; i < kSFVecSize; i += 2) {
        float s0 = vals[i] * inv_scale;
        float s1 = vals[i + 1] * inv_scale;
        packed_at[i / 2] = pack_fp4_pair_hw(s0, s1);
    }
}

// Direct FP16 quantize: load 16 FP16 values, pass to the above helper.
__device__ __forceinline__ void quantize_micro_block_nvfp4(const half* input_row_base, int k_group,
                                                           uint8_t* packed_out_row, uint8_t* sfa_target) {
    float vals[kSFVecSize];
    float local_absmax = 0.0f;
    const half2* src_h2 = reinterpret_cast<const half2*>(input_row_base +
                                                         (static_cast<int64_t>(k_group) * kSFVecSize));
    // #2218 bounded: i * 2 <= 14 (loop i < kSFVecSize / 2, kSFVecSize = 16 at :174)
#pragma unroll
    for (int i = 0; i < kSFVecSize / 2; i++) {
        half2 h2 = src_h2[i];
        vals[static_cast<ptrdiff_t>(i * 2)] = __half2float(h2.x);
        vals[(i * 2) + 1] = __half2float(h2.y);
        local_absmax = fmaxf(local_absmax,
                             fmaxf(fabsf(vals[static_cast<ptrdiff_t>(i * 2)]), fabsf(vals[(i * 2) + 1])));
    }
    quantize_micro_block_nvfp4_from_vals(vals, local_absmax, packed_out_row, k_group, sfa_target);
}

// Single-tensor quantize: row numbering is direct, SFA is a single linear buffer
// with SfAtom layout over (row, k_group).
__global__ void quantize_fp16_nvfp4_cutlass_kernel(
    const half* __restrict__ input,    // [M, K] FP16
    uint8_t* __restrict__ packed_out,  // [M, K/2] packed nibbles
    uint8_t* __restrict__ sf_out,      // SfAtom layout UE4M3
    int M, int K, int n_k_tiles) {
    int mb_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int K_groups = K / kSFVecSize;
    int total_mb = M * K_groups;
    if (mb_idx >= total_mb)
        return;

    int row = mb_idx / K_groups;
    int k_group = mb_idx % K_groups;

    quantize_micro_block_nvfp4(input + (static_cast<int64_t>(row) * K), k_group,
                               packed_out + (static_cast<int64_t>(row) * (K / 2)),
                               sf_out + sfatom_offset(row, k_group, n_k_tiles));
}

// Device helper: binary-search `offsets` for the expert owning `row`.
// Returns expert index and writes `local_row` (row relative to expert's slab).
__device__ __forceinline__ int moe_find_expert(const int* offsets, int ne, int row, int& local_row) {
    int lo = 0, hi = ne;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (offsets[mid] <= row)
            lo = mid;
        else
            hi = mid;
    }
    local_row = row - offsets[lo];
    return lo;
}

// MoE variant: one kernel quantizes all [expanded, K] rows into contiguous
// packed output + per-expert SFA slabs (one per expert).
__global__ void quantize_fp16_nvfp4_cutlass_moe_kernel(
    const half* __restrict__ input,          // [expanded, K] (gather=null) OR [n_tokens, K] (gather!=null)
    const int32_t* __restrict__ gather,      // [expanded] sorted_token_ids permutation, or null
    uint8_t* __restrict__ packed_out,        // [expanded, K/2] contiguous
    uint8_t* const* __restrict__ sfa_bases,  // [ne] per-expert SFA base (may be null)
    const int* __restrict__ offsets,         // [ne+1] cumulative row offsets
    int expanded, int K, int ne, int n_k_tiles) {
    int mb_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int K_groups = K / kSFVecSize;
    if (mb_idx >= expanded * K_groups)
        return;

    int row = mb_idx / K_groups;
    int k_group = mb_idx % K_groups;
    int local_row;
    int expert = moe_find_expert(offsets, ne, row, local_row);
    uint8_t* sfa = sfa_bases[expert];
    if (!sfa)
        return;

    // Fused-gather path: input is pre-permute MoE input in token order; gather[row]
    // indexes the source token for this expert-sorted row. Saves the gathered FP16
    // intermediate when the upstream moe_gather is skipped (gated on a lazy-gather
    // addition in the legacy MoE fallback, see docs/plans/moe_prefill_cudagraph_*.md).
    const int src_row = (gather != nullptr) ? gather[row] : row;
    quantize_micro_block_nvfp4(input + (static_cast<int64_t>(src_row) * K), k_group,
                               packed_out + (static_cast<int64_t>(row) * (K / 2)),
                               sfa + sfatom_offset(local_row, k_group, n_k_tiles));
}

// ---------------------------------------------------------------------------
// Host-callable functions
// ---------------------------------------------------------------------------

void convert_nvfp4_to_cutlass(const NvFP4QuantResult& src, CutlassNvFP4Weight& dst, cudaStream_t stream) {
    IMP_CHECK(src.packed_data != nullptr, "convert_nvfp4_to_cutlass: src.packed_data is null");
    int64_t N = src.N;
    int64_t K = src.K;

    // Data pointer is borrowed as RowMajor [N, K/2].
    // Despite LayoutBTag=ColumnMajor, CUTLASS block-scaled GEMM uses RowMajor stride.

    // Allocate SfAtom scale buffer
    size_t sf_bytes = cutlass_nvfp4_sf_size(static_cast<int>(N), static_cast<int>(K));
    void* d_sf = nullptr;
    cuda_alloc_or_throw(cudaMalloc(&d_sf, sf_bytes), "convert_nvfp4_to_cutlass scales");
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_sf, 0, sf_bytes, stream));  // zero-init for padding

    // Convert scales to SfAtom layout (micro_scale only, tensor_scale deferred to GEMM alpha)
    {
        int K_groups = static_cast<int>(K) / kSFVecSize;
        int total = static_cast<int>(N) * K_groups;
        int n_k_tiles = (static_cast<int>(K) + kAtomKElems - 1) / kAtomKElems;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        convert_scales_sfatom_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(src.micro_scales), reinterpret_cast<uint8_t*>(d_sf),
            static_cast<int>(N), static_cast<int>(K), n_k_tiles);
        IMP_CUDA_CHECK_LAUNCH();
    }

    dst.data = src.packed_data;  // borrowed pointer (not owned)
    dst.scale_factors = d_sf;
    dst.tensor_scale = src.tensor_scale;
    dst.N = N;
    dst.K = K;
    dst.sf_bytes = sf_bytes;

    IMP_LOG_DEBUG("convert_nvfp4_to_cutlass: N=%lld K=%lld sf=%.2f MiB (data borrowed)", (long long)N,
                  (long long)K, sf_bytes / (1024.0 * 1024.0));
}

void convert_nvfp4_to_cutlass_borrowed(const NvFP4QuantResult& src, CutlassNvFP4Weight& dst, void* sf_dst,
                                       cudaStream_t stream) {
    IMP_CHECK(src.packed_data != nullptr, "convert_nvfp4_to_cutlass_borrowed: src.packed_data is null");
    IMP_CHECK(sf_dst != nullptr, "convert_nvfp4_to_cutlass_borrowed: sf_dst is null");
    int64_t N = src.N;
    int64_t K = src.K;

    // Write micro-scales into the caller's pre-zeroed slab sub-region (no alloc,
    // no memset — the slab is zeroed once for all entries). Same kernel/layout
    // as convert_nvfp4_to_cutlass.
    int K_groups = static_cast<int>(K) / kSFVecSize;
    int total = static_cast<int>(N) * K_groups;
    int n_k_tiles = (static_cast<int>(K) + kAtomKElems - 1) / kAtomKElems;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    convert_scales_sfatom_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src.micro_scales), reinterpret_cast<uint8_t*>(sf_dst),
        static_cast<int>(N), static_cast<int>(K), n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();

    dst.data = src.packed_data;  // borrowed pointer (not owned)
    dst.scale_factors = sf_dst;  // borrowed slab sub-region
    dst.tensor_scale = src.tensor_scale;
    dst.N = N;
    dst.K = K;
    dst.sf_bytes = cutlass_nvfp4_sf_size(static_cast<int>(N), static_cast<int>(K));
    dst.sf_borrowed = true;  // slab owns the memory; skip per-tensor cudaFree
}

void free_cutlass_nvfp4_weight(CutlassNvFP4Weight& w) {
    // data is borrowed from NvFP4QuantResult — do NOT free it
    w.data = nullptr;
    if (w.scale_factors && !w.sf_borrowed) {
        IMP_CUDA_CHECK_LOG(cudaFree(w.scale_factors));
    }
    w.scale_factors = nullptr;
    w.sf_borrowed = false;
    w.N = w.K = 0;
    w.sf_bytes = 0;
}

void convert_nvfp4_moe_scales_to_sfatom(const void* src_native_ms, void* dst_sfatom_sf, int ne, int N, int K,
                                        cudaStream_t stream, const int32_t* expert_offsets) {
    IMP_CHECK(K % kSFVecSize == 0, "convert_nvfp4_moe_scales_to_sfatom: K=%d must be multiple of %d", K,
              kSFVecSize);
    int K_groups = K / kSFVecSize;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;
    size_t native_stride = static_cast<size_t>(N) * K_groups;
    size_t sfatom_stride = cutlass_nvfp4_sf_size(N, K);

    // Pre-zero so SfAtom row-tile padding bytes are well-defined (the kernel
    // writes only valid (n, k_group) positions; padding rows pad up to 128).
    // Without padding the kernel writes every byte of a converted expert.
    if (sfatom_stride != native_stride)
        IMP_CUDA_CHECK_LOG(
            cudaMemsetAsync(dst_sfatom_sf, 0, static_cast<size_t>(ne) * sfatom_stride, stream));

    int total = N * K_groups;
    int threads = 256;
    int blocks_x = (total + threads - 1) / threads;
    if (expert_offsets)  // most experts exit at once (prefill staging): few blocks each, grid-stride
        blocks_x = std::min(blocks_x, 16);
    dim3 grid(blocks_x, ne);
    convert_scales_sfatom_moe_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src_native_ms), reinterpret_cast<uint8_t*>(dst_sfatom_sf), N, K,
        n_k_tiles, native_stride, sfatom_stride, expert_offsets);
    IMP_CUDA_CHECK_LAUNCH();
}

void quantize_fp16_to_nvfp4_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream) {
    IMP_CHECK(K % kSFVecSize == 0, "quantize_fp16_to_nvfp4_cutlass: K=%d must be multiple of %d", K,
              kSFVecSize);

    // SfAtom padding bytes are pre-zeroed once at workspace allocation
    // (executor_workspace_buffers.cu). The kernel only writes valid (row,k_group) cells; padding
    // stays zero. Avoids a cudaMemsetAsync per call.

    int K_groups = K / kSFVecSize;
    int total_mb = M * K_groups;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;
    quantize_fp16_nvfp4_cutlass_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16), reinterpret_cast<uint8_t*>(dst_data),
        reinterpret_cast<uint8_t*>(dst_sf), M, K, n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();
}

void quantize_fp16_to_nvfp4_cutlass_moe(const void* src_fp16, void* dst_packed, uint8_t* const* d_sfa_bases,
                                        const int* d_offsets, int expanded, int K, int ne,
                                        cudaStream_t stream) {
    IMP_CHECK(K % kSFVecSize == 0, "quantize_fp16_to_nvfp4_cutlass_moe: K=%d must be multiple of %d", K,
              kSFVecSize);
    if (expanded == 0)
        return;

    int K_groups = K / kSFVecSize;
    int total_mb = expanded * K_groups;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;
    quantize_fp16_nvfp4_cutlass_moe_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16), /*gather=*/nullptr, reinterpret_cast<uint8_t*>(dst_packed),
        d_sfa_bases, d_offsets, expanded, K, ne, n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();
}

void quantize_fp16_to_nvfp4_cutlass_moe_gather(const void* src_fp16, const int32_t* sorted_token_ids,
                                               void* dst_packed, uint8_t* const* d_sfa_bases,
                                               const int* d_offsets, int expanded, int K, int ne,
                                               cudaStream_t stream) {
    IMP_CHECK(K % kSFVecSize == 0, "quantize_fp16_to_nvfp4_cutlass_moe_gather: K=%d must be multiple of %d",
              K, kSFVecSize);
    IMP_CHECK(sorted_token_ids != nullptr,
              "quantize_fp16_to_nvfp4_cutlass_moe_gather: sorted_token_ids must not be null");
    if (expanded == 0)
        return;

    int K_groups = K / kSFVecSize;
    int total_mb = expanded * K_groups;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;
    quantize_fp16_nvfp4_cutlass_moe_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16), sorted_token_ids, reinterpret_cast<uint8_t*>(dst_packed),
        d_sfa_bases, d_offsets, expanded, K, ne, n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();
}

// Fused activation + NVFP4 CUTLASS quantize (phase-5 review section 2.2, #604): reads gate+up
// from HBM, computes SwiGLU/GeGLU/ReLU^2 in registers, writes only packed FP4 + SFA. Replaces
// apply_expert_activation + quantize_..._moe, saving one HBM round-trip of the swiglu
// intermediate (never materialized in HBM).

// Compile-time activation tag: keeps the inner branch a no-op in PTX.
template <int kAct>
__global__ void fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel(
    const half* __restrict__ gate,           // [expanded, K] or nullptr when kAct == RELU_SQR
    const half* __restrict__ up,             // [expanded, K]
    uint8_t* __restrict__ packed_out,        // [expanded, K/2]
    uint8_t* const* __restrict__ sfa_bases,  // [ne]
    const int* __restrict__ offsets,         // [ne+1]
    int expanded, int K, int ne, int n_k_tiles,
    const half* __restrict__ gate_bias,  // [ne, K], read only when kAct == GPT_OSS_GLU
    const half* __restrict__ up_bias) {  // [ne, K], read only when kAct == GPT_OSS_GLU
    int mb_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int K_groups = K / kSFVecSize;
    if (mb_idx >= expanded * K_groups)
        return;

    int row = mb_idx / K_groups;
    int k_group = mb_idx % K_groups;
    int local_row;
    int expert = moe_find_expert(offsets, ne, row, local_row);
    uint8_t* sfa = sfa_bases[expert];
    if (!sfa)
        return;

    // Compute 16 activation values + their absmax in registers.
    float vals[kSFVecSize];
    float local_absmax = 0.0f;
    const int64_t row_off = (static_cast<int64_t>(row) * K) + (static_cast<int64_t>(k_group) * kSFVecSize);
    const half2* up_h2 = reinterpret_cast<const half2*>(up + row_off);
    const half2* gate_h2 = (gate != nullptr) ? reinterpret_cast<const half2*>(gate + row_off) : nullptr;
    const int64_t bias_off = (static_cast<int64_t>(expert) * K) +
                             (static_cast<int64_t>(k_group) * kSFVecSize);

    constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
    constexpr float kGeluCoeff = 0.044715f;

#pragma unroll
    for (int i = 0; i < kSFVecSize / 2; i++) {
        half2 uh2 = up_h2[i];
        float u0 = __half2float(uh2.x);
        // #2218 bounded: i * 2 <= 14 (loop i < kSFVecSize / 2, kSFVecSize = 16 at :174)
        float u1 = __half2float(uh2.y);
        float v0, v1;
        if (kAct == 0) {  // SWIGLU
            half2 gh2 = gate_h2[i];
            float g0 = __half2float(gh2.x);
            float g1 = __half2float(gh2.y);
            v0 = (g0 / (1.0f + __expf(-g0))) * u0;
            v1 = (g1 / (1.0f + __expf(-g1))) * u1;
        } else if (kAct == 1) {  // GEGLU (Gemma-3 tanh form, FP16-clamped)
            half2 gh2 = gate_h2[i];
            float g0 = __half2float(gh2.x);
            float g1 = __half2float(gh2.y);
            float gelu0 = g0 * 0.5f * (1.0f + tanhf(kGeluSqrt2OverPi * (g0 + (kGeluCoeff * g0 * g0 * g0))));
            float gelu1 = g1 * 0.5f * (1.0f + tanhf(kGeluSqrt2OverPi * (g1 + (kGeluCoeff * g1 * g1 * g1))));
            v0 = fminf(fmaxf(gelu0 * u0, -65504.0f), 65504.0f);
            v1 = fminf(fmaxf(gelu1 * u1, -65504.0f), 65504.0f);
        } else if (kAct == 3) {  // GPT_OSS_GLU: FP16 bias add + FP16 GLU output, as the 4-launch path
            const half2 g = __hadd2(gate_h2[i], reinterpret_cast<const half2*>(gate_bias + bias_off)[i]);
            const half2 u = __hadd2(uh2, reinterpret_cast<const half2*>(up_bias + bias_off)[i]);
            v0 = __half2float(__float2half_rn(gpt_oss_glu_elem(__half2float(g.x), __half2float(u.x))));
            v1 = __half2float(__float2half_rn(gpt_oss_glu_elem(__half2float(g.y), __half2float(u.y))));
        } else {  // RELU_SQR — non_gated experts; gate is nullptr, up holds the input
            v0 = (u0 > 0.0f) ? (u0 * u0) : 0.0f;
            v1 = (u1 > 0.0f) ? (u1 * u1) : 0.0f;
        }
        vals[static_cast<ptrdiff_t>(i * 2)] = v0;
        vals[(i * 2) + 1] = v1;
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(v0), fabsf(v1)));
    }

    quantize_micro_block_nvfp4_from_vals(vals, local_absmax,
                                         packed_out + (static_cast<int64_t>(row) * (K / 2)), k_group,
                                         sfa + sfatom_offset(local_row, k_group, n_k_tiles));
}

void fused_act_quantize_fp16_to_nvfp4_cutlass_moe(const void* gate_fp16, const void* up_fp16,
                                                  void* dst_packed, uint8_t* const* d_sfa_bases,
                                                  const int* d_offsets, int expanded, int K, int ne,
                                                  FFNActivation act_type, cudaStream_t stream,
                                                  const void* gate_bias_fp16, const void* up_bias_fp16) {
    IMP_CHECK(K % kSFVecSize == 0,
              "fused_act_quantize_fp16_to_nvfp4_cutlass_moe: K=%d must be multiple of %d", K, kSFVecSize);
    const bool relu = act_type == FFNActivation::RELU_SQR;
    const bool gpt_oss = act_type == FFNActivation::GPT_OSS_GLU;
    IMP_CHECK(relu || gate_fp16 != nullptr,
              "fused_act_quantize: gated activation requires a non-null gate tensor");
    IMP_CHECK(!gpt_oss || (gate_bias_fp16 && up_bias_fp16),
              "fused_act_quantize: GPT_OSS_GLU needs both biases");
    if (expanded == 0)
        return;

    const int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;
    const int threads = 256;
    const int blocks = ((expanded * (K / kSFVecSize)) + threads - 1) / threads;
    auto* kernel = act_type == FFNActivation::SWIGLU  ? fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<0>
                   : act_type == FFNActivation::GEGLU ? fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<1>
                   : gpt_oss                          ? fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<3>
                                                      : fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<2>;
    kernel<<<blocks, threads, 0, stream>>>(relu ? nullptr : static_cast<const half*>(gate_fp16),
                                           static_cast<const half*>(up_fp16),
                                           static_cast<uint8_t*>(dst_packed), d_sfa_bases, d_offsets,
                                           expanded, K, ne, n_k_tiles,
                                           static_cast<const half*>(gate_bias_fp16),
                                           static_cast<const half*>(up_bias_fp16));
    IMP_CUDA_CHECK_LAUNCH();
}

// Dense SwiGLU + CUTLASS NVFP4 quantize: one thread per 16-value micro-block, FP16 math of
// swiglu_fp16_kernel, quantized from the stored FP16 values (= quantize_fp16_nvfp4_cutlass_kernel).
__global__ void swiglu_quantize_fp16_nvfp4_cutlass_kernel(const half* __restrict__ gate,
                                                          const half* __restrict__ up, half* __restrict__ out,
                                                          uint8_t* __restrict__ packed_out,
                                                          uint8_t* __restrict__ sf_out, int M, int K,
                                                          int n_k_tiles) {
    const int k_groups = K / kSFVecSize;
    const int64_t mb = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (mb >= static_cast<int64_t>(M) * k_groups)
        return;
    const int row = static_cast<int>(mb / k_groups);
    const int k_group = static_cast<int>(mb % k_groups);
    const int64_t base = mb * kSFVecSize;
    float vals[kSFVecSize];
    float amax = 0.0f;
    float4 ov[2];
#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const float4 gv = reinterpret_cast<const float4*>(gate + base)[v];
        const float4 uv = reinterpret_cast<const float4*>(up + base)[v];
        const half2* g2 = reinterpret_cast<const half2*>(&gv);
        const half2* u2 = reinterpret_cast<const half2*>(&uv);
        half2* o2 = reinterpret_cast<half2*>(&ov[v]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 gf = __half22float2(g2[k]);
            const float2 uf = __half22float2(u2[k]);
            o2[k] = __float22half2_rn(
                make_float2(gf.x / (1.0f + __expf(-gf.x)) * uf.x, gf.y / (1.0f + __expf(-gf.y)) * uf.y));
            const float2 rf = __half22float2(o2[k]);
            vals[(v * 8) + (k * 2)] = rf.x;
            vals[(v * 8) + (k * 2) + 1] = rf.y;
            amax = fmaxf(amax, fmaxf(fabsf(rf.x), fabsf(rf.y)));
        }
    }
    reinterpret_cast<float4*>(out + base)[0] = ov[0];
    reinterpret_cast<float4*>(out + base)[1] = ov[1];
    quantize_micro_block_nvfp4_from_vals(vals, amax, packed_out + (static_cast<int64_t>(row) * (K / 2)),
                                         k_group, sf_out + sfatom_offset(row, k_group, n_k_tiles));
}

void swiglu_quantize_fp16_to_nvfp4_cutlass(const void* gate_fp16, const void* up_fp16, void* out_fp16,
                                           void* dst_data, void* dst_sf, int M, int K, cudaStream_t stream) {
    IMP_CHECK(K % kSFVecSize == 0, "swiglu_quantize_fp16_to_nvfp4_cutlass: K=%d must be multiple of %d", K,
              kSFVecSize);
    const int threads = 256;
    const int blocks = static_cast<int>(((static_cast<int64_t>(M) * (K / kSFVecSize)) + threads - 1) /
                                        threads);
    swiglu_quantize_fp16_nvfp4_cutlass_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(gate_fp16), static_cast<const half*>(up_fp16), static_cast<half*>(out_fp16),
        static_cast<uint8_t*>(dst_data), static_cast<uint8_t*>(dst_sf), M, K,
        (K + kAtomKElems - 1) / kAtomKElems);
    IMP_CUDA_CHECK_LAUNCH();
}

bool cutlass_sm120_nvfp4_available() { return true; }

// Reads the clip flag once, from gemm_cleanup() (engine shutdown): a "should never happen"
// event, checked there to keep it off the hot path. cudaMemcpyFromSymbol synchronises (#1544).
void nvfp4_report_scale_clipping() {
    unsigned int clipped = 0;
    if (cudaMemcpyFromSymbol(&clipped, g_nvfp4_scale_clipped, sizeof(clipped)) != cudaSuccess || clipped == 0)
        return;
    IMP_LOG_WARN(
        "[NVFP4] at least one activation micro-block had absmax above the UE4M3 scale ceiling "
        "(448 * 6 = 2688) during this run: its block scale was clipped, so those values were "
        "quantised against a scale that is too small (#1544). Measured headroom when this check "
        "was added: Gemma-4-12B reached 2468, 92%% of the ceiling.");
}

}  // namespace imp
