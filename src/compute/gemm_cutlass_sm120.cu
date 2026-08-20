// CUTLASS sm_120 block-scaled NVFP4×NVFP4 GEMM for prefill acceleration.
//
// Uses CUTLASS 4.4.1 Example 79a pattern: Warp-Specialized persistent kernel
// with block-scaled tensor core MMA (mma.sync.aligned.block_scale) on
// Blackwell GeForce (sm_120).
//
// Both A (activation) and B (weight) use nv_float4_t<float_e2m1_t> with
// float_ue4m3_t unsigned scale factors in SfAtom interleaved layout.
// Output D is FP16 (cutlass::half_t) for direct use in the inference pipeline.
//
// Weight format conversion (once at init):
//   - Borrow packed FP4 pointer [N, K/2] (K-contiguous RowMajor)
//   - Convert micro_scales from linear [N, K/16] to SfAtom UE4M3 layout
//   - tensor_scale is NOT absorbed into scale factors (to avoid UE4M3
//     denormalized range precision loss); instead applied as GEMM alpha
//
// Activation quantization (per-prefill-call):
//   - FP16 [M, K] → NVFP4 packed [M, K/2] + SfAtom UE4M3 scales
//

#include "compute/gemm_cutlass_sm120.h"
#include "quant/nvfp4_quant.h"
#include "quant/fp8_utils.cuh"
#include "core/cuda_static_reset.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ---------------------------------------------------------------------------
// CUTLASS GEMM type configuration: NVFP4 × NVFP4 → FP16
// Based on Example 79a but with half_t output instead of bfloat16_t.
// ---------------------------------------------------------------------------

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD = cutlass::half_t;  // FP16 output
using ElementC = cutlass::half_t;  // C matrix type (unused, beta=0)
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;  // GeForce = no multicast

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag,
    AlignmentD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                        CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ---------------------------------------------------------------------------
// FP32-output variant (large-N cooperative tile only). The LM head writes FP32
// logits (the samplers read `const float*`), so the batched-decode LM head GEMM
// (N = vocab » kSmallNThreshold) needs a float epilogue rather than the half_t
// output above. ElementC = ElementD = float keeps the (unused, beta=0) C and D
// pointers the same type so the shared impl reinterprets one buffer for both.
// ---------------------------------------------------------------------------
using ElementDFp32 = float;
using ElementCFp32 = float;
constexpr int AlignmentDFp32 = 128 / cutlass::sizeof_bits<ElementDFp32>::value;  // 4
constexpr int AlignmentCFp32 = 128 / cutlass::sizeof_bits<ElementCFp32>::value;  // 4

using CollectiveEpilogueFp32 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator, ElementCFp32, LayoutCTag, AlignmentCFp32, ElementDFp32,
    LayoutDTag, AlignmentDFp32, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloopFp32 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogueFp32::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernelFp32 = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                           CollectiveMainloopFp32, CollectiveEpilogueFp32,
                                                           void>;
using GemmFp32 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelFp32>;

// Strides / SF layouts are derived per-variant inside the templated impl
// below (typename GemmT::GemmKernel::StrideA etc.) — both variants share the
// same SfAtom data layout, only the CTA tiling differs.

// Verify SFVecSize matches our constant (kSFVecSize = 16)
static_assert(Gemm::GemmKernel::CollectiveMainloop::TiledMma::Traits::SFVecSize == 16,
              "CUTLASS SFVecSize mismatch — expected 16 for nv_float4_t");

// ---------------------------------------------------------------------------
// Small-N variant: pingpong schedule + 128x64x128 tile.
// The default cooperative 128x128 tile starves the GPU on small-N GEMMs
// (kv_proj N=1024 at M=512: 32 CTAs on 170 SMs). Pingpong with a 64-wide
// N-tile doubles the CTA count and overlaps two consumer warpgroups:
// measured 2026-06-07 (standalone config sweep on Qwen3-14B shapes, #596)
// 2.1x on 512x1024x5120 (27.0us -> 12.8us). It LOSES ~25% on large-N
// shapes, hence the dispatch threshold below. The SfAtom scale layout is
// tile-shape-independent (128-row x 4-group atoms from SFVecSize=16), so
// both variants consume identical A/B scale buffers.
// ---------------------------------------------------------------------------
using ThreadBlockShapeSmallN = Shape<_128, _64, _128>;

using CollectiveEpilogueSmallN = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShapeSmallN, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, ElementC,
    LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloopSmallN = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShapeSmallN, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogueSmallN::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernelSmallN = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                              CollectiveMainloopSmallN,
                                                              CollectiveEpilogueSmallN, void>;
using GemmSmallN = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelSmallN>;

static_assert(GemmSmallN::GemmKernel::CollectiveMainloop::TiledMma::Traits::SFVecSize == 16,
              "CUTLASS SFVecSize mismatch (small-N variant)");

// N at/below this routes to the small-N pingpong kernel. Measured: N=1024
// is 2.1x faster, N=5120 is ~25% slower — the crossover lies between;
// 2048 is the conservative cut.
static constexpr int kSmallNThreshold = 2048;

namespace imp {

// ---------------------------------------------------------------------------
// SfAtom layout computation (hardware-independent arithmetic)
// ---------------------------------------------------------------------------
// SfAtom for K-major, SFVecSize=16:
//   Shape:  ((32, 4), (16, 4))
//   Stride: ((16, 4), ( 0, 1))
//
// Each atom covers 128 rows × 4 scale-groups (= 64 data elements in K).
// Atom size = 128 * 4 = 512 bytes.
//
// tile_to_shape tiles atoms to cover (rows, K) with Step<_2, _1>:
//   K dimension tiles are inner (faster-changing), row tiles are outer.

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

    int atom_offset = n0 * 16 + n1 * 4 + k_local;
    int tile_base = (tile_row * n_k_tiles + tile_k) * kAtomSize;
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

// Convert micro_scales from linear layout to SfAtom layout (NO tensor_scale absorption).
// tensor_scale is deferred to the GEMM epilogue alpha parameter for precision.
// Source: [N, K/16] FP8 E4M3 (signed, but always positive for scale factors)
// Dest:   SfAtom layout UE4M3 (unsigned, just micro_scale — NOT combined)
__global__ void convert_scales_sfatom_kernel(const uint8_t* __restrict__ src_ms,  // [N, K/16] linear
                                             uint8_t* __restrict__ dst_sf,        // SfAtom layout
                                             int N, int K, int n_k_tiles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kSFVecSize;
    int total = N * K_groups;
    if (idx >= total)
        return;

    int n = idx / K_groups;
    int k_group = idx % K_groups;

    // Read signed E4M3 micro-scale, drop its sign (always positive for scales),
    // then re-encode as UE4M3 via the shared float↔E4M3 helper. UE4M3 is
    // bit-identical to positive E4M3, so float_to_fp8_e4m3 with a positive
    // argument yields the UE4M3 byte directly (sign bit = 0).
    float combined = fabsf(fp8_e4m3_to_float_fast(src_ms[idx]));
    dst_sf[sfatom_offset(n, k_group, n_k_tiles)] = float_to_fp8_e4m3(combined);
}

// MoE variant: one launch converts SF for all `ne` experts. Source has stride
// N*K_groups bytes per expert; destination has stride cutlass_nvfp4_sf_size(N,K)
// bytes per expert. blockIdx.y selects the expert; the inner work is identical
// to the single-tensor kernel above.
__global__ void convert_scales_sfatom_moe_kernel(const uint8_t* __restrict__ src_ms,
                                                 uint8_t* __restrict__ dst_sf, int N, int K,
                                                 int n_k_tiles, size_t native_stride_per_expert,
                                                 size_t sfatom_stride_per_expert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kSFVecSize;
    int total = N * K_groups;
    if (idx >= total)
        return;

    int e = blockIdx.y;
    const uint8_t* src_e = src_ms + static_cast<size_t>(e) * native_stride_per_expert;
    uint8_t* dst_e = dst_sf + static_cast<size_t>(e) * sfatom_stride_per_expert;

    int n = idx / K_groups;
    int k_group = idx % K_groups;

    float combined = fabsf(fp8_e4m3_to_float_fast(src_e[idx]));
    dst_e[sfatom_offset(n, k_group, n_k_tiles)] = float_to_fp8_e4m3(combined);
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

// Given 16 pre-computed float values + their absmax, encode UE4M3 scale and
// pack FP4 bytes. The caller supplies the values (so this helper is reusable
// for fused paths like SwiGLU+quantize where values come from a computation
// rather than a direct FP16 load).
__device__ __forceinline__ void quantize_micro_block_nvfp4_from_vals(const float vals[kSFVecSize],
                                                                     float local_absmax,
                                                                     uint8_t* packed_out_row, int k_group,
                                                                     uint8_t* sfa_target) {
    // Encode UE4M3 scale (positive — `float_to_fp8_e4m3` handles clamp + rounding
    // and returns sign=0 for non-negative input, which is a valid UE4M3 byte).
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

    uint8_t* packed_at = packed_out_row + k_group * (kSFVecSize / 2);
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
    const half2* src_h2 = reinterpret_cast<const half2*>(input_row_base + k_group * kSFVecSize);
#pragma unroll
    for (int i = 0; i < kSFVecSize / 2; i++) {
        half2 h2 = src_h2[i];
        vals[i * 2] = __half2float(h2.x);
        vals[i * 2 + 1] = __half2float(h2.y);
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(vals[i * 2]), fabsf(vals[i * 2 + 1])));
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
    int mb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kSFVecSize;
    int total_mb = M * K_groups;
    if (mb_idx >= total_mb)
        return;

    int row = mb_idx / K_groups;
    int k_group = mb_idx % K_groups;

    quantize_micro_block_nvfp4(input + static_cast<int64_t>(row) * K, k_group,
                               packed_out + static_cast<int64_t>(row) * (K / 2),
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
    int mb_idx = blockIdx.x * blockDim.x + threadIdx.x;
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

    // Fused-gather path: input is the pre-permute MoE input in token order,
    // and `gather[row]` indexes the source token for this expert-sorted row.
    // Saves the gathered FP16 intermediate when the upstream moe_gather is
    // skipped — that skip is gated on a lazy-gather addition in the legacy
    // MoE fallback path (see plan in docs/plans/moe_prefill_cudagraph_*.md).
    const int src_row = (gather != nullptr) ? gather[row] : row;
    quantize_micro_block_nvfp4(input + static_cast<int64_t>(src_row) * K, k_group,
                               packed_out + static_cast<int64_t>(row) * (K / 2),
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
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_sf, sf_bytes));
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

void convert_nvfp4_moe_scales_to_sfatom(const void* src_native_ms, void* dst_sfatom_sf, int ne, int N,
                                        int K, cudaStream_t stream) {
    IMP_CHECK(K % kSFVecSize == 0, "convert_nvfp4_moe_scales_to_sfatom: K=%d must be multiple of %d",
              K, kSFVecSize);
    int K_groups = K / kSFVecSize;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;
    size_t native_stride = static_cast<size_t>(N) * K_groups;
    size_t sfatom_stride = cutlass_nvfp4_sf_size(N, K);

    // Pre-zero so SfAtom row-tile padding bytes are well-defined (the kernel
    // writes only valid (n, k_group) positions; padding rows pad up to 128).
    IMP_CUDA_CHECK_LOG(
        cudaMemsetAsync(dst_sfatom_sf, 0, static_cast<size_t>(ne) * sfatom_stride, stream));

    int total = N * K_groups;
    int threads = 256;
    int blocks_x = (total + threads - 1) / threads;
    dim3 grid(blocks_x, ne);
    convert_scales_sfatom_moe_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src_native_ms), reinterpret_cast<uint8_t*>(dst_sfatom_sf), N, K,
        n_k_tiles, native_stride, sfatom_stride);
    IMP_CUDA_CHECK_LAUNCH();
}

void quantize_fp16_to_nvfp4_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream) {
    IMP_CHECK(K % kSFVecSize == 0, "quantize_fp16_to_nvfp4_cutlass: K=%d must be multiple of %d",
              K, kSFVecSize);

    // SfAtom padding bytes are pre-zeroed once at workspace allocation
    // (executor_workspace_buffers.cu). The kernel only writes valid (row, k_group)
    // cells; padding stays zero. Avoids a cudaMemsetAsync per call (~6720 in
    // Llama Q8 W1 prefill).

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
    IMP_CHECK(K % kSFVecSize == 0, "quantize_fp16_to_nvfp4_cutlass_moe: K=%d must be multiple of %d",
              K, kSFVecSize);
    if (expanded == 0)
        return;

    int K_groups = K / kSFVecSize;
    int total_mb = expanded * K_groups;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;
    quantize_fp16_nvfp4_cutlass_moe_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16), /*gather=*/nullptr,
        reinterpret_cast<uint8_t*>(dst_packed), d_sfa_bases,
        d_offsets, expanded, K, ne, n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();
}

void quantize_fp16_to_nvfp4_cutlass_moe_gather(const void* src_fp16,
                                               const int32_t* sorted_token_ids,
                                               void* dst_packed,
                                               uint8_t* const* d_sfa_bases,
                                               const int* d_offsets, int expanded, int K, int ne,
                                               cudaStream_t stream) {
    IMP_CHECK(K % kSFVecSize == 0,
              "quantize_fp16_to_nvfp4_cutlass_moe_gather: K=%d must be multiple of %d", K, kSFVecSize);
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
        reinterpret_cast<const half*>(src_fp16), sorted_token_ids,
        reinterpret_cast<uint8_t*>(dst_packed), d_sfa_bases,
        d_offsets, expanded, K, ne, n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Fused activation + NVFP4 CUTLASS quantize — M1 from the phase-5 review §2.2
// (archived in #604).
// Reads gate + up from HBM, computes SwiGLU/GeGLU/ReLU² in registers, and writes
// only the packed FP4 + SFA. Replaces the apply_expert_activation + quantize_..._moe
// pair in the device-args MoE prefill path; saves one full HBM round-trip of the
// swiglu intermediate (the activation tensor is never materialized in HBM).
// ---------------------------------------------------------------------------

// Compile-time activation tag: keeps the inner branch a no-op in PTX.
template <int kAct>
__global__ void fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel(
    const half* __restrict__ gate,           // [expanded, K] or nullptr when kAct == RELU_SQR
    const half* __restrict__ up,             // [expanded, K]
    uint8_t* __restrict__ packed_out,        // [expanded, K/2]
    uint8_t* const* __restrict__ sfa_bases,  // [ne]
    const int* __restrict__ offsets,         // [ne+1]
    int expanded, int K, int ne, int n_k_tiles) {
    int mb_idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int64_t row_off = static_cast<int64_t>(row) * K + k_group * kSFVecSize;
    const half2* up_h2 = reinterpret_cast<const half2*>(up + row_off);
    const half2* gate_h2 = (gate != nullptr) ? reinterpret_cast<const half2*>(gate + row_off) : nullptr;

    constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
    constexpr float kGeluCoeff = 0.044715f;

#pragma unroll
    for (int i = 0; i < kSFVecSize / 2; i++) {
        half2 uh2 = up_h2[i];
        float u0 = __half2float(uh2.x);
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
            float gelu0 = g0 * 0.5f *
                          (1.0f + tanhf(kGeluSqrt2OverPi * (g0 + kGeluCoeff * g0 * g0 * g0)));
            float gelu1 = g1 * 0.5f *
                          (1.0f + tanhf(kGeluSqrt2OverPi * (g1 + kGeluCoeff * g1 * g1 * g1)));
            v0 = fminf(fmaxf(gelu0 * u0, -65504.0f), 65504.0f);
            v1 = fminf(fmaxf(gelu1 * u1, -65504.0f), 65504.0f);
        } else {  // RELU_SQR — non_gated experts; gate is nullptr, up holds the input
            v0 = (u0 > 0.0f) ? (u0 * u0) : 0.0f;
            v1 = (u1 > 0.0f) ? (u1 * u1) : 0.0f;
        }
        vals[i * 2] = v0;
        vals[i * 2 + 1] = v1;
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(v0), fabsf(v1)));
    }

    quantize_micro_block_nvfp4_from_vals(
        vals, local_absmax,
        packed_out + static_cast<int64_t>(row) * (K / 2), k_group,
        sfa + sfatom_offset(local_row, k_group, n_k_tiles));
}

void fused_act_quantize_fp16_to_nvfp4_cutlass_moe(const void* gate_fp16, const void* up_fp16,
                                                  void* dst_packed, uint8_t* const* d_sfa_bases,
                                                  const int* d_offsets, int expanded, int K, int ne,
                                                  FFNActivation act_type, cudaStream_t stream) {
    IMP_CHECK(K % kSFVecSize == 0,
              "fused_act_quantize_fp16_to_nvfp4_cutlass_moe: K=%d must be multiple of %d", K, kSFVecSize);
    if (expanded == 0)
        return;

    int K_groups = K / kSFVecSize;
    int total_mb = expanded * K_groups;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;

    auto* gate_p = reinterpret_cast<const half*>(gate_fp16);
    auto* up_p = reinterpret_cast<const half*>(up_fp16);
    auto* dst_p = reinterpret_cast<uint8_t*>(dst_packed);

    switch (act_type) {
        case FFNActivation::SWIGLU:
            IMP_CHECK(gate_p != nullptr,
                      "fused_act_quantize: SWIGLU requires a non-null gate tensor");
            fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<0>
                <<<blocks, threads, 0, stream>>>(gate_p, up_p, dst_p, d_sfa_bases, d_offsets, expanded, K,
                                                 ne, n_k_tiles);
            IMP_CUDA_CHECK_LAUNCH();
            break;
        case FFNActivation::GEGLU:
            IMP_CHECK(gate_p != nullptr,
                      "fused_act_quantize: GEGLU requires a non-null gate tensor");
            fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<1>
                <<<blocks, threads, 0, stream>>>(gate_p, up_p, dst_p, d_sfa_bases, d_offsets, expanded, K,
                                                 ne, n_k_tiles);
            IMP_CUDA_CHECK_LAUNCH();
            break;
        case FFNActivation::RELU_SQR:
            fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<2>
                <<<blocks, threads, 0, stream>>>(nullptr, up_p, dst_p, d_sfa_bases, d_offsets, expanded, K,
                                                 ne, n_k_tiles);
            IMP_CUDA_CHECK_LAUNCH();
            break;
    }
}

// ---------------------------------------------------------------------------
// CUTLASS GEMM execution
// ---------------------------------------------------------------------------

template <class GemmT>
static size_t cutlass_workspace_for(int M, int N, int K) {
    using GK = typename GemmT::GemmKernel;
    auto stride_A = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {M, N, 1});

    using BlkCfg = typename GK::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = BlkCfg::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = BlkCfg::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    typename GemmT::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                   {M, N, K, 1},
                                   {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr,
                                    layout_SFB},
                                   {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}};

    return GemmT::get_workspace_size(args);
}

size_t gemm_nvfp4_cutlass_sm120_workspace(int M, int N, int K) {
    return (N <= kSmallNThreshold) ? cutlass_workspace_for<GemmSmallN>(M, N, K)
                                   : cutlass_workspace_for<Gemm>(M, N, K);
}

template <class GemmT, class ElemD = ElementD>
static bool gemm_nvfp4_cutlass_sm120_impl(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b,
                                          void* d_fp16, int M, int N, int K, void* workspace,
                                          size_t workspace_size, cudaStream_t stream) {
    // Flush any prior async errors — a sticky CUDA error will make
    // cuTensorMapEncodeTiled return 719 (LAUNCH_FAILED) instead of the real code.
    {
        cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            IMP_LOG_ERROR("CUTLASS sm120: prior CUDA error before GEMM: %s", cudaGetErrorString(prior));
            return false;
        }
    }

    using GK = typename GemmT::GemmKernel;
    auto stride_A = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {M, N, 1});

    using BlkCfg = typename GK::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = BlkCfg::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = BlkCfg::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    auto* a_ptr = reinterpret_cast<const ElementA::DataType*>(a_data);
    auto* b_ptr = reinterpret_cast<const ElementB::DataType*>(b.data);
    auto* sfa_ptr = reinterpret_cast<const ElementA::ScaleFactorType*>(a_sf);
    auto* sfb_ptr = reinterpret_cast<const ElementB::ScaleFactorType*>(b.scale_factors);

    // C pointer must be valid even with beta=0 — CUTLASS creates a TMA
    // descriptor for C during initialize() and cuTensorMapEncodeTiled
    // fails on nullptr.  Re-use the D buffer since it's never read.
    auto* d_ptr = reinterpret_cast<ElemD*>(d_fp16);

    // Use tensor_scale as alpha: compensates for not absorbing it into SFB.
    // D = tensor_scale * (A_fp4 * SFA * B_fp4 * micro_scale_only) = correct result.
    float alpha = b.tensor_scale;

    typename GemmT::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                   {M, N, K, 1},
                                   {a_ptr, stride_A, b_ptr, stride_B, sfa_ptr, layout_SFA, sfb_ptr,
                                    layout_SFB},
                                   {{alpha, 0.0f},
                                    d_ptr,
                                    stride_C,  // C = D buffer (beta=0, never read)
                                    d_ptr,
                                    stride_D}};

    GemmT gemm;
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_WARN("CUTLASS sm120 NVFP4 GEMM: can_implement failed (%d) for M=%d N=%d K=%d", (int)st, M, N,
                     K);
        return false;
    }

    // The caller's workspace is the whole workspace. A7 step 8 deleted the
    // cudaFree+cudaMalloc grow path that used to sit here: it ran at GEMM time,
    // on a code path reachable under CUDA-graph capture (where cudaMalloc is
    // illegal), to serve a case every in-tree caller already sizes against —
    // each one asks gemm_nvfp4_cutlass_sm120_workspace() for the same (or a
    // larger) shape and passes the answer. Refusing lets the dispatch fall back
    // to the dequant path with correct output; allocating here could not
    // (docs/internals/MEMORY.md A5.3).
    size_t needed = GemmT::get_workspace_size(args);
    if (needed > workspace_size) {
        IMP_LOG_WARN(
            "CUTLASS sm120 NVFP4 GEMM: workspace %zu B < %zu B needed for M=%d N=%d K=%d "
            "— refusing, the caller falls back",
            workspace_size, needed, M, N, K);
        return false;
    }

    st = gemm.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS sm120 NVFP4 GEMM: initialize failed (%d) M=%d N=%d K=%d", (int)st, M, N, K);
        return false;
    }

    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS sm120 NVFP4 GEMM: run failed (%d)", (int)st);
        return false;
    }

    return true;
}

bool gemm_nvfp4_cutlass_sm120(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b, void* d_fp16,
                              int M, int N, int K, void* workspace, size_t workspace_size,
                              cudaStream_t stream) {
    if (N <= kSmallNThreshold)
        return gemm_nvfp4_cutlass_sm120_impl<GemmSmallN>(a_data, a_sf, b, d_fp16, M, N, K, workspace,
                                                         workspace_size, stream);
    return gemm_nvfp4_cutlass_sm120_impl<Gemm>(a_data, a_sf, b, d_fp16, M, N, K, workspace, workspace_size,
                                               stream);
}

// FP32-output entry — large-N cooperative tile only (LM head: N = vocab » 2048).
bool gemm_nvfp4_cutlass_sm120_fp32(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b,
                                   void* d_fp32, int M, int N, int K, void* workspace,
                                   size_t workspace_size, cudaStream_t stream) {
    return gemm_nvfp4_cutlass_sm120_impl<GemmFp32, ElementDFp32>(a_data, a_sf, b, d_fp32, M, N, K, workspace,
                                                                workspace_size, stream);
}

size_t gemm_nvfp4_cutlass_sm120_fp32_workspace(int M, int N, int K) {
    return cutlass_workspace_for<GemmFp32>(M, N, K);
}

bool cutlass_sm120_nvfp4_available() { return true; }

}  // namespace imp
