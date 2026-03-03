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
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>

// CUTLASS headers — only included under IMP_USE_CUTLASS
#ifdef IMP_USE_CUTLASS

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

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

// ---------------------------------------------------------------------------
// CUTLASS GEMM type configuration: NVFP4 × NVFP4 → FP16
// Based on Example 79a but with half_t output instead of bfloat16_t.
// ---------------------------------------------------------------------------

using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD    = cutlass::half_t;           // FP16 output
using ElementC    = cutlass::half_t;           // C matrix type (unused, beta=0)
using LayoutCTag  = cutlass::layout::RowMajor;
using LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;   // 8
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;   // 8

using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;     // GeForce = no multicast

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Verify SFVecSize matches our constant (kSFVecSize = 16)
static_assert(Gemm::GemmKernel::CollectiveMainloop::TiledMma::Traits::SFVecSize == 16,
              "CUTLASS SFVecSize mismatch — expected 16 for nv_float4_t");

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED
#endif // IMP_USE_CUTLASS

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
static constexpr int kAtomRows = 128;     // 32 * 4
static constexpr int kAtomKGroups = 4;    // 4 scale groups per atom
static constexpr int kAtomKElems = kSFVecSize * kAtomKGroups;  // 64
static constexpr int kAtomSize = kAtomRows * kAtomKGroups;     // 512

// Compute SfAtom offset for logical scale factor at (row, k_group).
__device__ __host__ __forceinline__
int sfatom_offset(int row, int k_group, int n_k_tiles) {
    int tile_row = row / kAtomRows;
    int tile_k   = k_group / kAtomKGroups;
    int row_local = row % kAtomRows;
    int k_local   = k_group % kAtomKGroups;

    int n0 = row_local % 32;  // within 32-row sub-block
    int n1 = row_local / 32;  // which of 4 sub-blocks

    int atom_offset = n0 * 16 + n1 * 4 + k_local;
    int tile_base   = (tile_row * n_k_tiles + tile_k) * kAtomSize;
    return tile_base + atom_offset;
}

size_t cutlass_nvfp4_sf_size(int rows, int K) {
    int n_row_tiles = (rows + kAtomRows - 1) / kAtomRows;
    int n_k_tiles   = (K + kAtomKElems - 1) / kAtomKElems;
    return static_cast<size_t>(n_row_tiles) * n_k_tiles * kAtomSize;
}

// ---------------------------------------------------------------------------
// GPU kernels for weight conversion
// ---------------------------------------------------------------------------

// Convert micro_scales from linear layout to SfAtom layout (NO tensor_scale absorption).
// tensor_scale is deferred to the GEMM epilogue alpha parameter for precision.
// Source: [N, K/16] FP8 E4M3 (signed, but always positive for scale factors)
// Dest:   SfAtom layout UE4M3 (unsigned, just micro_scale — NOT combined)
__global__ void convert_scales_sfatom_kernel(
    const uint8_t* __restrict__ src_ms,    // [N, K/16] linear
    uint8_t*       __restrict__ dst_sf,    // SfAtom layout
    int N, int K, int n_k_tiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kSFVecSize;
    int total = N * K_groups;
    if (idx >= total) return;

    int n = idx / K_groups;
    int k_group = idx % K_groups;

    // Read signed E4M3 micro-scale and convert to float
    uint8_t ms_byte = src_ms[idx];
    uint32_t sign = (ms_byte >> 7) & 1;
    uint32_t exp  = (ms_byte >> 3) & 0x0F;
    uint32_t man  = ms_byte & 0x07;

    float ms_float;
    if (exp == 0) {
        ms_float = (float)man * (1.0f / 512.0f);
    } else {
        // Normal: (8 + man) * 2^(exp - 10)
        uint32_t fp32 = ((exp + 120u) << 23) | (man << 20);
        ms_float = __uint_as_float(fp32);
    }
    if (sign) ms_float = -ms_float;  // shouldn't happen for scales

    // Use micro_scale directly (tensor_scale deferred to GEMM alpha)
    float combined = fabsf(ms_float);

    // Convert to unsigned E4M3 (same bit layout as signed, sign=0)
    // Clamp to [0, 448]
    uint8_t ue4m3;
    if (combined < (1.0f / 512.0f)) {
        ue4m3 = 0;
    } else if (combined > 448.0f) {
        ue4m3 = (14 << 3) | 7;  // max E4M3 = 448
    } else {
        uint32_t fbits;
        memcpy(&fbits, &combined, sizeof(float));
        int f_exp = (int)((fbits >> 23) & 0xFF) - 127;
        uint32_t f_man = fbits & 0x7FFFFF;

        int e4 = f_exp + 7;  // E4M3 bias = 7
        if (e4 <= 0) {
            int shift = 1 - e4;
            uint32_t full_man = (1u << 23) | f_man;
            int right_shift = 20 + shift;
            uint8_t m3 = 0;
            if (right_shift < 32) {
                uint32_t shifted = full_man >> right_shift;
                uint32_t remainder = full_man & ((1u << right_shift) - 1);
                uint32_t half_point = 1u << (right_shift - 1);
                if (remainder > half_point || (remainder == half_point && (shifted & 1)))
                    shifted += 1;
                m3 = (uint8_t)(shifted & 0x07);
                if (shifted > 7) {
                    ue4m3 = (1 << 3);  // overflow to smallest normal
                } else {
                    ue4m3 = m3;
                }
            } else {
                ue4m3 = 0;
            }
            if (m3 <= 7 && e4 <= 0) ue4m3 = m3;
        } else if (e4 >= 15) {
            ue4m3 = (14 << 3) | 7;  // clamp to max
        } else {
            uint32_t round_bit = (f_man >> 19) & 1;
            uint32_t sticky = (f_man & 0x7FFFF) ? 1 : 0;
            uint8_t m3 = (uint8_t)((f_man >> 20) & 0x07);
            if (round_bit && (sticky || (m3 & 1))) {
                m3 += 1;
                if (m3 > 7) { m3 = 0; e4 += 1; }
                if (e4 >= 15) { ue4m3 = (14 << 3) | 7; } else {
                    ue4m3 = (uint8_t)(((e4 & 0x0F) << 3) | (m3 & 0x07));
                }
            } else {
                ue4m3 = (uint8_t)(((e4 & 0x0F) << 3) | (m3 & 0x07));
            }
        }
    }

    // Write to SfAtom position
    int dst_idx = sfatom_offset(n, k_group, n_k_tiles);
    dst_sf[dst_idx] = ue4m3;
}

// ---------------------------------------------------------------------------
// Activation quantization: FP16 [M, K] → NVFP4 packed + SfAtom UE4M3 scales
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t quantize_abs_to_fp4(float abs_val) {
    if (abs_val <= 0.0f)  return 0;
    if (abs_val >= 6.0f)  return 7;
    if (abs_val < 0.25f)  return 0;
    if (abs_val < 0.75f)  return 1;
    if (abs_val < 1.25f)  return 2;
    if (abs_val < 1.75f)  return 3;
    if (abs_val < 2.5f)   return 4;
    if (abs_val < 3.5f)   return 5;
    if (abs_val < 5.0f)   return 6;
    return 7;
}

// Each thread handles one micro-block of 16 elements.
// Outputs packed FP4 + computes UE4M3 scale factor written to SfAtom layout.
__global__ void quantize_fp16_nvfp4_cutlass_kernel(
    const half* __restrict__ input,        // [M, K] FP16
    uint8_t*    __restrict__ packed_out,    // [M, K/2] packed nibbles
    uint8_t*    __restrict__ sf_out,        // SfAtom layout UE4M3
    int M, int K, int n_k_tiles)
{
    int mb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kSFVecSize;
    int total_mb = M * K_groups;
    if (mb_idx >= total_mb) return;

    int row    = mb_idx / K_groups;
    int k_group = mb_idx % K_groups;
    int base   = row * K + k_group * kSFVecSize;

    // Load 16 values and find absmax
    float vals[kSFVecSize];
    float local_absmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < kSFVecSize; i++) {
        vals[i] = __half2float(input[base + i]);
        float av = fabsf(vals[i]);
        if (av > local_absmax) local_absmax = av;
    }

    // Compute UE4M3 scale factor = local_absmax / 6.0
    // (tensor_scale is absorbed: combined = local_absmax / 6.0)
    float scale_f = local_absmax / 6.0f;
    if (scale_f < (1.0f / 512.0f)) scale_f = (1.0f / 512.0f);
    if (scale_f > 448.0f) scale_f = 448.0f;

    // Encode as UE4M3 (same bit layout as E4M3, sign=0)
    uint8_t ue4m3;
    {
        uint32_t fbits;
        memcpy(&fbits, &scale_f, sizeof(float));
        int f_exp = (int)((fbits >> 23) & 0xFF) - 127;
        uint32_t f_man = fbits & 0x7FFFFF;
        int e4 = f_exp + 7;

        if (e4 <= 0) {
            int shift = 1 - e4;
            uint32_t full_man = (1u << 23) | f_man;
            int right_shift = 20 + shift;
            if (right_shift >= 32) { ue4m3 = 0; }
            else {
                uint32_t shifted = full_man >> right_shift;
                uint32_t rem = full_man & ((1u << right_shift) - 1);
                uint32_t half_pt = 1u << (right_shift - 1);
                if (rem > half_pt || (rem == half_pt && (shifted & 1))) shifted++;
                ue4m3 = (shifted > 7) ? (uint8_t)(1 << 3) : (uint8_t)(shifted & 0x07);
            }
        } else if (e4 >= 15) {
            ue4m3 = (14 << 3) | 7;
        } else {
            uint32_t rb = (f_man >> 19) & 1;
            uint32_t st = (f_man & 0x7FFFF) ? 1 : 0;
            uint8_t m3 = (uint8_t)((f_man >> 20) & 0x07);
            if (rb && (st || (m3 & 1))) {
                m3++;
                if (m3 > 7) { m3 = 0; e4++; }
                if (e4 >= 15) ue4m3 = (14 << 3) | 7;
                else ue4m3 = (uint8_t)(((e4 & 0x0F) << 3) | (m3 & 0x07));
            } else {
                ue4m3 = (uint8_t)(((e4 & 0x0F) << 3) | (m3 & 0x07));
            }
        }
    }

    // Reconstruct actual scale from UE4M3 for consistent quantization
    float actual_scale;
    {
        uint32_t exp_bits = (ue4m3 >> 3) & 0x0F;
        uint32_t man_bits = ue4m3 & 0x07;
        if (exp_bits == 0) {
            actual_scale = (float)man_bits * (1.0f / 512.0f);
        } else {
            uint32_t fp32 = ((exp_bits + 120u) << 23) | (man_bits << 20);
            actual_scale = __uint_as_float(fp32);
        }
    }
    if (actual_scale == 0.0f) actual_scale = 1.0f / 512.0f;
    float inv_scale = 1.0f / actual_scale;

    // Write scale to SfAtom position
    int sf_idx = sfatom_offset(row, k_group, n_k_tiles);
    sf_out[sf_idx] = ue4m3;

    // Quantize and pack FP4 values
    int packed_base = row * (K / 2) + k_group * (kSFVecSize / 2);
    #pragma unroll
    for (int i = 0; i < kSFVecSize; i += 2) {
        float s0 = vals[i] * inv_scale;
        uint8_t sign0 = (s0 < 0.0f) ? 1u : 0u;
        uint8_t code0 = quantize_abs_to_fp4(fabsf(s0));
        uint8_t fp4_0 = (sign0 << 3) | code0;

        float s1 = vals[i + 1] * inv_scale;
        uint8_t sign1 = (s1 < 0.0f) ? 1u : 0u;
        uint8_t code1 = quantize_abs_to_fp4(fabsf(s1));
        uint8_t fp4_1 = (sign1 << 3) | code1;

        packed_out[packed_base + i / 2] = (fp4_1 << 4) | fp4_0;
    }
}

// ---------------------------------------------------------------------------
// Host-callable functions
// ---------------------------------------------------------------------------

void convert_nvfp4_to_cutlass(const NvFP4QuantResult& src,
                               CutlassNvFP4Weight& dst,
                               cudaStream_t stream)
{
    assert(src.packed_data && "source must be quantized");
    int64_t N = src.N;
    int64_t K = src.K;

    // Data pointer is borrowed as RowMajor [N, K/2].
    // Despite LayoutBTag=ColumnMajor, CUTLASS block-scaled GEMM uses RowMajor stride.

    // Allocate SfAtom scale buffer
    size_t sf_bytes = cutlass_nvfp4_sf_size(static_cast<int>(N), static_cast<int>(K));
    void* d_sf = nullptr;
    cudaMalloc(&d_sf, sf_bytes);
    cudaMemsetAsync(d_sf, 0, sf_bytes, stream);  // zero-init for padding

    // Convert scales to SfAtom layout (micro_scale only, tensor_scale deferred to GEMM alpha)
    {
        int K_groups = static_cast<int>(K) / kSFVecSize;
        int total = static_cast<int>(N) * K_groups;
        int n_k_tiles = (static_cast<int>(K) + kAtomKElems - 1) / kAtomKElems;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        convert_scales_sfatom_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(src.micro_scales),
            reinterpret_cast<uint8_t*>(d_sf),
            static_cast<int>(N), static_cast<int>(K), n_k_tiles);
    }

    dst.data = src.packed_data;  // borrowed pointer (not owned)
    dst.scale_factors = d_sf;
    dst.tensor_scale = src.tensor_scale;
    dst.N = N;
    dst.K = K;
    dst.sf_bytes = sf_bytes;

    IMP_LOG_DEBUG("convert_nvfp4_to_cutlass: N=%lld K=%lld sf=%.2f MiB (data borrowed)",
                  (long long)N, (long long)K,
                  sf_bytes / (1024.0 * 1024.0));
}

void free_cutlass_nvfp4_weight(CutlassNvFP4Weight& w) {
    // data is borrowed from NvFP4QuantResult — do NOT free it
    w.data = nullptr;
    if (w.scale_factors) { cudaFree(w.scale_factors); w.scale_factors = nullptr; }
    w.N = w.K = 0;
    w.sf_bytes = 0;
}

void quantize_fp16_to_nvfp4_cutlass(const void* src_fp16, void* dst_data,
                                     void* dst_sf, int M, int K,
                                     cudaStream_t stream)
{
    assert(K % kSFVecSize == 0 && "K must be multiple of 16");

    // Zero the SF buffer for padding safety
    size_t sf_bytes = cutlass_nvfp4_sf_size(M, K);
    cudaMemsetAsync(dst_sf, 0, sf_bytes, stream);

    int K_groups = K / kSFVecSize;
    int total_mb = M * K_groups;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;
    quantize_fp16_nvfp4_cutlass_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16),
        reinterpret_cast<uint8_t*>(dst_data),
        reinterpret_cast<uint8_t*>(dst_sf),
        M, K, n_k_tiles);
}

// ---------------------------------------------------------------------------
// CUTLASS GEMM execution
// ---------------------------------------------------------------------------

#if defined(IMP_USE_CUTLASS) && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

// Persistent workspace and GEMM instance
static void* s_cutlass_workspace = nullptr;
static size_t s_cutlass_workspace_size = 0;

size_t gemm_nvfp4_cutlass_sm120_workspace(int M, int N, int K) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
    };

    return Gemm::get_workspace_size(args);
}

bool gemm_nvfp4_cutlass_sm120(const void* a_data, const void* a_sf,
                               const CutlassNvFP4Weight& b,
                               void* d_fp16, int M, int N, int K,
                               void* workspace, size_t workspace_size,
                               cudaStream_t stream)
{
    // Flush any prior async errors — a sticky CUDA error will make
    // cuTensorMapEncodeTiled return 719 (LAUNCH_FAILED) instead of the real code.
    {
        cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            IMP_LOG_ERROR("CUTLASS sm120: prior CUDA error before GEMM: %s", cudaGetErrorString(prior));
            return false;
        }
    }

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    auto* a_ptr = reinterpret_cast<const ElementA::DataType*>(a_data);
    auto* b_ptr = reinterpret_cast<const ElementB::DataType*>(b.data);
    auto* sfa_ptr = reinterpret_cast<const ElementA::ScaleFactorType*>(a_sf);
    auto* sfb_ptr = reinterpret_cast<const ElementB::ScaleFactorType*>(b.scale_factors);

    // C pointer must be valid even with beta=0 — CUTLASS creates a TMA
    // descriptor for C during initialize() and cuTensorMapEncodeTiled
    // fails on nullptr.  Re-use the D buffer since it's never read.
    auto* d_ptr = reinterpret_cast<ElementD*>(d_fp16);

    // Use tensor_scale as alpha: compensates for not absorbing it into SFB.
    // D = tensor_scale * (A_fp4 * SFA * B_fp4 * micro_scale_only) = correct result.
    float alpha = b.tensor_scale;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {a_ptr, stride_A, b_ptr, stride_B, sfa_ptr, layout_SFA, sfb_ptr, layout_SFB},
        {{alpha, 0.0f},
         d_ptr, stride_C,  // C = D buffer (beta=0, never read)
         d_ptr, stride_D}
    };

    Gemm gemm;
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_WARN("CUTLASS sm120 NVFP4 GEMM: can_implement failed (%d) for M=%d N=%d K=%d",
                     (int)st, M, N, K);
        return false;
    }

    // Ensure workspace
    size_t needed = Gemm::get_workspace_size(args);
    void* ws = workspace;
    if (needed > workspace_size) {
        if (needed > s_cutlass_workspace_size) {
            if (s_cutlass_workspace) cudaFree(s_cutlass_workspace);
            cudaMalloc(&s_cutlass_workspace, needed);
            s_cutlass_workspace_size = needed;
        }
        ws = s_cutlass_workspace;
    }

    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS sm120 NVFP4 GEMM: initialize failed (%d) M=%d N=%d K=%d",
                      (int)st, M, N, K);
        return false;
    }

    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS sm120 NVFP4 GEMM: run failed (%d)", (int)st);
        return false;
    }

    return true;
}

bool cutlass_sm120_nvfp4_available() {
    return true;
}

#else // !CUTLASS_ARCH_MMA_SM120_SUPPORTED || !IMP_USE_CUTLASS

// Stubs when CUTLASS sm_120 is not compiled
size_t gemm_nvfp4_cutlass_sm120_workspace(int, int, int) { return 0; }

bool gemm_nvfp4_cutlass_sm120(const void*, const void*,
                               const CutlassNvFP4Weight&,
                               void*, int, int, int,
                               void*, size_t,
                               cudaStream_t) {
    return false;
}

bool cutlass_sm120_nvfp4_available() { return false; }

#endif

} // namespace imp
