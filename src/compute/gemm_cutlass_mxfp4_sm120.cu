// CUTLASS sm_120 block-scaled MXFP4×MXFP4 GEMM for prefill acceleration.
//
// Parallel to the NVFP4 CUTLASS path (gemm_cutlass_sm120.cu) but uses
// mx_float4_t<float_e2m1_t> with UE8M0 scale factors (SFVecSize=32).
//
// MXFP4 uses the same E2M1 data encoding as NVFP4 — the packed nibble
// layout is identical. Only the scale factor format differs:
//   NVFP4: UE4M3 per 16 elements (finer granularity, 3 mantissa bits)
//   MXFP4: UE8M0 per 32 elements (wider dynamic range, pure exponent)
//
// The hardware tensor core instruction for MXFP4 groups 32 elements per
// scale factor vs 16 for NVFP4, potentially allowing different scheduling.

#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "compute/hadamard.h"
#include "quant/nvfp4_quant.h"
#include "quant/fp8_utils.cuh"
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
// CUTLASS GEMM type configuration: MXFP4 × MXFP4 → FP16
// Uses mx_float4_t with UE8M0 scales (SFVecSize=32).
// ---------------------------------------------------------------------------

using MxElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using MxLayoutATag = cutlass::layout::RowMajor;
constexpr int MxAlignmentA = 32;

using MxElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using MxLayoutBTag = cutlass::layout::ColumnMajor;
constexpr int MxAlignmentB = 32;

using MxElementD = cutlass::half_t;
using MxElementC = cutlass::half_t;
using MxLayoutCTag = cutlass::layout::RowMajor;
using MxLayoutDTag = cutlass::layout::RowMajor;
constexpr int MxAlignmentD = 128 / cutlass::sizeof_bits<MxElementD>::value;
constexpr int MxAlignmentC = 128 / cutlass::sizeof_bits<MxElementC>::value;

using MxElementAccumulator = float;
using MxArchTag = cutlass::arch::Sm120;
using MxOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using MxThreadBlockShape = Shape<_128, _128, _128>;
using MxClusterShape = Shape<_1, _1, _1>;

using MxCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    MxArchTag, MxOperatorClass, MxThreadBlockShape, MxClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, MxElementAccumulator, MxElementAccumulator, MxElementC,
    MxLayoutCTag, MxAlignmentC, MxElementD, MxLayoutDTag, MxAlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using MxCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    MxArchTag, MxOperatorClass, MxElementA, MxLayoutATag, MxAlignmentA, MxElementB, MxLayoutBTag,
    MxAlignmentB, MxElementAccumulator, MxThreadBlockShape, MxClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename MxCollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using MxGemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, MxCollectiveMainloop,
                                                          MxCollectiveEpilogue, void>;

using MxGemm = cutlass::gemm::device::GemmUniversalAdapter<MxGemmKernel>;

using MxStrideA = typename MxGemm::GemmKernel::StrideA;
using MxStrideB = typename MxGemm::GemmKernel::StrideB;
using MxStrideC = typename MxGemm::GemmKernel::StrideC;
using MxStrideD = typename MxGemm::GemmKernel::StrideD;
using MxLayoutSFA = typename MxGemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using MxLayoutSFB = typename MxGemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using MxSm1xxConfig = typename MxGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

static_assert(MxGemm::GemmKernel::CollectiveMainloop::TiledMma::Traits::SFVecSize == 32,
              "CUTLASS SFVecSize mismatch — expected 32 for mx_float4_t");

namespace imp {

// ---------------------------------------------------------------------------
// SfAtom layout for MXFP4: SFVecSize=32 (one UE8M0 per 32 data elements)
// ---------------------------------------------------------------------------
static constexpr int kMxSFVecSize = 32;
static constexpr int kMxAtomRows = 128;
static constexpr int kMxAtomKGroups = 4;
static constexpr int kMxAtomKElems = kMxSFVecSize * kMxAtomKGroups;  // 128
static constexpr int kMxAtomSize = kMxAtomRows * kMxAtomKGroups;     // 512

__device__ __host__ __forceinline__ int mx_sfatom_offset(int row, int k_group, int n_k_tiles) {
    int tile_row = row / kMxAtomRows;
    int tile_k = k_group / kMxAtomKGroups;
    int row_local = row % kMxAtomRows;
    int k_local = k_group % kMxAtomKGroups;

    int n0 = row_local % 32;
    int n1 = row_local / 32;

    int atom_offset = n0 * 16 + n1 * 4 + k_local;
    int tile_base = (tile_row * n_k_tiles + tile_k) * kMxAtomSize;
    return tile_base + atom_offset;
}

size_t cutlass_mxfp4_sf_size(int rows, int K) {
    int n_row_tiles = (rows + kMxAtomRows - 1) / kMxAtomRows;
    int n_k_tiles = (K + kMxAtomKElems - 1) / kMxAtomKElems;
    return static_cast<size_t>(n_row_tiles) * n_k_tiles * kMxAtomSize;
}

// ---------------------------------------------------------------------------
// Convert UE4M3 micro-scales (NVFP4, per 16 elements) to UE8M0 (MXFP4, per 32).
// Merges 2 consecutive UE4M3 values into 1 UE8M0 by taking the max exponent.
// ---------------------------------------------------------------------------

// UE8M0: pure exponent, value = 2^(bits - 127).  bits=0 → 2^-127, bits=254 → 2^127.
__device__ __forceinline__ uint8_t float_to_ue8m0(float val) {
    if (val <= 0.0f)
        return 0;  // map zero/negative to minimum
    uint32_t fbits;
    memcpy(&fbits, &val, sizeof(float));
    int f_exp = (int)((fbits >> 23) & 0xFF);
    // Round up if mantissa bits are set (ceil to next power of 2)
    if (fbits & 0x7FFFFF)
        f_exp++;
    // UE8M0 exponent = f_exp (FP32 biased exponent maps directly)
    // Clamp to [0, 254] (255 = inf/nan reserved)
    if (f_exp < 0)
        return 0;
    if (f_exp > 254)
        return 254;
    return (uint8_t)f_exp;
}

// fp8_e4m3_to_float is provided by quant/fp8_utils.cuh (shared across NVFP4/MXFP4).

// Merge 2 NVFP4 micro-scales (UE4M3, per 16 elements each) into 1 MXFP4 scale (UE8M0, per 32).
// Takes the max of the two micro-scales, then encodes as UE8M0 (nearest power of 2).
// This loses some precision compared to NVFP4's finer per-16 scales.
__global__ void convert_nvfp4_to_mxfp4_scales_kernel(
    const uint8_t* __restrict__ src_ms,  // [N, K/16] NVFP4 UE4M3
    uint8_t* __restrict__ dst_sf,        // SfAtom MXFP4 UE8M0
    uint8_t* __restrict__ dst_linear,    // [N, K/32] row-major UE8M0 (for GEMV)
    float tensor_scale,                  // NVFP4 tensor scale
    int N, int K, int n_k_tiles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups_mx = K / kMxSFVecSize;  // number of MXFP4 groups (per 32)
    int total = N * K_groups_mx;
    if (idx >= total)
        return;

    int n = idx / K_groups_mx;
    int mx_group = idx % K_groups_mx;

    // Each MXFP4 group of 32 corresponds to 2 consecutive NVFP4 groups of 16
    int nv_group0 = mx_group * 2;
    int nv_group1 = mx_group * 2 + 1;
    int K_groups_nv = K / 16;

    float ms0 = fabsf(fp8_e4m3_to_float(src_ms[n * K_groups_nv + nv_group0]));
    float ms1 = fabsf(fp8_e4m3_to_float(src_ms[n * K_groups_nv + nv_group1]));

    // Combined scale = tensor_scale * max(ms0, ms1)
    // This is the scale that maps the group of 32 elements to FP4 range [0, 6]
    float combined = tensor_scale * fmaxf(ms0, ms1);

    // Encode as UE8M0
    uint8_t ue8m0 = float_to_ue8m0(combined / 6.0f);

    int sf_idx = mx_sfatom_offset(n, mx_group, n_k_tiles);
    dst_sf[sf_idx] = ue8m0;
    if (dst_linear)
        dst_linear[static_cast<size_t>(n) * K_groups_mx + mx_group] = ue8m0;
}

void convert_nvfp4_to_mxfp4_cutlass(const NvFP4QuantResult& src, CutlassMxFP4Weight& dst,
                                    cudaStream_t stream) {
    IMP_CHECK(src.packed_data != nullptr, "convert_nvfp4_to_mxfp4_cutlass: src.packed_data is null");
    int64_t N = src.N;
    int64_t K = src.K;
    IMP_CHECK(K % kMxSFVecSize == 0, "convert_nvfp4_to_mxfp4_cutlass: K=%lld must be multiple of %d",
              static_cast<long long>(K), kMxSFVecSize);

    size_t sf_bytes = cutlass_mxfp4_sf_size(static_cast<int>(N), static_cast<int>(K));
    int K_groups_mx = static_cast<int>(K) / kMxSFVecSize;
    size_t linear_sf_bytes = static_cast<size_t>(N) * K_groups_mx;

    void* d_sf = nullptr;
    void* d_linear_sf = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_sf, sf_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_linear_sf, linear_sf_bytes));
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_sf, 0, sf_bytes, stream));

    int total = static_cast<int>(N) * K_groups_mx;
    int n_k_tiles = (static_cast<int>(K) + kMxAtomKElems - 1) / kMxAtomKElems;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    convert_nvfp4_to_mxfp4_scales_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src.micro_scales), reinterpret_cast<uint8_t*>(d_sf),
        reinterpret_cast<uint8_t*>(d_linear_sf), src.tensor_scale, static_cast<int>(N), static_cast<int>(K),
        n_k_tiles);

    dst.data = src.packed_data;
    dst.scale_factors = d_sf;
    dst.linear_scales = d_linear_sf;
    dst.tensor_scale = 1.0f;  // absorbed into UE8M0 scales (unlike NVFP4 path)
    dst.N = N;
    dst.K = K;
    dst.sf_bytes = sf_bytes;

    IMP_LOG_DEBUG("convert_nvfp4_to_mxfp4: N=%lld K=%lld sf=%.2f MiB linear_sf=%.2f MiB", (long long)N,
                  (long long)K, sf_bytes / (1024.0 * 1024.0), linear_sf_bytes / (1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------
// Unpack native MXFP4 GGUF blocks into CUTLASS format.
// GGUF block: [16 bytes E2M1 data | 1 byte UE8M0 scale] × N_blocks
// Output: separate contiguous data array + SfAtom scale array.
// ---------------------------------------------------------------------------

// Build SfAtom + linear scale layouts from pre-split scale bytes.
// Input: linear_scales at gpu_buf + data_bytes (already [N, K/32] row-major)
// Output: SfAtom layout for CUTLASS + copy to separate linear buffer
__global__ void build_mxfp4_scales_kernel(
    const uint8_t* __restrict__ linear_src,  // [total_blocks] row-major scales
    uint8_t* __restrict__ sf_out,            // SfAtom layout
    uint8_t* __restrict__ linear_out,        // [N, K/32] separate linear buffer
    int total_blocks, int blocks_per_row, int n_k_tiles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_blocks)
        return;
    int row = idx / blocks_per_row;
    int blk = idx % blocks_per_row;
    uint8_t scale = linear_src[idx];
    sf_out[mx_sfatom_offset(row, blk, n_k_tiles)] = scale;
    linear_out[idx] = scale;
}

bool unpack_mxfp4_gguf(const void* raw_gpu, int64_t N, int64_t K, CutlassMxFP4Weight& dst,
                       cudaStream_t stream) {
    if (K % 32 != 0) {
        IMP_LOG_WARN("unpack_mxfp4_gguf: K=%lld not multiple of 32", (long long)K);
        return false;
    }

    int blocks_per_row = static_cast<int>(K) / 32;
    int total_blocks = static_cast<int>(N) * blocks_per_row;

    // Input from weight_upload: [data(N×K/2) | scales(total_blocks)] in one buffer.
    // Data is already contiguous. Build SfAtom + linear scale layouts from appended scales.
    size_t data_bytes = static_cast<size_t>(N) * (K / 2);
    size_t sf_bytes = cutlass_mxfp4_sf_size(static_cast<int>(N), static_cast<int>(K));
    size_t linear_sf_bytes = static_cast<size_t>(total_blocks);

    void* d_sf = nullptr;
    void* d_linear_sf = nullptr;
    if (cudaMalloc(&d_sf, sf_bytes) != cudaSuccess ||
        cudaMalloc(&d_linear_sf, linear_sf_bytes) != cudaSuccess) {
        if (d_sf)
            cudaFree(d_sf);
        if (d_linear_sf)
            cudaFree(d_linear_sf);
        return false;
    }
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_sf, 0, sf_bytes, stream));

    const uint8_t* scale_src = static_cast<const uint8_t*>(raw_gpu) + data_bytes;
    int n_k_tiles = (static_cast<int>(K) + kMxAtomKElems - 1) / kMxAtomKElems;
    int threads = 256;
    int grid = (total_blocks + threads - 1) / threads;
    build_mxfp4_scales_kernel<<<grid, threads, 0, stream>>>(scale_src, static_cast<uint8_t*>(d_sf),
                                                            static_cast<uint8_t*>(d_linear_sf), total_blocks,
                                                            blocks_per_row, n_k_tiles);

    dst.data = raw_gpu;  // data at start of buffer (contiguous packed nibbles)
    dst.scale_factors = d_sf;
    dst.linear_scales = d_linear_sf;
    dst.tensor_scale = 1.0f;
    dst.N = N;
    dst.K = K;
    dst.sf_bytes = sf_bytes;
    dst.owns_data = false;

    IMP_LOG_DEBUG("unpack_mxfp4_gguf: N=%lld K=%lld sf=%.2f MiB (pre-split)", (long long)N, (long long)K,
                  sf_bytes / (1024.0 * 1024.0));
    return true;
}

// ---------------------------------------------------------------------------
// Dequantize MXFP4 → FP16 for decode fallback (GEMV needs FP16 weights).
// ---------------------------------------------------------------------------

// E2M1 dequant table: 16 FP32 values for 4-bit indices 0-15.
// E2M1: sign(1) + exp(2) + mantissa(1), bias=1
__device__ __constant__ float kE2M1Table[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,  // positive
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // negative
};

// Dequant from split MXFP4 layout: [data(N×K/2) | scales(total_blocks)].
// Used for alpha/beta FP16 dequant before MXFP4 registration.
__global__ void dequant_mxfp4_split_kernel(const uint8_t* __restrict__ data,    // [N × K/2] packed E2M1
                                           const uint8_t* __restrict__ scales,  // [total_blocks] UE8M0
                                           half* __restrict__ out,              // [N, K]
                                           int N, int K, int blocks_per_row) {
    int64_t blk_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_blocks = static_cast<int64_t>(N) * blocks_per_row;
    if (blk_idx >= total_blocks)
        return;

    int row = static_cast<int>(blk_idx / blocks_per_row);
    int blk = static_cast<int>(blk_idx % blocks_per_row);

    uint8_t ue8m0 = scales[blk_idx];
    float scale;
    uint32_t fbits = static_cast<uint32_t>(ue8m0) << 23;
    memcpy(&scale, &fbits, sizeof(float));

    const uint8_t* block_data = data + blk_idx * 16;
    int64_t out_base = static_cast<int64_t>(row) * K + blk * 32;
    int64_t out_limit = static_cast<int64_t>(N) * K;
    if (out_base + 31 >= out_limit)
        return;

    for (int i = 0; i < 16; i++) {
        uint8_t packed = block_data[i];
        out[out_base + i * 2] = __float2half(kE2M1Table[packed & 0xF] * scale);
        out[out_base + i * 2 + 1] = __float2half(kE2M1Table[(packed >> 4) & 0xF] * scale);
    }
}

void dequant_mxfp4_to_fp16(const void* raw_mxfp4_data, int64_t N, int64_t K, void* dst_fp16,
                           cudaStream_t stream) {
    int blocks_per_row = static_cast<int>(K / 32);
    int64_t total_blocks = N * blocks_per_row;
    size_t data_bytes = static_cast<size_t>(N) * (K / 2);

    // Split layout: [data(N×K/2) | scales(total_blocks)]
    const uint8_t* data_ptr = static_cast<const uint8_t*>(raw_mxfp4_data);
    const uint8_t* scale_ptr = data_ptr + data_bytes;

    int threads = 256;
    int grid = static_cast<int>((total_blocks + threads - 1) / threads);
    dequant_mxfp4_split_kernel<<<grid, threads, 0, stream>>>(data_ptr, scale_ptr,
                                                             static_cast<half*>(dst_fp16),
                                                             static_cast<int>(N), static_cast<int>(K),
                                                             blocks_per_row);
}

// Forward declaration (defined below in activation quantization section)
__global__ void quantize_fp16_mxfp4_cutlass_kernel(const half* __restrict__ input,
                                                   uint8_t* __restrict__ packed_out,
                                                   uint8_t* __restrict__ sf_out, int M, int K, int n_k_tiles);

void convert_nvfp4_to_mxfp4_hadamard(const NvFP4QuantResult& src, CutlassMxFP4Weight& dst, void* scratch_fp16,
                                     int hadamard_block_size, cudaStream_t stream) {
    IMP_CHECK(src.packed_data != nullptr, "convert_nvfp4_to_mxfp4_hadamard: src.packed_data is null");
    int64_t N = src.N;
    int64_t K = src.K;
    IMP_CHECK(K % kMxSFVecSize == 0, "convert_nvfp4_to_mxfp4_hadamard: K=%lld must be multiple of %d",
              static_cast<long long>(K), kMxSFVecSize);
    IMP_CHECK(K % hadamard_block_size == 0,
              "convert_nvfp4_to_mxfp4_hadamard: K=%lld must be multiple of hadamard_block_size=%d",
              static_cast<long long>(K), hadamard_block_size);

    // 1. Dequant NVFP4 → FP16 into scratch
    dequantize_nvfp4_to_fp16(src, scratch_fp16, stream);

    // 2. Apply block-diagonal Hadamard along K dimension (in-place)
    // Each row of K elements gets block-diagonal WHT applied.
    // WHT is self-inverse and symmetric: H·H^T = N·I, so (H/√N)^2 = I.
    hadamard_transform_fp16(reinterpret_cast<const half*>(scratch_fp16),
                            reinterpret_cast<half*>(scratch_fp16), static_cast<int>(N), static_cast<int>(K),
                            hadamard_block_size, stream);
    IMP_LOG_DEBUG("mxfp4_hadamard: applied WHT bs=%d to weight [%lld, %lld]", hadamard_block_size,
                  (long long)N, (long long)K);

    // 3. Quantize rotated FP16 → MXFP4 (new packed data + new scales)
    size_t packed_bytes = static_cast<size_t>(N) * (K / 2);
    size_t sf_bytes = cutlass_mxfp4_sf_size(static_cast<int>(N), static_cast<int>(K));

    void* d_packed = nullptr;
    void* d_sf = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_packed, packed_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_sf, sf_bytes));
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_sf, 0, sf_bytes, stream));

    int K_groups = static_cast<int>(K) / kMxSFVecSize;
    int total_mb = static_cast<int>(N) * K_groups;
    int n_k_tiles = (static_cast<int>(K) + kMxAtomKElems - 1) / kMxAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;
    quantize_fp16_mxfp4_cutlass_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(scratch_fp16), reinterpret_cast<uint8_t*>(d_packed),
        reinterpret_cast<uint8_t*>(d_sf), static_cast<int>(N), static_cast<int>(K), n_k_tiles);

    dst.data = d_packed;
    dst.scale_factors = d_sf;
    dst.tensor_scale = 1.0f;
    dst.N = N;
    dst.K = K;
    dst.sf_bytes = sf_bytes;
    dst.owns_data = true;

    IMP_LOG_DEBUG(
        "convert_nvfp4_to_mxfp4_hadamard: N=%lld K=%lld had_bs=%d "
        "packed=%.2f MiB sf=%.2f MiB",
        (long long)N, (long long)K, hadamard_block_size, packed_bytes / (1024.0 * 1024.0),
        sf_bytes / (1024.0 * 1024.0));
}

void free_cutlass_mxfp4_weight(CutlassMxFP4Weight& w) {
    if (w.owns_data && w.data) {
        IMP_CUDA_CHECK_LOG(cudaFree(const_cast<void*>(w.data)));
    }
    w.data = nullptr;
    if (w.scale_factors) {
        IMP_CUDA_CHECK_LOG(cudaFree(w.scale_factors));
        w.scale_factors = nullptr;
    }
    if (w.linear_scales) {
        IMP_CUDA_CHECK_LOG(cudaFree(w.linear_scales));
        w.linear_scales = nullptr;
    }
    w.N = w.K = 0;
    w.sf_bytes = 0;
    w.owns_data = false;
}

// ---------------------------------------------------------------------------
// Activation quantization: FP16 [M, K] → MXFP4 packed + SfAtom UE8M0
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t quantize_abs_to_fp4_mx(float abs_val) {
    // Branchless: count of midpoint thresholds exceeded gives the E2M1 code.
    uint8_t code = (abs_val >= 0.25f) + (abs_val >= 0.75f) + (abs_val >= 1.25f) + (abs_val >= 1.75f) +
                   (abs_val >= 2.5f) + (abs_val >= 3.5f) + (abs_val >= 5.0f);
    return code;  // 0..7
}

// HW FP4 conversion: two scaled FP32 → packed byte (low=v0, high=v1).
// IEEE RNE rounding; sm_120+ only.
__device__ __forceinline__ uint8_t pack_fp4_pair_hw_mx(float v0, float v1) {
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
    uint8_t c0 = (sign0 << 3) | quantize_abs_to_fp4_mx(fabsf(v0));
    uint8_t c1 = (sign1 << 3) | quantize_abs_to_fp4_mx(fabsf(v1));
    return (c1 << 4) | c0;
#endif
}

__device__ __forceinline__ float ue8m0_to_float(uint8_t bits) {
    // UE8M0: value = 2^(bits - 127)
    if (bits == 0)
        return 5.877472e-39f;  // 2^-127
    uint32_t fp32 = ((uint32_t)bits) << 23;
    return __uint_as_float(fp32);
}

// Each thread handles one micro-block of 32 elements (MXFP4 group size).
__global__ void quantize_fp16_mxfp4_cutlass_kernel(const half* __restrict__ input,
                                                   uint8_t* __restrict__ packed_out,
                                                   uint8_t* __restrict__ sf_out, int M, int K,
                                                   int n_k_tiles) {
    int mb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kMxSFVecSize;
    int total_mb = M * K_groups;
    if (mb_idx >= total_mb)
        return;

    int row = mb_idx / K_groups;
    int k_group = mb_idx % K_groups;
    int base = row * K + k_group * kMxSFVecSize;

    // Load 32 values via vectorized half2 loads and find absmax
    float vals[32];
    float local_absmax = 0.0f;
    const half2* src_h2 = reinterpret_cast<const half2*>(input + base);
#pragma unroll
    for (int i = 0; i < 16; i++) {
        half2 h2 = src_h2[i];
        vals[i * 2] = __half2float(h2.x);
        vals[i * 2 + 1] = __half2float(h2.y);
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(vals[i * 2]), fabsf(vals[i * 2 + 1])));
    }

    // Compute UE8M0 scale = ceil_pow2(absmax / 6.0)
    float scale_target = local_absmax / 6.0f;
    uint8_t ue8m0 = float_to_ue8m0(scale_target);
    float actual_scale = ue8m0_to_float(ue8m0);
    if (actual_scale == 0.0f)
        actual_scale = 5.877472e-39f;
    float inv_scale = 1.0f / actual_scale;

    // Write scale to SfAtom position
    int sf_idx = mx_sfatom_offset(row, k_group, n_k_tiles);
    sf_out[sf_idx] = ue8m0;

    // Quantize and pack FP4 values (2 per byte) via HW conversion
    int packed_base = row * (K / 2) + k_group * (kMxSFVecSize / 2);
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
        float s0 = vals[i] * inv_scale;
        float s1 = vals[i + 1] * inv_scale;
        packed_out[packed_base + i / 2] = pack_fp4_pair_hw_mx(s0, s1);
    }
}

void quantize_fp16_to_mxfp4_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream) {
    IMP_CHECK(K % kMxSFVecSize == 0, "quantize_fp16_to_mxfp4_cutlass: K=%d must be multiple of %d",
              K, kMxSFVecSize);

    size_t sf_bytes = cutlass_mxfp4_sf_size(M, K);
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(dst_sf, 0, sf_bytes, stream));

    int K_groups = K / kMxSFVecSize;
    int total_mb = M * K_groups;
    int n_k_tiles = (K + kMxAtomKElems - 1) / kMxAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;
    quantize_fp16_mxfp4_cutlass_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16), reinterpret_cast<uint8_t*>(dst_data),
        reinterpret_cast<uint8_t*>(dst_sf), M, K, n_k_tiles);
}

// ---------------------------------------------------------------------------
// CUTLASS GEMM execution
// ---------------------------------------------------------------------------

static void* s_mxfp4_workspace = nullptr;
static size_t s_mxfp4_workspace_size = 0;

size_t gemm_mxfp4_cutlass_sm120_workspace(int M, int N, int K) {
    auto stride_A = cutlass::make_cute_packed_stride(MxStrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(MxStrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(MxStrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(MxStrideD{}, {M, N, 1});

    auto layout_SFA = MxSm1xxConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = MxSm1xxConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    typename MxGemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                    {M, N, K, 1},
                                    {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr,
                                     layout_SFB},
                                    {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}};

    return MxGemm::get_workspace_size(args);
}

bool gemm_mxfp4_cutlass_sm120(const void* a_data, const void* a_sf, const CutlassMxFP4Weight& b, void* d_fp16,
                              int M, int N, int K, void* workspace, size_t workspace_size,
                              cudaStream_t stream) {
    {
        cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            IMP_LOG_ERROR("CUTLASS MXFP4 sm120: prior CUDA error: %s", cudaGetErrorString(prior));
            return false;
        }
    }

    auto stride_A = cutlass::make_cute_packed_stride(MxStrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(MxStrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(MxStrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(MxStrideD{}, {M, N, 1});

    auto layout_SFA = MxSm1xxConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = MxSm1xxConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    auto* a_ptr = reinterpret_cast<const MxElementA::DataType*>(a_data);
    auto* b_ptr = reinterpret_cast<const MxElementB::DataType*>(b.data);
    auto* sfa_ptr = reinterpret_cast<const MxElementA::ScaleFactorType*>(a_sf);
    auto* sfb_ptr = reinterpret_cast<const MxElementB::ScaleFactorType*>(b.scale_factors);
    auto* d_ptr = reinterpret_cast<MxElementD*>(d_fp16);

    float alpha = b.tensor_scale;

    typename MxGemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                    {M, N, K, 1},
                                    {a_ptr, stride_A, b_ptr, stride_B, sfa_ptr, layout_SFA, sfb_ptr,
                                     layout_SFB},
                                    {{alpha, 0.0f}, d_ptr, stride_C, d_ptr, stride_D}};

    MxGemm gemm;
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_WARN("CUTLASS MXFP4 sm120: can_implement failed (%d) M=%d N=%d K=%d", (int)st, M, N, K);
        return false;
    }

    size_t needed = MxGemm::get_workspace_size(args);
    void* ws = workspace;
    if (needed > workspace_size) {
        if (needed > s_mxfp4_workspace_size) {
            if (s_mxfp4_workspace)
                IMP_CUDA_CHECK_LOG(cudaFree(s_mxfp4_workspace));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&s_mxfp4_workspace, needed));
            s_mxfp4_workspace_size = needed;
        }
        ws = s_mxfp4_workspace;
    }

    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS MXFP4 sm120: initialize failed (%d) M=%d N=%d K=%d", (int)st, M, N, K);
        return false;
    }

    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS MXFP4 sm120: run failed (%d)", (int)st);
        return false;
    }

    return true;
}

bool cutlass_sm120_mxfp4_available() { return true; }

}  // namespace imp
