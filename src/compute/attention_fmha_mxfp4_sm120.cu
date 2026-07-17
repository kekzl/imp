// =============================================================================
// attention_fmha_mxfp4_sm120.cu -- FP4 E2M1 Flash Attention for sm_120
// =============================================================================
//
// Tiled flash attention with FP4 E2M1 Q·K^T score compute and FP16 P·V.
// Uses inline PTX mma.sync.aligned.kind::f8f6f4.m16n8k32 with E2M1 operands.
// Same k-dim as FP8, but 2x faster TC dispatch (halved register count).
//
// Per-row scale quantization: Q and K are quantized per-tile with per-row
// absmax scales.  After MMA, scores are corrected:
//   S_true[i,j] = q_scale[i] * k_scale[j] * S_mma[i,j]
//
// Pipeline per KV tile:
//   1. Load K tile → quantize to FP4 with per-row scale
//   2. FP4 MMA: S = Q_fp4 · K_fp4^T  (m16n8k32, E2M1)
//   3. Scale correction + softcap + causal mask + online softmax
//   4. Load V tile as FP16
//   5. FP16 WMMA: O += P · V
//
// Q is quantized once at kernel start and reused across all KV tiles.
// =============================================================================

#include "compute/attention_fmha_mxfp4_sm120.h"
#include "compute/attention_paged_common.cuh"
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "quant/fp8_utils.cuh"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

namespace imp {

// =============================================================================
// Constants
// =============================================================================

static constexpr int MX_WARP_SIZE = 32;
static constexpr int MX_NUM_WARPS = 8;
static constexpr int MX_BLOCK_THREADS = MX_WARP_SIZE * MX_NUM_WARPS;  // 256
static constexpr int MX_Bkv = 64;                                     // KV tile columns

// FP4 MMA tile dimensions: m16n8k32 (same shape as FP8, halved registers)
// FP4 A uses 2 uint32 (vs 4 for FP8), B uses 1 uint32 (vs 2 for FP8).
// 2x throughput comes from faster TC dispatch, not larger k-dim.
static constexpr int MX_MMA_M = 16;
static constexpr int MX_MMA_N = 8;
static constexpr int MX_MMA_K = 32;  // same as FP8

// FP16 WMMA for P·V (unchanged from FP16/FP8 FMHA)
static constexpr int MX_WMMA_M = 16;
static constexpr int MX_WMMA_N = 16;
static constexpr int MX_WMMA_K = 16;

// PVFP4 two-level P scaling (#846 / SageAttention3 §3.2): each P row is
// rescaled so its max lands at 448·6 before 1x16 microscaling — the largest
// value an E2M1 nibble (max 6) times the largest UE4M3 scale (448) can carry.
// Tail blocks (post-softmax P spans 6+ orders of magnitude) then map to
// mid-range UE4M3 scales instead of collapsing to zero, which is the measured
// failure mode of single-level per-16 quantization (fp4_pv_bench, p99=797%).
static constexpr float MX_PV_LEVEL1 = 448.0f * 6.0f;

// =============================================================================
// Device helpers: FP4 E2M1 quantization
// =============================================================================

// Pack two FP32 values into one FP4 E2M1 byte via hardware instruction.
// Layout: low nibble = v0, high nibble = v1. Values must already be scaled
// so that |v| ≤ 6 (values outside saturate to ±6).
//
// Uses the PTX hardware conversion on sm_120+ (works on CUDA 13.2+; the
// `f16x2` variant is broken per dead_ends.md, but the `f32` variant is
// correct — see sageattention3_study_2026_04_24 memory). Single PTX
// instruction replaces the former branchless cascade (14 compares +
// sign handling per call). Rounding is RNE (IEEE round-to-nearest-even)
// vs the software midpoint cascade — tiny output divergence on boundary
// values is acceptable (validated via A/B test against legacy path).
__device__ __forceinline__ uint8_t pack_fp4_pair(float v0, float v1) {
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
    // Software fallback: branchless compare cascade (E2M1 magnitudes
    // {0, 0.5, 1, 1.5, 2, 3, 4, 6} with midpoint thresholds).
    auto quant_abs = [](float a) -> uint8_t {
        return (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) + (a >= 2.5f) + (a >= 3.5f) +
               (a >= 5.0f);
    };
    uint8_t sign0 = (v0 < 0.0f) ? 1u : 0u;
    uint8_t code0 = (sign0 << 3) | quant_abs(fabsf(v0));
    uint8_t sign1 = (v1 < 0.0f) ? 1u : 0u;
    uint8_t code1 = (sign1 << 3) | quant_abs(fabsf(v1));
    return (code1 << 4) | code0;
#endif
}

// Decode two packed FP4 E2M1 nibbles into a half2 via the HW converter
// (low nibble → .x, high nibble → .y — matches the KV-cache nibble packing
// and pack_fp4_pair above).
__device__ __forceinline__ half2 unpack_fp4_pair(uint8_t byte) {
#if __CUDA_ARCH__ >= 1200
    unsigned int fp16x2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
        : "=r"(fp16x2)
        : "r"((unsigned int)byte));
    return *reinterpret_cast<half2*>(&fp16x2);
#else
    (void)byte;
    return __half2half2(__float2half(0.0f));
#endif
}

// Paged NVFP4 KV cache pointers for the PagedKV kernel variant (#846
// KV-append-quant path). Layouts match write_kv_cache_nvfp4_kernel:
//   data:   [num_blocks, block_size, nkv, hd/2]  uint8 (2 nibbles/byte,
//           even d = low nibble)
//   scales: [num_blocks, block_size, nkv, hd/16] uint8 UE4M3 (absmax/6, RAW —
//           no attention-scale fold; the kernel applies `scale` post-MMA)
struct MxPagedKVArgs {
    const uint8_t* k_data = nullptr;
    const uint8_t* k_scales = nullptr;
    const uint8_t* v_data = nullptr;
    const uint8_t* v_scales = nullptr;
    const int* block_table = nullptr;  // flat (single sequence)
    int block_size = 0;
};

// =============================================================================
// Kernel template
// =============================================================================

// UseBlockScaleMma=false: legacy kind::f8f6f4.m16n8k32 (2× K-chunks, padded regs).
// UseBlockScaleMma=true:  kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 with
//                         per-16-element UE4M3 scales applied in the MMA instruction.
//                         Quantization uses per-k_group (16 elements) absmax,
//                         preserving local precision vs per-row. Post-MMA manual
//                         scaling is dropped — HW scales during the dot product.
// PVFP4 (#846, requires UseBlockScaleMma): P·V also runs on the block-scaled
//                         MMA. P is quantized per-row TWO-LEVEL: each row is
//                         rescaled so its max hits the full E4M3-scale range
//                         (448·6) before 1x16 microscaling — small-magnitude
//                         tail blocks then get representable UE4M3 scales
//                         instead of collapsing to zero. The inverse row factor
//                         is applied when accumulating the MMA output into
//                         O_acc. V is quantized per-16-block along the KV dim
//                         into a transposed smem tile (B-operand layout).
// d_kmean (#846 K-smoothing): per-(batch, kv_head, channel) mean of K,
//                         subtracted before quantization (nullptr = off). The
//                         dropped Q·mean^T term is constant per query row and
//                         cancels under softmax (launcher gates softcap == 0).
// Promote (#846 ThriftAttention, arXiv 2605.23081): d_promote is a per-
//                         (batch_head, q_tile, kv_tile) uint8 mask from the
//                         block-mean top-k pre-pass. Promoted KV tiles skip FP4
//                         quantization entirely: S is computed exactly in FP32
//                         from global-memory FP16 Q/K, and P·V takes the FP16
//                         WMMA path even under PVFP4. The promotion flag is
//                         uniform per (block, kv tile), so phase-level branches
//                         are __syncthreads()-safe. Quality spike — the FP32
//                         dot path is not a perf path.
// PagedKV (#846 KV-append-quant): K and V are read DIRECTLY from the paged
//                         NVFP4 KV cache (MxPagedKVArgs) — no in-kernel
//                         quantization at all. K's packed nibbles + UE4M3
//                         scales are byte-compatible with the blockscale-MMA
//                         smem layout (per-16-along-hd groups, even d = low
//                         nibble), so the K phase is a pure copy. V is
//                         dequantized FP16 into smem for the WMMA PV phase.
//                         Cache scales carry no attention-scale fold — Q is
//                         quantized with RAW absmax/6 scales and `scale` is
//                         applied post-MMA in the fused store. Promoted tiles
//                         compute FP32 dots over the DEQUANTIZED cache K
//                         (exact arithmetic over FP4-stored values). The K/V
//                         global-pointer args carry the FRESH FP16 current
//                         chunk (rows [q_offset, seq_kv)) — current-chunk
//                         tiles are force-promoted and read it exactly (the
//                         recency window is where FP4 storage hurts; the
//                         past is read FP4 from the cache).
template <int Bq, int HD, bool UseBlockScaleMma = false, bool PVFP4 = false, bool Promote = false,
          bool PagedKV = false>
__global__ void __launch_bounds__(MX_BLOCK_THREADS, 1) fmha_sm120_mxfp4_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
    int batch_size, int seq_q, int seq_kv, int n_heads, int n_kv_heads, float scale, bool causal,
    int sliding_window, float softcap, int q_offset, const float* __restrict__ d_kmean,
    const uint8_t* __restrict__ d_promote, int n_kv_tiles_mask, MxPagedKVArgs pkv = {}) {
    static_assert(!PVFP4 || UseBlockScaleMma, "PVFP4 rides on the block-scaled MMA path");
    static_assert(!Promote || UseBlockScaleMma, "Promote rides on the block-scaled MMA path");
    static_assert(!PagedKV || (UseBlockScaleMma && !PVFP4),
                  "PagedKV rides on the block-scaled MMA path with FP16 WMMA PV");
    constexpr int Bkv = MX_Bkv;
    constexpr int head_dim = HD;
    constexpr int hd_half = HD / 2;  // packed FP4 data bytes per row
    // Pad FP4 row stride by 4 bytes to avoid SMEM bank conflicts.
    // Without padding: stride=64 bytes for HD=128 → rows 0,2,4,6 map to same
    // bank set (64 = 16 banks, period 2). With +4: stride=68 → coprime with 32
    // banks, all rows unique.
    constexpr int hd_half_padded = hd_half + 4;

    // Threads-per-row for parallel softmax and quantization
    constexpr int TPR = MX_BLOCK_THREADS / Bq;
    static_assert(TPR >= 1 && (TPR & (TPR - 1)) == 0, "TPR must be power of 2");

    // ---- index computation --------------------------------------------------
    const int tile_q = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx = batch_head / n_heads;
    const int head_idx = batch_head % n_heads;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / MX_WARP_SIZE;
    const int lane_id = tid % MX_WARP_SIZE;
    const int q_start = tile_q * Bq;

    const int sm_row = tid / TPR;
    const int sm_lane = tid % TPR;

    const int64_t q_row_stride = (int64_t)n_heads * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;

    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                        (int64_t)head_idx * head_dim;
    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    half* O_ptr = O + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                  (int64_t)head_idx * head_dim;

    // ---- shared memory layout -----------------------------------------------
    //
    // Q_fp4:    uint8[Bq × HD/2]       — Q as packed FP4 nibbles (persistent)
    // q_scales: float[Bq]              — Q per-row scale
    // KV_buf:   char[Bkv × HD × 2]    — shared: FP4 K (HD/2 bytes/row) or FP16 V (HD×2 bytes/row)
    // k_scales: float[Bkv]            — K per-row scale (overwritten each KV tile)
    // S_tile:   float[Bq × Bkv]       — score tile (aliased as half P)
    // O_acc:    float[Bq × HD]        — output accumulator
    // row_m:    float[Bq]             — running max
    // row_l:    float[Bq]             — running sum
    //
    extern __shared__ char smem[];

    uint8_t* Q_fp4 = reinterpret_cast<uint8_t*>(smem);
    float* q_scales = reinterpret_cast<float*>(Q_fp4 + Bq * hd_half_padded);
    // KV_buf is aligned to 16 bytes for vectorized loads
    char* KV_raw = reinterpret_cast<char*>(q_scales + Bq);
    // Align KV_raw to 16 bytes
    KV_raw = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(KV_raw) + 15) & ~15ULL);
    uint8_t* KV_fp4 = reinterpret_cast<uint8_t*>(KV_raw);  // K as FP4
    half* KV_fp16 = reinterpret_cast<half*>(KV_raw);       // V as FP16 (same slot)
    float* k_scales = reinterpret_cast<float*>(KV_raw + Bkv * head_dim * sizeof(half));
    float* S_tile = k_scales + Bkv;
    float* O_acc = S_tile + Bq * Bkv;
    float* row_m = O_acc + Bq * head_dim;
    float* row_l = row_m + Bq;
    // Per-k_group UE4M3 scales for blockscale-MMA sfa/sfb operands.
    // Each row has n_k_groups = HD/16 scales (one per 16-K-block).
    // Only populated when UseBlockScaleMma=true. Placed at the tail.
    constexpr int n_k_groups = HD / 16;
    // PVFP4 buffers (sized 0-ish when off; kept in the pointer chain so the
    // layout matches compute_smem_mxfp4). All strides are multiples of 4 so
    // the uint32 sfa/sfb loads stay aligned. p_rowf goes first (float align).
    constexpr int n_kv_groups = Bkv / 16;              // P/V k-groups (KV dim)
    constexpr int kv_half_padded = Bkv / 2 + 4;        // packed P/V^T row bytes
    float* p_rowf = row_l + Bq;                        // [Bq] per-row PV post-factor
    float* p_rowq = p_rowf + (PVFP4 ? Bq : 0);         // [Bq] per-row P quant multiplier
    uint8_t* q_scales_fp8 = reinterpret_cast<uint8_t*>(p_rowq + (PVFP4 ? Bq : 0));
    uint8_t* k_scales_fp8 = q_scales_fp8 + Bq * n_k_groups;
    uint8_t* p_scales_fp8 = k_scales_fp8 + Bkv * n_k_groups;  // [Bq][n_kv_groups]
    uint8_t* v_scales_fp8 = p_scales_fp8 + Bq * n_kv_groups;  // [HD][n_kv_groups]
    uint8_t* P_fp4 = v_scales_fp8 + HD * n_kv_groups;         // [Bq][kv_half_padded]
    uint8_t* V_fp4T = P_fp4 + Bq * kv_half_padded;            // [HD][kv_half_padded]

    // K-smoothing: per-channel mean pointer for this (batch, kv_head).
    const float* kmean = (d_kmean != nullptr)
                             ? d_kmean + ((int64_t)batch_idx * n_kv_heads + kv_head) * head_dim
                             : nullptr;

    // Pre-compute sqrt(attention_scale) to absorb into Q and K scales (Opt 3).
    // S_true[i,j] = q_scale[i] * k_scale[j] * mma[i,j], and we want the result
    // pre-multiplied by attention_scale.  Split sqrt evenly: q_scales *= sqrt_scale,
    // k_scales *= sqrt_scale, so the product gives q*k*scale automatically.
    const float sqrt_scale = sqrtf(scale);
    // PagedKV: cache K scales are raw (absmax/6) — quantize Q raw too and
    // apply the full attention scale post-MMA in the fused store instead.
    const float q_sfold = PagedKV ? 1.0f : sqrt_scale;

    // Can we use S_tile as temporary staging for FP16 data? (Opt 1/4)
    // Requires: Bkv * HD * sizeof(half) <= Bq * Bkv * sizeof(float), i.e. HD <= 2*Bq
    constexpr bool can_stage_in_stile = (HD <= 2 * Bq);

    // ========================================================================
    // Phase 0: Quantize Q tile to FP4 E2M1 with per-row scale
    // ========================================================================
    if constexpr (can_stage_in_stile) {
        // Opt 4: Load Q FP16 → S_tile (as staging), then read from shared for both passes.
        half* Q_stage = reinterpret_cast<half*>(S_tile);
        {
            // float4 = 8 halves per iter (all supported HDs are multiples of 8)
            const int total_vec8 = (Bq * head_dim) / 8;
            for (int vi = tid; vi < total_vec8; vi += MX_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / head_dim;
                int d = i % head_dim;
                float4* dst = reinterpret_cast<float4*>(&Q_stage[i]);
                if (q_start + r < seq_q) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &Q_ptr[(int64_t)r * q_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        if constexpr (UseBlockScaleMma) {
            // Per-k_group absmax + UE4M3 encoding. Each thread owns one (row,
            // k_group) pair; 16 elements scanned directly.
            const int total_groups = Bq * n_k_groups;
            for (int idx = tid; idx < total_groups; idx += MX_BLOCK_THREADS) {
                int r = idx / n_k_groups;
                int kg = idx % n_k_groups;
                float absmax = 0.0f;
                const half* row_start = &Q_stage[r * head_dim + kg * 16];
#pragma unroll
                for (int i = 0; i < 16; i++)
                    absmax = fmaxf(absmax, fabsf(__half2float(row_start[i])));
                float raw = absmax / 6.0f;
                q_scales_fp8[r * n_k_groups + kg] = float_to_fp8_e4m3(raw * q_sfold);
            }
        } else {
            // Absmax from shared memory (fast L1 reads) — per-row (legacy)
            const int r = sm_row;
            float local_max = 0.0f;
            if (r < Bq) {
                for (int d = sm_lane; d < head_dim; d += TPR)
                    local_max = fmaxf(local_max, fabsf(__half2float(Q_stage[r * head_dim + d])));
            }
#pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
            if (sm_lane == 0 && r < Bq)
                q_scales[r] = local_max / 6.0f * sqrt_scale;
        }
        __syncthreads();

        // Quantize from shared → Q_fp4 (vectorized: 8 halves = 4 bytes/iter, uint32 store)
        // In blockscale mode: each 8-halves chunk (b4) falls in exactly one k_group
        // (kg = b4/2), so we dequant one UE4M3 byte per chunk and apply the resulting
        // inverse scale to all 8 values.
        {
            const int total_packed_u32 = (Bq * hd_half) / 4;
            for (int idx = tid; idx < total_packed_u32; idx += MX_BLOCK_THREADS) {
                int r = idx / (hd_half / 4);
                int b4 = idx % (hd_half / 4);
                int d = b4 * 8;
                float inv_scale;
                if constexpr (UseBlockScaleMma) {
                    int kg = b4 / 2;  // 8 halves fit inside a 16-half k_group
                    float dequant = fp8_e4m3_to_float_fast(q_scales_fp8[r * n_k_groups + kg]);
                    inv_scale = (dequant > 0.0f) ? (q_sfold / dequant) : 0.0f;
                } else {
                    inv_scale = (q_scales[r] > 0.0f) ? (sqrt_scale / q_scales[r]) : 0.0f;
                }
                const half* src = &Q_stage[r * head_dim + d];
                half2 h01 = reinterpret_cast<const half2*>(src)[0];
                half2 h23 = reinterpret_cast<const half2*>(src)[1];
                half2 h45 = reinterpret_cast<const half2*>(src)[2];
                half2 h67 = reinterpret_cast<const half2*>(src)[3];
                uint32_t b0 = pack_fp4_pair(__half2float(h01.x) * inv_scale, __half2float(h01.y) * inv_scale);
                uint32_t b1 = pack_fp4_pair(__half2float(h23.x) * inv_scale, __half2float(h23.y) * inv_scale);
                uint32_t b2 = pack_fp4_pair(__half2float(h45.x) * inv_scale, __half2float(h45.y) * inv_scale);
                uint32_t b3 = pack_fp4_pair(__half2float(h67.x) * inv_scale, __half2float(h67.y) * inv_scale);
                uint32_t packed = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                *reinterpret_cast<uint32_t*>(&Q_fp4[r * hd_half_padded + b4 * 4]) = packed;
            }
        }
    } else {
        // Fallback: 2-pass global reads (for Bq=32, HD>64)
        if constexpr (UseBlockScaleMma) {
            // Per-(row, k_group) absmax from global. Each thread owns one or
            // more (r, kg) pairs via a stride loop. No cross-thread reduction.
            const int total_groups = Bq * n_k_groups;
            for (int idx = tid; idx < total_groups; idx += MX_BLOCK_THREADS) {
                int r = idx / n_k_groups;
                int kg = idx % n_k_groups;
                float absmax = 0.0f;
                if (q_start + r < seq_q) {
                    const half* row = &Q_ptr[(int64_t)r * q_row_stride + kg * 16];
#pragma unroll
                    for (int i = 0; i < 16; i++)
                        absmax = fmaxf(absmax, fabsf(__half2float(row[i])));
                }
                float raw = absmax / 6.0f;
                q_scales_fp8[r * n_k_groups + kg] = float_to_fp8_e4m3(raw * q_sfold);
            }
        } else {
            const int r = sm_row;
            float local_max = 0.0f;
            if (r < Bq && q_start + r < seq_q) {
                for (int d = sm_lane; d < head_dim; d += TPR)
                    local_max = fmaxf(local_max, fabsf(__half2float(Q_ptr[(int64_t)r * q_row_stride + d])));
            }
#pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
            if (sm_lane == 0 && r < Bq)
                q_scales[r] = local_max / 6.0f * sqrt_scale;
        }
        __syncthreads();

        {
            const int total_packed = Bq * hd_half;
            for (int idx = tid; idx < total_packed; idx += MX_BLOCK_THREADS) {
                int r = idx / hd_half;
                int d_byte = idx % hd_half;
                int d = d_byte * 2;
                if (q_start + r < seq_q) {
                    float inv_scale;
                    if constexpr (UseBlockScaleMma) {
                        int kg = d / 16;
                        float dequant = fp8_e4m3_to_float_fast(q_scales_fp8[r * n_k_groups + kg]);
                        inv_scale = (dequant > 0.0f) ? (q_sfold / dequant) : 0.0f;
                    } else {
                        inv_scale = (q_scales[r] > 0.0f) ? (sqrt_scale / q_scales[r]) : 0.0f;
                    }
                    float v0 = __half2float(Q_ptr[(int64_t)r * q_row_stride + d]) * inv_scale;
                    float v1 = __half2float(Q_ptr[(int64_t)r * q_row_stride + d + 1]) * inv_scale;
                    Q_fp4[r * hd_half_padded + d_byte] = pack_fp4_pair(v0, v1);
                } else {
                    Q_fp4[r * hd_half_padded + d_byte] = 0;
                }
            }
        }
    }

    // Zero O accumulator + init softmax state
    {
        // float4 = 4 FP32 zeros per iter
        const int total_vec4 = (Bq * head_dim) / 4;
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int vi = tid; vi < total_vec4; vi += MX_BLOCK_THREADS) {
            reinterpret_cast<float4*>(O_acc)[vi] = zero;
        }
    }
    if (tid < Bq) {
        row_m[tid] = -FLT_MAX;
        row_l[tid] = 0.0f;
    }
    bool first_kv_iter = true;  // skip O_acc rescale on first tile (l_old=0 → rescale=0)
    __syncthreads();

    // ---- KV tile loop bounds ----
    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv, causal, sliding_window, first_kv_tile,
                           num_kv_tiles, q_offset);

    // FP4 MMA tiling: m16n8k32 E2M1
    constexpr int FP4_K = MX_MMA_K;              // 32
    constexpr int FP4_K_BYTES = FP4_K / 2;       // 16 bytes per k-chunk (packed nibbles)
    const int hd_chunks_fp4 = head_dim / FP4_K;  // k-loop iterations (same count as FP8)
    const int s_row_tiles = Bq / MX_MMA_M;
    const int s_col_tiles_half = Bkv / MX_MMA_N;  // each m16n8 tile
    const int s_total_tiles = s_row_tiles * s_col_tiles_half;

    // FP16 WMMA for P·V
    const int o_row_tiles = Bq / MX_WMMA_M;
    const int o_col_tiles = head_dim / MX_WMMA_N;
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks = Bkv / MX_WMMA_K;

    // ================================================================
    // Main KV tile loop
    // ================================================================
    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * Bkv;

        // #846 promotion: block-uniform per-tile flag. Promoted tiles skip the
        // K-quant phase and the FP4 QK MMA; Phase 1' computes exact scores.
        // PagedKV force-promotes CURRENT-CHUNK tiles (kv_start >= q_offset):
        // their K/V exist as fresh FP16 and quantizing the recency window is
        // where the quality damage lives (B-arm finding) — the past reads FP4
        // from the cache, the own chunk stays exact.
        bool promoted = false;
        if constexpr (Promote) {
            if (d_promote != nullptr)
                promoted = d_promote[((size_t)batch_head * gridDim.x + tile_q) * n_kv_tiles_mask + j] != 0;
            if constexpr (PagedKV)
                promoted = promoted || (kv_start >= q_offset);
        }

        // ---- Quantize K tile to FP4 (Opt 1: shared-memory staging) ----
        if (promoted) {
            // Promoted tile: no K-side quantization work.
        } else if constexpr (PagedKV) {
            // Paged-FP4 K (#846): copy packed nibbles + UE4M3 scales straight
            // from the NVFP4 KV cache into the blockscale-MMA smem layout —
            // the cache format is bit-compatible with what the in-kernel
            // quantizer produces, so this phase is pure data movement (the
            // 3.3× quant instruction overhead measured on the dense path
            // disappears). Zero bytes/scales for out-of-range rows: scale 0
            // makes the HW MMA contribute 0 and the fused store masks them.
            constexpr int kRowU32 = hd_half / 4;
            const int kv_row_bytes = n_kv_heads * hd_half;
            const int sc_row_bytes = n_kv_heads * n_k_groups;
            {
                const int total_u32 = Bkv * kRowU32;
                for (int idx = tid; idx < total_u32; idx += MX_BLOCK_THREADS) {
                    const int r = idx / kRowU32;
                    const int b4 = idx % kRowU32;
                    const int pos = kv_start + r;
                    uint32_t val = 0;
                    if (pos < seq_kv) {
                        const int blk = pkv.block_table[pos / pkv.block_size];
                        const int slot = pos % pkv.block_size;
                        const uint8_t* row = pkv.k_data +
                                             ((size_t)blk * pkv.block_size + slot) * kv_row_bytes +
                                             (size_t)kv_head * hd_half;
                        val = reinterpret_cast<const uint32_t*>(row)[b4];
                    }
                    *reinterpret_cast<uint32_t*>(&KV_fp4[r * hd_half_padded + b4 * 4]) = val;
                }
                const int total_sc = Bkv * n_k_groups;
                for (int idx = tid; idx < total_sc; idx += MX_BLOCK_THREADS) {
                    const int r = idx / n_k_groups;
                    const int kg = idx % n_k_groups;
                    const int pos = kv_start + r;
                    uint8_t sc = 0;
                    if (pos < seq_kv) {
                        const int blk = pkv.block_table[pos / pkv.block_size];
                        const int slot = pos % pkv.block_size;
                        sc = pkv.k_scales[((size_t)blk * pkv.block_size + slot) * sc_row_bytes +
                                          (size_t)kv_head * n_k_groups + kg];
                    }
                    k_scales_fp8[r * n_k_groups + kg] = sc;
                }
            }
            __syncthreads();
        } else if constexpr (can_stage_in_stile) {
            // Load K FP16 → S_tile (as staging buffer), then read from shared
            // float4 = 8 halves per iter
            half* K_stage = reinterpret_cast<half*>(S_tile);
            {
                const int total_vec8 = (Bkv * head_dim) / 8;
                for (int vi = tid; vi < total_vec8; vi += MX_BLOCK_THREADS) {
                    int i = vi * 8;
                    int r = i / head_dim;
                    int d = i % head_dim;
                    float4* dst = reinterpret_cast<float4*>(&K_stage[i]);
                    if (kv_start + r < seq_kv) {
                        const float4* src = reinterpret_cast<const float4*>(
                            &K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                        if (kmean == nullptr) {
                            *dst = *src;
                        } else {
                            // K-smoothing: stage K - mean(K) so absmax + quant
                            // both see the smoothed values.
                            float4 raw = *src;
                            half* h = reinterpret_cast<half*>(&raw);
#pragma unroll
                            for (int e = 0; e < 8; e++)
                                h[e] = __float2half(__half2float(h[e]) - kmean[d + e]);
                            *dst = raw;
                        }
                    } else {
                        *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
            }
            __syncthreads();

            if constexpr (UseBlockScaleMma) {
                // Per-k_group absmax for K tile
                const int total_groups = Bkv * n_k_groups;
                for (int idx = tid; idx < total_groups; idx += MX_BLOCK_THREADS) {
                    int r = idx / n_k_groups;
                    int kg = idx % n_k_groups;
                    float absmax = 0.0f;
                    const half* row_start = &K_stage[r * head_dim + kg * 16];
#pragma unroll
                    for (int i = 0; i < 16; i++)
                        absmax = fmaxf(absmax, fabsf(__half2float(row_start[i])));
                    float raw = absmax / 6.0f;
                    k_scales_fp8[r * n_k_groups + kg] = float_to_fp8_e4m3(raw * sqrt_scale);
                }
            } else {
                // Per-row absmax (legacy)
                const int r = sm_row;
                float local_max = 0.0f;
                if (r < Bkv) {
                    for (int d = sm_lane; d < head_dim; d += TPR)
                        local_max = fmaxf(local_max, fabsf(__half2float(K_stage[r * head_dim + d])));
                }
#pragma unroll
                for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
                if (sm_lane == 0 && r < Bkv)
                    k_scales[r] = local_max / 6.0f * sqrt_scale;
            }
            __syncthreads();

            // Quantize from shared → KV_fp4 (vectorized: 8 halves = 4 bytes/iter)
            {
                const int total_packed_u32 = (Bkv * hd_half) / 4;
                for (int idx = tid; idx < total_packed_u32; idx += MX_BLOCK_THREADS) {
                    int r = idx / (hd_half / 4);
                    int b4 = idx % (hd_half / 4);
                    int d = b4 * 8;
                    float inv_scale;
                    if constexpr (UseBlockScaleMma) {
                        int kg = b4 / 2;
                        float dequant = fp8_e4m3_to_float_fast(k_scales_fp8[r * n_k_groups + kg]);
                        inv_scale = (dequant > 0.0f) ? (sqrt_scale / dequant) : 0.0f;
                    } else {
                        inv_scale = (k_scales[r] > 0.0f) ? (sqrt_scale / k_scales[r]) : 0.0f;
                    }
                    const half* src = &K_stage[r * head_dim + d];
                    half2 h01 = reinterpret_cast<const half2*>(src)[0];
                    half2 h23 = reinterpret_cast<const half2*>(src)[1];
                    half2 h45 = reinterpret_cast<const half2*>(src)[2];
                    half2 h67 = reinterpret_cast<const half2*>(src)[3];
                    uint32_t b0 = pack_fp4_pair(__half2float(h01.x) * inv_scale,
                                                __half2float(h01.y) * inv_scale);
                    uint32_t b1 = pack_fp4_pair(__half2float(h23.x) * inv_scale,
                                                __half2float(h23.y) * inv_scale);
                    uint32_t b2 = pack_fp4_pair(__half2float(h45.x) * inv_scale,
                                                __half2float(h45.y) * inv_scale);
                    uint32_t b3 = pack_fp4_pair(__half2float(h67.x) * inv_scale,
                                                __half2float(h67.y) * inv_scale);
                    uint32_t packed = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                    *reinterpret_cast<uint32_t*>(&KV_fp4[r * hd_half_padded + b4 * 4]) = packed;
                }
            }
            __syncthreads();
        } else {
            // Fallback: 2-pass global reads
            if constexpr (UseBlockScaleMma) {
                // Per-(row, k_group) absmax from global
                const int total_groups = Bkv * n_k_groups;
                for (int idx = tid; idx < total_groups; idx += MX_BLOCK_THREADS) {
                    int r = idx / n_k_groups;
                    int kg = idx % n_k_groups;
                    float absmax = 0.0f;
                    if (kv_start + r < seq_kv) {
                        const half* row = &K_ptr[(int64_t)(kv_start + r) * kv_row_stride + kg * 16];
#pragma unroll
                        for (int i = 0; i < 16; i++) {
                            float kv = __half2float(row[i]);
                            if (kmean != nullptr)
                                kv -= kmean[kg * 16 + i];
                            absmax = fmaxf(absmax, fabsf(kv));
                        }
                    }
                    float raw = absmax / 6.0f;
                    k_scales_fp8[r * n_k_groups + kg] = float_to_fp8_e4m3(raw * sqrt_scale);
                }
            } else {
                const int r = sm_row;
                float local_max = 0.0f;
                if (r < Bkv && kv_start + r < seq_kv) {
                    for (int d = sm_lane; d < head_dim; d += TPR)
                        local_max = fmaxf(local_max,
                                          fabsf(__half2float(
                                              K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d])));
                }
#pragma unroll
                for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
                if (sm_lane == 0 && r < Bkv)
                    k_scales[r] = local_max / 6.0f * sqrt_scale;
            }
            __syncthreads();

            {
                const int total_packed = Bkv * hd_half;
                for (int idx = tid; idx < total_packed; idx += MX_BLOCK_THREADS) {
                    int r = idx / hd_half;
                    int d_byte = idx % hd_half;
                    int d = d_byte * 2;
                    if (kv_start + r < seq_kv) {
                        float inv_scale;
                        if constexpr (UseBlockScaleMma) {
                            int kg = d / 16;
                            float dequant = fp8_e4m3_to_float_fast(k_scales_fp8[r * n_k_groups + kg]);
                            inv_scale = (dequant > 0.0f) ? (sqrt_scale / dequant) : 0.0f;
                        } else {
                            inv_scale = (k_scales[r] > 0.0f) ? (sqrt_scale / k_scales[r]) : 0.0f;
                        }
                        float v0 = __half2float(K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                        float v1 = __half2float(K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d + 1]);
                        if (kmean != nullptr) {
                            v0 -= kmean[d];
                            v1 -= kmean[d + 1];
                        }
                        KV_fp4[r * hd_half_padded + d_byte] = pack_fp4_pair(v0 * inv_scale, v1 * inv_scale);
                    } else {
                        KV_fp4[r * hd_half_padded + d_byte] = 0;
                    }
                }
            }
            __syncthreads();
        }

        // ============================================================
        // Phase 1: S = Q_fp4 · K_fp4^T  using FP4 m16n8k32 MMA
        // ============================================================
        // kind::f8f6f4 m16n8k32 ALWAYS uses 4 A regs + 2 B regs (uniform encoding).
        // For E2M1 (FP4): a0/a1 hold real data (8 bytes = 16 FP4 per pair of rows),
        // a2/a3 = 0 (padding). b0 holds real data, b1 = 0 (padding).
        // Thread mapping: groupID = lane/4 (0-7), threadID = lane%4 (0-3).
        //   A: row = groupID (+8 for a1), k_offset = threadID * 4 bytes (8 FP4)
        //   B: col = groupID (0-7 for n=8), k_offset = threadID * 4 bytes
        // Iteration strategy:
        //   UseBlockScaleMma=true:  outer iterates over (ri, ci_meta) where each
        //     meta covers 4 consecutive ci values (16x32 output via 4 MMAs).
        //     A operand + sfa loaded once per k iteration and reused across the
        //     4 MMAs. Quarters Q_fp4 SMEM-A bandwidth in Phase 1.
        //   UseBlockScaleMma=false: original single-tile distribution.
        constexpr int CI_PER_META = 4;
        const int s_col_tile_metas = s_col_tiles_half / CI_PER_META;
        const int s_meta_total_tiles = s_row_tiles * s_col_tile_metas;
        const int outer_total = UseBlockScaleMma ? s_meta_total_tiles : s_total_tiles;

        if (Promote && promoted) {
            // ============================================================
            // Phase 1' (#846 promotion): exact FP32 scores from global FP16.
            // Scalar dots are slower than the MMA but exact — this is the
            // quality arm, not a perf path. With ksmooth active the FP4
            // tiles score Q·(K−mean)^T; the promoted tile MUST apply the
            // same shift, otherwise its columns sit on a different additive
            // offset inside the same softmax row (silent corruption).
            // ============================================================
            const int r = sm_row;  // TPR lanes cooperate per Q row
            const int lq = q_start + r;
            const int gq = q_offset + lq;
            const half* qrow = &Q_ptr[(int64_t)r * q_row_stride];
            for (int c = sm_lane; c < Bkv; c += TPR) {
                const int gk = kv_start + c;
                float s = -FLT_MAX;
                if (lq < seq_q && gk < seq_kv && !(causal && gq < gk) &&
                    !(sliding_window > 0 && (gq - gk) >= sliding_window)) {
                    float acc = 0.0f;
                    if constexpr (PagedKV) {
                      if (gk >= q_offset) {
                        // Current-chunk key: exact FP16 from the fresh chunk
                        // tensor (K arg holds rows [q_offset, seq_kv)).
                        const half* krow = &K_ptr[(int64_t)(gk - q_offset) * kv_row_stride];
                        for (int d = 0; d < head_dim; d += 2) {
                            const half2 qh = *reinterpret_cast<const half2*>(&qrow[d]);
                            const half2 kh = *reinterpret_cast<const half2*>(&krow[d]);
                            acc = fmaf(__half2float(qh.x), __half2float(kh.x), acc);
                            acc = fmaf(__half2float(qh.y), __half2float(kh.y), acc);
                        }
                      } else {
                        // Past key: exact FP32 arithmetic over the FP4-STORED
                        // cache K (dequant nibbles × UE4M3 group scale) — the
                        // best available K; storage quantization already
                        // happened at append time.
                        const int blk = pkv.block_table[gk / pkv.block_size];
                        const int slot = gk % pkv.block_size;
                        const uint8_t* krow = pkv.k_data +
                                              ((size_t)blk * pkv.block_size + slot) *
                                                  (n_kv_heads * hd_half) +
                                              (size_t)kv_head * hd_half;
                        const uint8_t* ksc = pkv.k_scales +
                                             ((size_t)blk * pkv.block_size + slot) *
                                                 (n_kv_heads * n_k_groups) +
                                             (size_t)kv_head * n_k_groups;
                        for (int d = 0; d < head_dim; d += 2) {
                            const half2 kh = unpack_fp4_pair(krow[d / 2]);
                            const float sc = fp8_e4m3_to_float_fast(ksc[d / 16]);
                            const half2 qh = *reinterpret_cast<const half2*>(&qrow[d]);
                            acc = fmaf(__half2float(qh.x), __half2float(kh.x) * sc, acc);
                            acc = fmaf(__half2float(qh.y), __half2float(kh.y) * sc, acc);
                        }
                      }
                    } else {
                        const half* krow = &K_ptr[(int64_t)gk * kv_row_stride];
                        for (int d = 0; d < head_dim; d += 2) {
                            const half2 qh = *reinterpret_cast<const half2*>(&qrow[d]);
                            const half2 kh = *reinterpret_cast<const half2*>(&krow[d]);
                            float k0 = __half2float(kh.x);
                            float k1 = __half2float(kh.y);
                            if (kmean != nullptr) {
                                k0 -= kmean[d];
                                k1 -= kmean[d + 1];
                            }
                            acc = fmaf(__half2float(qh.x), k0, acc);
                            acc = fmaf(__half2float(qh.y), k1, acc);
                        }
                    }
                    s = acc * scale;
                    if (softcap > 0.0f)
                        s = softcap * tanhf(s / softcap);
                }
                S_tile[r * Bkv + c] = s;
            }
        }
        for (int tile_idx = warp_id; !promoted && tile_idx < outer_total; tile_idx += MX_NUM_WARPS) {
            int ri, ci;
            int ci_meta_base = 0;
            if constexpr (UseBlockScaleMma) {
                ri = tile_idx / s_col_tile_metas;
                ci_meta_base = (tile_idx % s_col_tile_metas) * CI_PER_META;
                ci = ci_meta_base;  // first ci (used by code shared with legacy)
            } else {
                ri = tile_idx / s_col_tiles_half;
                ci = tile_idx % s_col_tiles_half;
            }

            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            // Extra accumulators for ci_meta_base + 1..3 (4-issue blockscale only)
            float d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
            float d8 = 0.0f, d9 = 0.0f, dA = 0.0f, dB = 0.0f;
            float dC = 0.0f, dD = 0.0f, dE = 0.0f, dF = 0.0f;

            const int group_id = lane_id / 4;
            const int thread_in_group = lane_id % 4;
            const int byte_offset = thread_in_group * 4;  // 4 bytes = 8 FP4 nibbles

            if constexpr (UseBlockScaleMma) {
                // New path: kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
                // Each issue consumes 64 K-elements (2× legacy). With uniform
                // scale=1.0 (sfa=sfb=0x38383838), the block-scale MMA reduces
                // to a plain E2M1 × E2M1 dot product — output is bit-equivalent
                // to two legacy m16n8k32 issues summed.
                // Register distribution (per CUTLASS ALayout/BLayout in
                // mma_traits_sm120.hpp:136, column-major (M,K) / (N,K)):
                //   a0: row[group_id],   k-stripe 2k
                //   a1: row[group_id+8], k-stripe 2k
                //   a2: row[group_id],   k-stripe 2k+1
                //   a3: row[group_id+8], k-stripe 2k+1
                //   b0: col[group_id],   k-stripe 2k
                //   b1: col[group_id],   k-stripe 2k+1
                const int k_pairs = hd_chunks_fp4 / 2;
                const int m_sfa = (lane_id / 4) + (lane_id % 2) * 8;
                const int n_sfb = lane_id / 4;
                constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
#define MXF4_MMA(d_a, d_b, d_c, d_d, b_lo, b_hi, sfb_)                                                      \
    asm volatile(                                                                                           \
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32."     \
        "ue4m3 "                                                                                            \
        "{%0, %1, %2, %3},"                                                                                 \
        "{%4, %5, %6, %7},"                                                                                 \
        "{%8, %9},"                                                                                         \
        "{%10, %11, %12, %13},"                                                                             \
        "{%14},"                                                                                            \
        "{%15, %16},"                                                                                       \
        "{%17},"                                                                                            \
        "{%18, %19};\n"                                                                                     \
        : "=f"(d_a), "=f"(d_b), "=f"(d_c), "=f"(d_d)                                                        \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b_lo), "r"(b_hi), "f"(d_a), "f"(d_b), "f"(d_c), "f"(d_d), \
          "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb_), "h"(bidB), "h"(tidB))

                for (int k = 0; k < k_pairs; k++) {
                    const int k0 = 2 * k;
                    const int k1 = k0 + 1;
                    const uint8_t* q_base0 = Q_fp4 + ri * MX_MMA_M * hd_half_padded + k0 * FP4_K_BYTES;
                    const uint8_t* q_base1 = Q_fp4 + ri * MX_MMA_M * hd_half_padded + k1 * FP4_K_BYTES;
                    // A loaded once, reused for all 4 ci's
                    uint32_t a0 = *reinterpret_cast<const uint32_t*>(q_base0 + group_id * hd_half_padded +
                                                                     byte_offset);
                    uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                        q_base0 + (group_id + 8) * hd_half_padded + byte_offset);
                    uint32_t a2 = *reinterpret_cast<const uint32_t*>(q_base1 + group_id * hd_half_padded +
                                                                     byte_offset);
                    uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                        q_base1 + (group_id + 8) * hd_half_padded + byte_offset);

                    // sfa shared across all 4 ci's
                    const int kg_base = k * 4;
                    uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                        &q_scales_fp8[(ri * MX_MMA_M + m_sfa) * n_k_groups + kg_base]);

// 4 MMAs for ci_meta_base + 0..3, all sharing A and sfa
#pragma unroll
                    for (int co = 0; co < CI_PER_META; co++) {
                        int ci_local = ci_meta_base + co;
                        const uint8_t* k_base0 = KV_fp4 + ci_local * MX_MMA_N * hd_half_padded +
                                                 k0 * FP4_K_BYTES;
                        const uint8_t* k_base1 = KV_fp4 + ci_local * MX_MMA_N * hd_half_padded +
                                                 k1 * FP4_K_BYTES;
                        uint32_t b0 = *reinterpret_cast<const uint32_t*>(k_base0 + group_id * hd_half_padded +
                                                                         byte_offset);
                        uint32_t b1 = *reinterpret_cast<const uint32_t*>(k_base1 + group_id * hd_half_padded +
                                                                         byte_offset);
                        uint32_t sfb = *reinterpret_cast<const uint32_t*>(
                            &k_scales_fp8[(ci_local * MX_MMA_N + n_sfb) * n_k_groups + kg_base]);
#if __CUDA_ARCH__ >= 1200
                        if (co == 0) {
                            MXF4_MMA(d0, d1, d2, d3, b0, b1, sfb);
                        } else if (co == 1) {
                            MXF4_MMA(d4, d5, d6, d7, b0, b1, sfb);
                        } else if (co == 2) {
                            MXF4_MMA(d8, d9, dA, dB, b0, b1, sfb);
                        } else {
                            MXF4_MMA(dC, dD, dE, dF, b0, b1, sfb);
                        }
#endif
                    }
                }
#undef MXF4_MMA
            } else {
                for (int k = 0; k < hd_chunks_fp4; k++) {
                    // Load A fragment: a0=row[groupID], a1=row[groupID+8], a2=a3=0
                    uint32_t a0, a1;
                    {
                        const uint8_t* q_base = Q_fp4 + ri * MX_MMA_M * hd_half_padded + k * FP4_K_BYTES;
                        a0 = *reinterpret_cast<const uint32_t*>(q_base + group_id * hd_half_padded +
                                                                byte_offset);
                        a1 = *reinterpret_cast<const uint32_t*>(q_base + (group_id + 8) * hd_half_padded +
                                                                byte_offset);
                    }
                    uint32_t a2 = 0, a3 = 0;  // padding for uniform register encoding

                    // Load B fragment: b0=col[groupID], b1=0
                    uint32_t b0;
                    {
                        const uint8_t* k_base = KV_fp4 + ci * MX_MMA_N * hd_half_padded + k * FP4_K_BYTES;
                        b0 = *reinterpret_cast<const uint32_t*>(k_base + group_id * hd_half_padded +
                                                                byte_offset);
                    }
                    uint32_t b1 = 0;  // padding

                    // FP4 E2M1 MMA: d += A_e2m1 × B_e2m1^T
#if __CUDA_ARCH__ >= 1200
                    asm volatile(
                        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
                        "{%0, %1, %2, %3},"
                        "{%4, %5, %6, %7},"
                        "{%8, %9},"
                        "{%10, %11, %12, %13};\n"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2),
                          "f"(d3));
#endif
                }
            }

            // Store 16×8 (or 16×16 for 2-issue blockscale) result to S_tile.
            // Fuses scale, softcap, causal mask, sliding window.
            {
                int base_row = ri * MX_MMA_M;
                int r0 = (lane_id / 4) % 8;
                int c0 = (lane_id % 4) * 2;

// lq = local Q row (bounds vs seq_q); gq = q_offset + lq (causal/SWA masks,
// chunked-prefill continuation).
#define FUSED_STORE(val, lq, gq, gk, qs, ks, idx)                       \
    do {                                                                \
        float s = (val) * (qs) * (ks);                                  \
        if (softcap > 0.0f)                                             \
            s = softcap * tanhf(s / softcap);                           \
        if ((lq) >= seq_q || (gk) >= seq_kv)                            \
            s = -FLT_MAX;                                               \
        else if (causal && (gq) < (gk))                                 \
            s = -FLT_MAX;                                               \
        else if (sliding_window > 0 && ((gq) - (gk)) >= sliding_window) \
            s = -FLT_MAX;                                               \
        S_tile[idx] = s;                                                \
    } while (0)

                int lq0 = q_start + base_row + r0;
                int lq1 = q_start + base_row + r0 + 8;
                int gq0 = q_offset + lq0;
                int gq1 = q_offset + lq1;
                // PagedKV: raw Q/K scales → apply the attention scale here.
                const float pg_qs = PagedKV ? scale : 1.0f;
                (void)pg_qs;

                if constexpr (UseBlockScaleMma) {
// 4-issue path: store 4 ci's, scales already HW-applied
#define STORE_CI(d_a, d_b, d_c, d_d, ci_off)                                                             \
    do {                                                                                                 \
        int base_col_x = (ci_meta_base + ci_off) * MX_MMA_N;                                             \
        int gk0_x = kv_start + base_col_x + c0;                                                          \
        int gk1_x = kv_start + base_col_x + c0 + 1;                                                      \
        FUSED_STORE(d_a, lq0, gq0, gk0_x, pg_qs, 1.0f, (base_row + r0) * Bkv + base_col_x + c0);         \
        FUSED_STORE(d_b, lq0, gq0, gk1_x, pg_qs, 1.0f, (base_row + r0) * Bkv + base_col_x + c0 + 1);     \
        FUSED_STORE(d_c, lq1, gq1, gk0_x, pg_qs, 1.0f, (base_row + r0 + 8) * Bkv + base_col_x + c0);     \
        FUSED_STORE(d_d, lq1, gq1, gk1_x, pg_qs, 1.0f, (base_row + r0 + 8) * Bkv + base_col_x + c0 + 1); \
    } while (0)
                    STORE_CI(d0, d1, d2, d3, 0);
                    STORE_CI(d4, d5, d6, d7, 1);
                    STORE_CI(d8, d9, dA, dB, 2);
                    STORE_CI(dC, dD, dE, dF, 3);
#undef STORE_CI
                } else {
                    // Legacy single-tile path
                    int base_col = ci * MX_MMA_N;
                    float qs0 = q_scales[base_row + r0];
                    float qs1 = q_scales[base_row + r0 + 8];
                    float ks0 = k_scales[base_col + c0];
                    float ks1 = k_scales[base_col + c0 + 1];
                    int gk0 = kv_start + base_col + c0;
                    int gk1 = kv_start + base_col + c0 + 1;
                    FUSED_STORE(d0, lq0, gq0, gk0, qs0, ks0, (base_row + r0) * Bkv + base_col + c0);
                    FUSED_STORE(d1, lq0, gq0, gk1, qs0, ks1, (base_row + r0) * Bkv + base_col + c0 + 1);
                    FUSED_STORE(d2, lq1, gq1, gk0, qs1, ks0, (base_row + r0 + 8) * Bkv + base_col + c0);
                    FUSED_STORE(d3, lq1, gq1, gk1, qs1, ks1, (base_row + r0 + 8) * Bkv + base_col + c0 + 1);
                }
#undef FUSED_STORE
            }
        }
        __syncthreads();

        // ============================================================
        // Prefetch V tile via cp.async (overlaps with softmax below).
        // KV_fp16 and S_tile are separate SMEM regions, so V writes
        // don't conflict with softmax reads/writes on S_tile.
        // PagedKV: V is dequantized FP16 into KV_fp16 from the paged
        // cache instead (indirect addressing — no cp.async).
        // ============================================================
        if constexpr (PagedKV) {
            const int total_bytes = Bkv * hd_half;  // one packed byte = 2 V elems
            for (int c = tid; c < total_bytes; c += MX_BLOCK_THREADS) {
                const int r = c / hd_half;
                const int b = c % hd_half;
                const int pos = kv_start + r;
                half2 out = __half2half2(__float2half(0.0f));
                if (pos >= q_offset && pos < seq_kv) {
                    // Current-chunk V: fresh FP16 (V arg holds the chunk rows).
                    out = *reinterpret_cast<const half2*>(
                        &V_ptr[(int64_t)(pos - q_offset) * kv_row_stride + 2 * b]);
                } else if (pos < seq_kv) {
                    const int blk = pkv.block_table[pos / pkv.block_size];
                    const int slot = pos % pkv.block_size;
                    const uint8_t* vrow = pkv.v_data +
                                          ((size_t)blk * pkv.block_size + slot) * (n_kv_heads * hd_half) +
                                          (size_t)kv_head * hd_half;
                    const uint8_t* vsc = pkv.v_scales +
                                         ((size_t)blk * pkv.block_size + slot) *
                                             (n_kv_heads * n_k_groups) +
                                         (size_t)kv_head * n_k_groups;
                    const half2 hh = unpack_fp4_pair(vrow[b]);
                    const float sc = fp8_e4m3_to_float_fast(vsc[(2 * b) / 16]);
                    out = __floats2half2_rn(__half2float(hh.x) * sc, __half2float(hh.y) * sc);
                }
                *reinterpret_cast<half2*>(&KV_fp16[r * head_dim + 2 * b]) = out;
            }
        } else {
            constexpr int CHUNK_HALVES = 8;  // 16 bytes = 8 halves per cp.async
            const int total_chunks = (Bkv * head_dim) / CHUNK_HALVES;
            for (int c = tid; c < total_chunks; c += MX_BLOCK_THREADS) {
                int elem = c * CHUNK_HALVES;
                int r = elem / head_dim;
                int d = elem % head_dim;
                if (kv_start + r < seq_kv) {
                    cp_async_ca_16(&KV_fp16[r * head_dim + d],
                                   &V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                } else {
                    // cp.async can't zero — use regular store
                    reinterpret_cast<float4*>(&KV_fp16[r * head_dim + d])[0] = make_float4(0.0f, 0.0f, 0.0f,
                                                                                           0.0f);
                }
            }
            cp_async_commit();
        }

        // ============================================================
        // Phase 2+3: Online softmax + convert P to FP16
        // (runs concurrently with V prefetch above)
        // ============================================================
        {
            half* SP_half = reinterpret_cast<half*>(S_tile);
            const int r = sm_row;
            const bool row_valid = (r < Bq) && (q_start + r < seq_q);

            // Step 1: Row max
            float partial_max = -FLT_MAX;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR)
                    partial_max = fmaxf(partial_max, S_tile[r * Bkv + c]);
            }
#pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                partial_max = fmaxf(partial_max, __shfl_xor_sync(0xffffffff, partial_max, offset));
            float m_ij = partial_max;

            // Step 2: Update running max + correction
            float m_old = row_valid ? row_m[r] : -FLT_MAX;
            float m_new = fmaxf(m_old, m_ij);
            float alpha = __expf(m_old - m_new);

            // Step 3: Exp + sum. Sentinel guard: fully-masked rows (m_new
            // stuck at -FLT_MAX) would turn every masked score into
            // expf(0) = 1 — map masked scores to 0 explicitly (mirrors the
            // guard in fmha_sm120_kernel / the FA2 kernel).
            float partial_sum = 0.0f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    float s_val = S_tile[r * Bkv + c];
                    float p = (s_val <= -FLT_MAX * 0.5f) ? 0.0f : __expf(s_val - m_new);
                    partial_sum += p;
                    S_tile[r * Bkv + c] = p;
                }
            }
#pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, offset);

            // Step 4: Update running sum
            float l_old = row_valid ? row_l[r] : 0.0f;
            float l_new = alpha * l_old + partial_sum;
            if (sm_lane == 0 && row_valid) {
                row_m[r] = m_new;
                row_l[r] = l_new;
            }

            // Step 5: Rescale O accumulator (skip on first tile: l_old=0 → rescale=0)
            if (!first_kv_iter) {
                float rescale = (l_old > 0.0f) ? (alpha * l_old / l_new) : 0.0f;
                if (row_valid) {
                    for (int d = sm_lane; d < head_dim; d += TPR)
                        O_acc[r * head_dim + d] *= rescale;
                }
            }

            // Promoted tiles always take the FP16 P path (their P·V runs on
            // the FP16 WMMA Phase 3 below, even under PVFP4). The condition
            // folds to a compile-time constant unless Promote is instantiated.
            if (!PVFP4 || (Promote && promoted)) {
                // Step 6: Normalize + float→half for P.
                // In-place float→half compaction: stage in registers + barrier
                // (SP_half row r aliases the bytes of float row r/2 — unsynced
                // stores clobber float scores other threads have not read yet,
                // issue #528; see attention_fmha_sm120.cu).
                constexpr int CPT = Bkv / TPR;
                float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
                half hbuf[CPT];
#pragma unroll
                for (int i = 0; i < CPT; i++) {
                    int c = sm_lane + i * TPR;
                    hbuf[i] = __float2half(row_valid ? S_tile[r * Bkv + c] * spv : 0.0f);
                }
                __syncthreads();  // all float reads of S_tile complete before any half write
#pragma unroll
                for (int i = 0; i < CPT; i++)
                    SP_half[r * Bkv + sm_lane + i * TPR] = hbuf[i];
            } else {
                // Step 6' (#846): two-level FP4 quantization of P.
                // Level 1: the row's tile-local max p is exp(m_ij - m_new);
                // rescale the row so that max lands at MX_PV_LEVEL1 (448·6).
                // Level 2: per-16 absmax → UE4M3 scale, nibbles vs dequantized
                // scale (mirrors the Q/K blockscale quant above). The inverse
                // row factor (including the 1/l_new normalization the legacy
                // path folds into P) is applied post-MMA via p_rowf.
                float rowmax_p = (m_ij <= -FLT_MAX * 0.5f) ? 0.0f : __expf(m_ij - m_new);
                if (sm_lane == 0 && r < Bq) {
                    const bool ok = row_valid && l_new > 0.0f && rowmax_p > 0.0f;
                    p_rowf[r] = ok ? rowmax_p / (MX_PV_LEVEL1 * l_new) : 0.0f;
                    p_rowq[r] = ok ? MX_PV_LEVEL1 / rowmax_p : 0.0f;
                }
                __syncthreads();  // p values (Step 3) + row factors visible block-wide

                const int total_groups = Bq * n_kv_groups;
                for (int idx = tid; idx < total_groups; idx += MX_BLOCK_THREADS) {
                    const int rr = idx / n_kv_groups;
                    const int kg = idx % n_kv_groups;
                    const float rq = p_rowq[rr];
                    const float* prow = &S_tile[rr * Bkv + kg * 16];
                    float bmax = 0.0f;  // p >= 0, no fabs needed
#pragma unroll
                    for (int i = 0; i < 16; i++)
                        bmax = fmaxf(bmax, prow[i]);
                    bmax *= rq;
                    const uint8_t sf = float_to_fp8_e4m3(bmax / 6.0f);
                    p_scales_fp8[rr * n_kv_groups + kg] = sf;
                    const float dq = fp8_e4m3_to_float_fast(sf);
                    const float inv = (dq > 0.0f) ? (rq / dq) : 0.0f;
                    uint8_t* dst = &P_fp4[rr * kv_half_padded + kg * 8];
#pragma unroll
                    for (int i = 0; i < 8; i++)
                        dst[i] = pack_fp4_pair(prow[2 * i] * inv, prow[2 * i + 1] * inv);
                }
                (void)SP_half;
            }
        }

        // Wait for V prefetch to complete, then sync all SMEM writes (P + V)
        if constexpr (!PagedKV)
            cp_async_wait_group<0>();
        __syncthreads();

        // Promoted tiles fall through to the FP16 WMMA Phase 3 (V is already
        // FP16 in KV_fp16); constant-folds unless Promote is instantiated.
        if (PVFP4 && !(Promote && promoted)) {
            // ============================================================
            // Phase 3' (#846): V^T per-16-block quant + FP4 P·V MMA
            // ============================================================
            // V^T quant: one (hd_col, kv_group) block = 16 consecutive KV
            // rows at one head_dim column; packed transposed so the PV
            // B-operand reads k-consecutive nibbles (KV is the MMA k-dim).
            {
                const int total_groups = head_dim * n_kv_groups;
                for (int idx = tid; idx < total_groups; idx += MX_BLOCK_THREADS) {
                    const int dcol = idx / n_kv_groups;
                    const int kg = idx % n_kv_groups;
                    float vals[16];
                    float bmax = 0.0f;
#pragma unroll
                    for (int i = 0; i < 16; i++) {
                        vals[i] = __half2float(KV_fp16[(kg * 16 + i) * head_dim + dcol]);
                        bmax = fmaxf(bmax, fabsf(vals[i]));
                    }
                    const uint8_t sf = float_to_fp8_e4m3(bmax / 6.0f);
                    v_scales_fp8[dcol * n_kv_groups + kg] = sf;
                    const float dq = fp8_e4m3_to_float_fast(sf);
                    const float inv = (dq > 0.0f) ? (1.0f / dq) : 0.0f;
                    uint8_t* dst = &V_fp4T[dcol * kv_half_padded + kg * 8];
#pragma unroll
                    for (int i = 0; i < 8; i++)
                        dst[i] = pack_fp4_pair(vals[2 * i] * inv, vals[2 * i + 1] * inv);
                }
            }
            __syncthreads();

            // FP4 P·V: m16n8k64, k = Bkv = 64 → exactly ONE block-scaled MMA
            // per 16×8 O tile. Same operand/scale register layout as the QK
            // blockscale path above ((T32,V32)→(M16,K64) per CUTLASS traits).
            static_assert(!PVFP4 || Bkv == 64, "PV MMA consumes k = 64 = Bkv in one issue");
            {
                const int pv_col_tiles = head_dim / MX_MMA_N;
                const int pv_total = (Bq / MX_MMA_M) * pv_col_tiles;
                const int group_id = lane_id / 4;
                const int byte_offset = (lane_id % 4) * 4;
                const int m_sfa = (lane_id / 4) + (lane_id % 2) * 8;
                const int n_sfb = lane_id / 4;
                constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
                for (int tile_idx = warp_id; tile_idx < pv_total; tile_idx += MX_NUM_WARPS) {
                    const int ri = tile_idx / pv_col_tiles;
                    const int ci = tile_idx % pv_col_tiles;
                    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                    const uint8_t* p_base = P_fp4 + ri * MX_MMA_M * kv_half_padded;
                    uint32_t a0 = *reinterpret_cast<const uint32_t*>(p_base + group_id * kv_half_padded +
                                                                     byte_offset);
                    uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                        p_base + (group_id + 8) * kv_half_padded + byte_offset);
                    uint32_t a2 = *reinterpret_cast<const uint32_t*>(p_base + group_id * kv_half_padded +
                                                                     16 + byte_offset);
                    uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                        p_base + (group_id + 8) * kv_half_padded + 16 + byte_offset);
                    const uint8_t* v_base = V_fp4T + (size_t)(ci * MX_MMA_N) * kv_half_padded;
                    uint32_t b0 = *reinterpret_cast<const uint32_t*>(v_base + group_id * kv_half_padded +
                                                                     byte_offset);
                    uint32_t b1 = *reinterpret_cast<const uint32_t*>(v_base + group_id * kv_half_padded +
                                                                     16 + byte_offset);
                    uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                        &p_scales_fp8[(ri * MX_MMA_M + m_sfa) * n_kv_groups]);
                    uint32_t sfb = *reinterpret_cast<const uint32_t*>(
                        &v_scales_fp8[(ci * MX_MMA_N + n_sfb) * n_kv_groups]);
#if __CUDA_ARCH__ >= 1200
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32."
                        "e2m1.e2m1.f32.ue4m3 "
                        "{%0, %1, %2, %3},"
                        "{%4, %5, %6, %7},"
                        "{%8, %9},"
                        "{%10, %11, %12, %13},"
                        "{%14},"
                        "{%15, %16},"
                        "{%17},"
                        "{%18, %19};\n"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2),
                          "f"(d3), "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB));
#endif
                    // Per-row two-level factor folds back the level-1 rescale
                    // AND the 1/l_new softmax normalization.
                    const int r0 = ri * MX_MMA_M + (lane_id / 4);
                    const int c0 = ci * MX_MMA_N + (lane_id % 4) * 2;
                    O_acc[r0 * head_dim + c0] += p_rowf[r0] * d0;
                    O_acc[r0 * head_dim + c0 + 1] += p_rowf[r0] * d1;
                    O_acc[(r0 + 8) * head_dim + c0] += p_rowf[r0 + 8] * d2;
                    O_acc[(r0 + 8) * head_dim + c0 + 1] += p_rowf[r0 + 8] * d3;
                }
            }
            first_kv_iter = false;
            __syncthreads();
            continue;
        }

        // ============================================================
        // Phase 3: O_acc += P · V using FP16 WMMA (m16n16k16)
        // ============================================================
        {
            half* P_half = reinterpret_cast<half*>(S_tile);
            for (int tile_idx = warp_id; tile_idx < o_total_tiles; tile_idx += MX_NUM_WARPS) {
                int ri = tile_idx / o_col_tiles;
                int di = tile_idx % o_col_tiles;

                wmma::fragment<wmma::accumulator, MX_WMMA_M, MX_WMMA_N, MX_WMMA_K, float> o_frag;
                wmma::load_matrix_sync(o_frag, O_acc + ri * MX_WMMA_M * head_dim + di * MX_WMMA_N, head_dim,
                                       wmma::mem_row_major);

                for (int k = 0; k < pv_chunks; k++) {
                    wmma::fragment<wmma::matrix_a, MX_WMMA_M, MX_WMMA_N, MX_WMMA_K, half, wmma::row_major>
                        p_frag;
                    wmma::load_matrix_sync(p_frag, P_half + ri * MX_WMMA_M * Bkv + k * MX_WMMA_K, Bkv);

                    wmma::fragment<wmma::matrix_b, MX_WMMA_M, MX_WMMA_N, MX_WMMA_K, half, wmma::row_major>
                        v_frag;
                    wmma::load_matrix_sync(v_frag, KV_fp16 + k * MX_WMMA_N * head_dim + di * MX_WMMA_N,
                                           head_dim);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(O_acc + ri * MX_WMMA_M * head_dim + di * MX_WMMA_N, o_frag, head_dim,
                                        wmma::mem_row_major);
            }
        }
        first_kv_iter = false;
        __syncthreads();
    }

    // ---- Write final output (vectorized: 4 FP32 → 4 FP16 per iter) ----
    {
        const int total_vec4 = (Bq * head_dim) / 4;
        for (int vi = tid; vi < total_vec4; vi += MX_BLOCK_THREADS) {
            int i = vi * 4;
            int r = i / head_dim;
            if (q_start + r >= seq_q)
                continue;
            float4 v = reinterpret_cast<const float4*>(O_acc)[vi];
            half2 lo = __float22half2_rn(make_float2(v.x, v.y));
            half2 hi = __float22half2_rn(make_float2(v.z, v.w));
            uint2 packed;
            packed.x = *reinterpret_cast<const uint32_t*>(&lo);
            packed.y = *reinterpret_cast<const uint32_t*>(&hi);
            *reinterpret_cast<uint2*>(&O_ptr[(int64_t)r * q_row_stride + (i % head_dim)]) = packed;
        }
    }
}

// =============================================================================
// Shared memory computation
// =============================================================================

static size_t compute_smem_mxfp4(int Bq, int Bkv, int head_dim, bool pv_fp4 = false) {
    size_t q_fp4 = (size_t)Bq * (head_dim / 2 + 4);         // Q packed FP4 (padded stride)
    size_t q_scales = (size_t)Bq * sizeof(float);           // Q row scales
    size_t align = 16;                                      // alignment padding
    size_t kv_buf = (size_t)Bkv * head_dim * sizeof(half);  // KV (FP4 K or FP16 V)
    size_t k_scales = (size_t)Bkv * sizeof(float);          // K row scales
    size_t s_tile = (size_t)Bq * Bkv * sizeof(float);       // S_tile
    size_t o_acc = (size_t)Bq * head_dim * sizeof(float);   // O_acc
    size_t softmax = 2 * (size_t)Bq * sizeof(float);        // row_m + row_l
    int n_k_groups = head_dim / 16;
    size_t scales_fp8 = (size_t)(Bq + Bkv) * n_k_groups;  // UE4M3 per-16-K scales (blockscale)
    size_t pv = 0;
    if (pv_fp4) {
        const int n_kv_groups = Bkv / 16;
        const int kv_half_padded = Bkv / 2 + 4;
        pv = 2 * (size_t)Bq * sizeof(float)          // p_rowf + p_rowq
             + (size_t)Bq * n_kv_groups              // P scales
             + (size_t)head_dim * n_kv_groups        // V scales
             + (size_t)Bq * kv_half_padded           // P_fp4
             + (size_t)head_dim * kv_half_padded;    // V_fp4T
    }
    return q_fp4 + q_scales + align + kv_buf + k_scales + s_tile + o_acc + softmax + scales_fp8 + pv;
}

// =============================================================================
// K per-channel mean pre-pass (#846 smoothing)
// =============================================================================
// One block per (batch, kv_head); threads stride over head_dim channels, each
// summing seq_kv rows (consecutive threads read consecutive channels →
// coalesced per row). Output: mean[batch * n_kv_heads + kv_head][head_dim].
__global__ void mxfp4_k_channel_mean_kernel(const half* __restrict__ K, float* __restrict__ mean,
                                            int seq_kv, int n_kv_heads, int head_dim) {
    const int bh = blockIdx.x;
    const int batch = bh / n_kv_heads;
    const int kvh = bh % n_kv_heads;
    const int64_t row_stride = (int64_t)n_kv_heads * head_dim;
    const half* base = K + (int64_t)batch * seq_kv * row_stride + (int64_t)kvh * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < seq_kv; r++)
            s += __half2float(base[(int64_t)r * row_stride + d]);
        mean[(int64_t)bh * head_dim + d] = s / (float)seq_kv;
    }
}

// =============================================================================
// Promotion pre-pass (#846 ThriftAttention outlier selection)
// =============================================================================
// Generic per-tile channel mean over [batch, seq, heads, head_dim] FP16 input.
// One block per (tile, batch*head); threads stride head_dim (coalesced per
// row). Output: mean[(bh * gridDim.x + tile) * head_dim + d]. Serves both Q̄
// (tile_rows = Bq) and K̄ (tile_rows = MX_Bkv).
__global__ void mxfp4_tile_mean_kernel(const half* __restrict__ X, float* __restrict__ mean, int seq,
                                       int n_heads_x, int head_dim, int tile_rows) {
    const int tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int batch = bh / n_heads_x;
    const int h = bh % n_heads_x;
    const int64_t row_stride = (int64_t)n_heads_x * head_dim;
    const int r0 = tile * tile_rows;
    const int rows = min(tile_rows, seq - r0);
    const half* base = X + (int64_t)batch * seq * row_stride + (int64_t)r0 * row_stride +
                       (int64_t)h * head_dim;
    const float inv = 1.0f / (float)rows;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < rows; r++)
            s += __half2float(base[(int64_t)r * row_stride + d]);
        mean[((int64_t)bh * gridDim.x + tile) * head_dim + d] = s * inv;
    }
}

// Paged variant of the K̄ tile mean: decodes FP4 nibbles × UE4M3 group scales
// straight from the NVFP4 KV cache (single sequence, batch = 1). One block
// per (kv_tile, kv_head); output layout matches mxfp4_tile_mean_kernel with
// bh = kv_head.
__global__ void mxfp4_tile_mean_paged_kernel(const uint8_t* __restrict__ k_data,
                                             const uint8_t* __restrict__ k_scales,
                                             const int* __restrict__ block_table, int block_size,
                                             int seq_kv, int n_kv_heads, int head_dim, int tile_rows,
                                             float* __restrict__ mean) {
    const int tile = blockIdx.x;
    const int kvh = blockIdx.y;
    const int r0 = tile * tile_rows;
    const int rows = min(tile_rows, seq_kv - r0);
    const int n_groups = head_dim / 16;
    const int row_bytes = n_kv_heads * (head_dim / 2);
    const int sc_row_bytes = n_kv_heads * n_groups;
    const float inv = 1.0f / (float)rows;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < rows; r++) {
            const int pos = r0 + r;
            const int blk = block_table[pos / block_size];
            const int slot = pos % block_size;
            const uint8_t byte = k_data[((size_t)blk * block_size + slot) * row_bytes +
                                        (size_t)kvh * (head_dim / 2) + d / 2];
            const half2 hh = unpack_fp4_pair(byte);
            const float sc = fp8_e4m3_to_float_fast(
                k_scales[((size_t)blk * block_size + slot) * sc_row_bytes + (size_t)kvh * n_groups +
                         d / 16]);
            s += __half2float((d & 1) ? hh.y : hh.x) * sc;
        }
        mean[((int64_t)kvh * gridDim.x + tile) * head_dim + d] = s * inv;
    }
}

// Top-k KV-tile selection per (batch_head, q_tile): importance score
// Ŝ_j = Q̄_i · K̄_j (ThriftAttention block-mean heuristic), budget = fraction
// of the causally visible tiles, sink tile (j=0) and diagonal tile (j=last)
// force-included within the budget. Sliding-window occlusion is ignored here
// (a promoted-but-masked tile is wasted work, never wrong). Dynamic smem:
// head_dim + n_kv_tiles floats. Writes a uint8 mask row [n_kv_tiles].
__global__ void mxfp4_promote_select_kernel(const float* __restrict__ qmean,
                                            const float* __restrict__ kmean_tiles,
                                            uint8_t* __restrict__ mask, int n_heads, int n_kv_heads,
                                            int n_q_tiles, int n_kv_tiles, int bq, int seq_q, int q_offset,
                                            bool causal, float budget, int head_dim) {
    const int q_tile = blockIdx.x;
    const int bh = blockIdx.y;  // batch * n_heads + head
    const int batch = bh / n_heads;
    const int head = bh % n_heads;
    const int kvh = head / (n_heads / n_kv_heads);
    const int bkvh = batch * n_kv_heads + kvh;

    extern __shared__ float sel_sm[];
    float* qm = sel_sm;                 // [head_dim]
    float* score = sel_sm + head_dim;   // [n_kv_tiles]
    __shared__ float red_v[128];
    __shared__ int red_j[128];

    const float* qmv = qmean + ((int64_t)bh * n_q_tiles + q_tile) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        qm[d] = qmv[d];

    int last = n_kv_tiles - 1;
    if (causal) {
        const int gq_max = q_offset + min(seq_q, (q_tile + 1) * bq) - 1;
        last = min(last, gq_max / MX_Bkv);
        last = max(last, 0);
    }

    uint8_t* mrow = mask + ((int64_t)bh * n_q_tiles + q_tile) * n_kv_tiles;
    for (int j = threadIdx.x; j < n_kv_tiles; j += blockDim.x)
        mrow[j] = 0;
    __syncthreads();

    for (int j = threadIdx.x; j <= last; j += blockDim.x) {
        const float* km = kmean_tiles + ((int64_t)bkvh * n_kv_tiles + j) * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++)
            s = fmaf(qm[d], km[d], s);
        score[j] = s;
    }
    __syncthreads();

    int k = (int)ceilf(budget * (float)(last + 1));
    k = max(k, 1);
    k = min(k, last + 1);
    if (threadIdx.x == 0) {
        mrow[0] = 1;
        mrow[last] = 1;
        score[0] = -FLT_MAX;
        score[last] = -FLT_MAX;
    }
    __syncthreads();

    // Iterative block-wide argmax for the remaining budget.
    for (int sel = (last == 0) ? 1 : 2; sel < k; sel++) {
        __syncthreads();  // break-check reads of red_j[0] complete before rewrite
        float bv = -FLT_MAX;
        int bj = -1;
        for (int j = threadIdx.x; j <= last; j += blockDim.x) {
            if (score[j] > bv) {
                bv = score[j];
                bj = j;
            }
        }
        red_v[threadIdx.x] = bv;
        red_j[threadIdx.x] = bj;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int t = 1; t < blockDim.x; t++) {
                if (red_v[t] > red_v[0]) {
                    red_v[0] = red_v[t];
                    red_j[0] = red_j[t];
                }
            }
            if (red_j[0] >= 0) {
                mrow[red_j[0]] = 1;
                score[red_j[0]] = -FLT_MAX;
            }
        }
        __syncthreads();
        if (red_j[0] < 0)
            break;  // all visible tiles already selected (uniform read)
    }
}

// =============================================================================
// Host launcher
// =============================================================================

// Persistent grow-only device scratches, lazily created in the two host
// launchers below (file-scope so the reset hook can free them).
// Dense-prefill launcher (fmha_sm120_mxfp4_prefill):
static float* s_d_kmean = nullptr;    // K-smoothing per-channel means
static size_t s_kmean_cap = 0;
static float* s_d_means = nullptr;    // promotion pre-pass: Q̄/K̄ tile means
static size_t s_means_cap = 0;
static uint8_t* s_d_promote = nullptr;  // promotion mask
static size_t s_promote_cap = 0;
// Paged-KV launcher (fmha_sm120_mxfp4_prefill_paged):
static float* s_d_means_paged = nullptr;
static size_t s_means_paged_cap = 0;
static uint8_t* s_d_promote_paged = nullptr;
static size_t s_promote_paged_cap = 0;

// Pre-cudaDeviceReset hook (see core/cuda_static_reset.h).
void fmha_mxfp4_reset_static_cuda_state() {
    auto free_f = [](float*& p, size_t& cap) {
        if (p) {
            (void)cudaFree(p);
            p = nullptr;
        }
        cap = 0;
    };
    auto free_u8 = [](uint8_t*& p, size_t& cap) {
        if (p) {
            (void)cudaFree(p);
            p = nullptr;
        }
        cap = 0;
    };
    free_f(s_d_kmean, s_kmean_cap);
    free_f(s_d_means, s_means_cap);
    free_u8(s_d_promote, s_promote_cap);
    free_f(s_d_means_paged, s_means_paged_cap);
    free_u8(s_d_promote_paged, s_promote_paged_cap);
}

bool fmha_sm120_mxfp4_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                              bool causal, int sliding_window, float softcap, cudaStream_t stream,
                              bool use_blockscale, int q_offset) {
    if (Q.qtype != QType::F16)
        return false;
    // Blockscale MMA operates on K=64 per issue; head_dim must be a multiple of 64.
    // head_dim=96 (Gemma-class) is a multiple of 32 but NOT 64 → fall back to legacy.
    if (use_blockscale && (static_cast<int>(Q.shape[3]) % 64 != 0)) {
        use_blockscale = false;
    }

    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q = static_cast<int>(Q.shape[1]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int seq_kv = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0)
        return false;
    if (seq_q == 0 || seq_kv == 0)
        return false;
    if (head_dim % MX_MMA_K != 0)
        return false;  // FP4 MMA needs head_dim % 32 == 0

    // #846 recipe knobs (blockscale only). K-smoothing needs softcap == 0:
    // the dropped Q·mean^T term only cancels under a pure shift-invariant
    // softmax — tanh softcapping breaks that.
    const bool pv_fp4 = use_blockscale && process_diag_mxfp4_pv_fp4();
    const bool ksmooth = use_blockscale && process_diag_mxfp4_ksmooth() && softcap == 0.0f;

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Select Bq: prefer larger tiles, fall back for larger HD
    int Bq;
    {
        size_t smem_128 = compute_smem_mxfp4(128, MX_Bkv, head_dim, pv_fp4);
        size_t smem_64 = compute_smem_mxfp4(64, MX_Bkv, head_dim, pv_fp4);
        size_t smem_32 = compute_smem_mxfp4(32, MX_Bkv, head_dim, pv_fp4);
        if (smem_128 <= (size_t)max_smem) {
            Bq = 128;
        } else if (smem_64 <= (size_t)max_smem) {
            Bq = 64;
        } else if (smem_32 <= (size_t)max_smem) {
            Bq = 32;
        } else {
            IMP_LOG_DEBUG("FMHA MXFP4: no Bq fits smem (hd=%d, smem_32=%zu, max=%d)", head_dim, smem_32,
                          max_smem);
            return false;
        }
    }

    const size_t smem = compute_smem_mxfp4(Bq, MX_Bkv, head_dim, pv_fp4);

    // K-smoothing pre-pass: per-(batch, kv_head, channel) mean into a
    // persistent grow-only scratch (process lifetime, tiny: bh × hd floats).
    const float* d_kmean = nullptr;
    if (ksmooth) {
        const size_t need = (size_t)batch_size * n_kv_heads * head_dim;
        if (need > s_kmean_cap) {
            if (s_d_kmean != nullptr)
                cudaFree(s_d_kmean);
            if (cudaMalloc(&s_d_kmean, need * sizeof(float)) != cudaSuccess) {
                s_d_kmean = nullptr;
                s_kmean_cap = 0;
            } else {
                s_kmean_cap = need;
            }
        }
        if (s_d_kmean != nullptr) {
            mxfp4_k_channel_mean_kernel<<<batch_size * n_kv_heads, 128, 0, stream>>>(
                reinterpret_cast<const half*>(K.data), s_d_kmean, seq_kv, n_kv_heads, head_dim);
            IMP_CUDA_CHECK_LAUNCH();
            d_kmean = s_d_kmean;
        }
    }

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;

    // #846 ThriftAttention promotion pre-pass: block means → top-k mask.
    // Grow-only static scratch (process lifetime), same pattern as s_d_kmean.
    // Promote instantiations exist only for head_dim 64/128 (bounds ptxas
    // time; the PPL target Qwen3-14B is hd=128).
    const float promote_budget = use_blockscale ? process_diag_mxfp4_promote_budget() : 0.0f;
    const int n_kv_tiles = (seq_kv + MX_Bkv - 1) / MX_Bkv;
    const uint8_t* d_promote = nullptr;
    if (promote_budget > 0.0f && (head_dim == 64 || head_dim == 128)) {
        // s_d_means: Q̄ tile means followed by K̄ tile means (file-scope above)
        const size_t qmean_elems = (size_t)batch_size * n_heads * num_q_tiles * head_dim;
        const size_t kmean_elems = (size_t)batch_size * n_kv_heads * n_kv_tiles * head_dim;
        const size_t means_need = qmean_elems + kmean_elems;
        const size_t mask_need = (size_t)batch_size * n_heads * num_q_tiles * n_kv_tiles;
        if (means_need > s_means_cap) {
            if (s_d_means != nullptr)
                cudaFree(s_d_means);
            if (cudaMalloc(&s_d_means, means_need * sizeof(float)) != cudaSuccess) {
                s_d_means = nullptr;
                s_means_cap = 0;
            } else {
                s_means_cap = means_need;
            }
        }
        if (mask_need > s_promote_cap) {
            if (s_d_promote != nullptr)
                cudaFree(s_d_promote);
            if (cudaMalloc(&s_d_promote, mask_need) != cudaSuccess) {
                s_d_promote = nullptr;
                s_promote_cap = 0;
            } else {
                s_promote_cap = mask_need;
            }
        }
        if (s_d_means != nullptr && s_d_promote != nullptr) {
            float* d_qmean = s_d_means;
            float* d_kmean_tiles = s_d_means + qmean_elems;
            dim3 qmg(num_q_tiles, batch_size * n_heads);
            mxfp4_tile_mean_kernel<<<qmg, 128, 0, stream>>>(reinterpret_cast<const half*>(Q.data), d_qmean,
                                                            seq_q, n_heads, head_dim, Bq);
            IMP_CUDA_CHECK_LAUNCH();
            dim3 kmg(n_kv_tiles, batch_size * n_kv_heads);
            mxfp4_tile_mean_kernel<<<kmg, 128, 0, stream>>>(reinterpret_cast<const half*>(K.data),
                                                            d_kmean_tiles, seq_kv, n_kv_heads, head_dim,
                                                            MX_Bkv);
            IMP_CUDA_CHECK_LAUNCH();
            dim3 sg(num_q_tiles, batch_size * n_heads);
            const size_t sel_smem = (size_t)(head_dim + n_kv_tiles) * sizeof(float);
            mxfp4_promote_select_kernel<<<sg, 128, sel_smem, stream>>>(
                d_qmean, d_kmean_tiles, s_d_promote, n_heads, n_kv_heads, num_q_tiles, n_kv_tiles, Bq, seq_q,
                q_offset, causal, promote_budget, head_dim);
            IMP_CUDA_CHECK_LAUNCH();
            d_promote = s_d_promote;
        }
    }

    // Engagement log (INFO once): the quality gate is only meaningful if this
    // kernel actually serves prefill — see the #511/#656 vacuous-closure lesson.
    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        IMP_LOG_INFO(
            "FMHA MXFP4 sm120 ACTIVE (hd=%d, Bq=%d, blockscale=%d, ksmooth=%d, pv_fp4=%d, promote=%.2f)",
            head_dim, Bq, (int)use_blockscale, (int)(d_kmean != nullptr), (int)pv_fp4,
            (d_promote != nullptr) ? promote_budget : 0.0f);
    }

    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(MX_WARP_SIZE, MX_NUM_WARPS);

    IMP_LOG_DEBUG(
        "FMHA MXFP4 sm120: B=%d Sq=%d Skv=%d nh=%d nkv=%d hd=%d Bq=%d Bkv=%d smem=%zu "
        "causal=%d sw=%d softcap=%.1f",
        batch_size, seq_q, seq_kv, n_heads, n_kv_heads, head_dim, Bq, MX_Bkv, smem, causal, sliding_window,
        softcap);

#define LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, BS, PV, PR)                                                         \
    do {                                                                                                   \
        cudaError_t attr_err = cudaFuncSetAttribute(fmha_sm120_mxfp4_kernel<BQ, HD, BS, PV, PR>,           \
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,           \
                                                    static_cast<int>(smem));                               \
        if (attr_err != cudaSuccess) {                                                                     \
            IMP_LOG_WARN("FMHA MXFP4: cudaFuncSetAttribute failed Bq=%d HD=%d bs=%d smem=%zu: %s", BQ, HD, \
                         (int)BS, smem, cudaGetErrorString(attr_err));                                     \
            return false;                                                                                  \
        }                                                                                                  \
        cudaFuncSetAttribute(fmha_sm120_mxfp4_kernel<BQ, HD, BS, PV, PR>,                                  \
                             cudaFuncAttributePreferredSharedMemoryCarveout,                               \
                             cudaSharedmemCarveoutMaxShared);                                              \
        fmha_sm120_mxfp4_kernel<BQ, HD, BS, PV, PR>                                                        \
            <<<grid, block, smem, stream>>>(reinterpret_cast<const half*>(Q.data),                         \
                                            reinterpret_cast<const half*>(K.data),                         \
                                            reinterpret_cast<const half*>(V.data),                         \
                                            reinterpret_cast<half*>(O.data), batch_size, seq_q, seq_kv,    \
                                            n_heads, n_kv_heads, scale, causal, sliding_window, softcap,   \
                                            q_offset, d_kmean, d_promote, n_kv_tiles);                     \
        IMP_CUDA_CHECK_LAUNCH();                                                                           \
    } while (0)

// Promote variants only instantiated for head_dim 64/128 (see the pre-pass
// gate above) — HD 96/256 use the no-promote macro to bound ptxas time.
#define LAUNCH_FMHA_MXFP4(BQ, HD)                                \
    do {                                                         \
        if (d_promote != nullptr && pv_fp4)                      \
            LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, true, true, true);    \
        else if (d_promote != nullptr)                           \
            LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, true, false, true);   \
        else if (pv_fp4)                                         \
            LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, true, true, false);   \
        else if (use_blockscale)                                 \
            LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, true, false, false);  \
        else                                                     \
            LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, false, false, false); \
    } while (0)

#define LAUNCH_FMHA_MXFP4_NOPROMOTE(BQ, HD)                      \
    do {                                                         \
        if (pv_fp4)                                              \
            LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, true, true, false);   \
        else if (use_blockscale)                                 \
            LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, true, false, false);  \
        else                                                     \
            LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, false, false, false); \
    } while (0)

    if (Bq == 128) {
        switch (head_dim) {
            case 64:
                LAUNCH_FMHA_MXFP4(128, 64);
                return true;
            case 96:
                LAUNCH_FMHA_MXFP4_NOPROMOTE(128, 96);
                return true;
            case 128:
                LAUNCH_FMHA_MXFP4(128, 128);
                return true;
            case 256:
                LAUNCH_FMHA_MXFP4_NOPROMOTE(128, 256);
                return true;
            default:
                break;
        }
    } else if (Bq == 64) {
        switch (head_dim) {
            case 64:
                LAUNCH_FMHA_MXFP4(64, 64);
                return true;
            case 96:
                LAUNCH_FMHA_MXFP4_NOPROMOTE(64, 96);
                return true;
            case 128:
                LAUNCH_FMHA_MXFP4(64, 128);
                return true;
            case 256:
                LAUNCH_FMHA_MXFP4_NOPROMOTE(64, 256);
                return true;
            default:
                break;
        }
    } else {
        switch (head_dim) {
            case 64:
                LAUNCH_FMHA_MXFP4(32, 64);
                return true;
            case 96:
                LAUNCH_FMHA_MXFP4_NOPROMOTE(32, 96);
                return true;
            case 128:
                LAUNCH_FMHA_MXFP4(32, 128);
                return true;
            case 256:
                LAUNCH_FMHA_MXFP4_NOPROMOTE(32, 256);
                return true;
            default:
                break;
        }
    }

#undef LAUNCH_FMHA_MXFP4
#undef LAUNCH_FMHA_MXFP4_NOPROMOTE
#undef LAUNCH_FMHA_MXFP4_IMPL
    return false;
}

// =============================================================================
// Paged-FP4-K launcher (#846 KV-append-quant path)
// =============================================================================
// Q: fresh chunk [1, seq_q, n_heads, hd] FP16 (post-RoPE). K/V come straight
// from the NVFP4 paged cache covering [0, seq_kv) — the current chunk must
// already be appended (write_kv_cache BEFORE attention). hd=128 only (the
// only instantiated PagedKV head_dim; the PPL target Qwen3-14B).
bool fmha_sm120_mxfp4_prefill_paged(const Tensor& Q, Tensor& O, const half* k_fresh,
                                    const half* v_fresh, const uint8_t* k_data,
                                    const uint8_t* k_scales, const uint8_t* v_data,
                                    const uint8_t* v_scales, const int* block_table, int block_size,
                                    int seq_kv, int n_kv_heads, float scale, bool causal,
                                    int sliding_window, float softcap, cudaStream_t stream, int q_offset,
                                    float promote_budget) {
    if (Q.qtype != QType::F16)
        return false;
    // Current-chunk tiles read the fresh FP16 K/V and are force-promoted;
    // q_offset must be tile-aligned so no tile straddles the boundary.
    if (k_fresh == nullptr || v_fresh == nullptr || q_offset <= 0 || (q_offset % MX_Bkv) != 0)
        return false;
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q = static_cast<int>(Q.shape[1]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    if (batch_size != 1 || head_dim != 128)
        return false;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0)
        return false;
    if (seq_q == 0 || seq_kv == 0 || block_size <= 0)
        return false;
    if (k_data == nullptr || k_scales == nullptr || v_data == nullptr || v_scales == nullptr ||
        block_table == nullptr)
        return false;

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    int Bq;
    {
        size_t smem_128 = compute_smem_mxfp4(128, MX_Bkv, head_dim, false);
        size_t smem_64 = compute_smem_mxfp4(64, MX_Bkv, head_dim, false);
        size_t smem_32 = compute_smem_mxfp4(32, MX_Bkv, head_dim, false);
        if (smem_128 <= (size_t)max_smem) {
            Bq = 128;
        } else if (smem_64 <= (size_t)max_smem) {
            Bq = 64;
        } else if (smem_32 <= (size_t)max_smem) {
            Bq = 32;
        } else {
            return false;
        }
    }
    const size_t smem = compute_smem_mxfp4(Bq, MX_Bkv, head_dim, false);

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    const int n_kv_tiles = (seq_kv + MX_Bkv - 1) / MX_Bkv;

    // Promotion pre-pass: Q̄ from the fresh chunk, K̄ decoded from the cache.
    const uint8_t* d_promote = nullptr;
    if (promote_budget > 0.0f) {
        // s_d_means_paged / s_d_promote_paged are the file-scope scratches above.
        const size_t qmean_elems = (size_t)n_heads * num_q_tiles * head_dim;
        const size_t kmean_elems = (size_t)n_kv_heads * n_kv_tiles * head_dim;
        const size_t means_need = qmean_elems + kmean_elems;
        const size_t mask_need = (size_t)n_heads * num_q_tiles * n_kv_tiles;
        if (means_need > s_means_paged_cap) {
            if (s_d_means_paged != nullptr)
                cudaFree(s_d_means_paged);
            if (cudaMalloc(&s_d_means_paged, means_need * sizeof(float)) != cudaSuccess) {
                s_d_means_paged = nullptr;
                s_means_paged_cap = 0;
            } else {
                s_means_paged_cap = means_need;
            }
        }
        if (mask_need > s_promote_paged_cap) {
            if (s_d_promote_paged != nullptr)
                cudaFree(s_d_promote_paged);
            if (cudaMalloc(&s_d_promote_paged, mask_need) != cudaSuccess) {
                s_d_promote_paged = nullptr;
                s_promote_paged_cap = 0;
            } else {
                s_promote_paged_cap = mask_need;
            }
        }
        if (s_d_means_paged != nullptr && s_d_promote_paged != nullptr) {
            float* d_qmean = s_d_means_paged;
            float* d_kmean_tiles = s_d_means_paged + qmean_elems;
            dim3 qmg(num_q_tiles, n_heads);
            mxfp4_tile_mean_kernel<<<qmg, 128, 0, stream>>>(reinterpret_cast<const half*>(Q.data), d_qmean,
                                                            seq_q, n_heads, head_dim, Bq);
            IMP_CUDA_CHECK_LAUNCH();
            dim3 kmg(n_kv_tiles, n_kv_heads);
            mxfp4_tile_mean_paged_kernel<<<kmg, 128, 0, stream>>>(k_data, k_scales, block_table,
                                                                  block_size, seq_kv, n_kv_heads,
                                                                  head_dim, MX_Bkv, d_kmean_tiles);
            IMP_CUDA_CHECK_LAUNCH();
            dim3 sg(num_q_tiles, n_heads);
            const size_t sel_smem = (size_t)(head_dim + n_kv_tiles) * sizeof(float);
            mxfp4_promote_select_kernel<<<sg, 128, sel_smem, stream>>>(
                d_qmean, d_kmean_tiles, s_d_promote_paged, n_heads, n_kv_heads, num_q_tiles, n_kv_tiles, Bq,
                seq_q, q_offset, causal, promote_budget, head_dim);
            IMP_CUDA_CHECK_LAUNCH();
            d_promote = s_d_promote_paged;
        }
    }

    static bool logged_once_paged = false;
    if (!logged_once_paged) {
        logged_once_paged = true;
        IMP_LOG_INFO("FMHA MXFP4 sm120 PAGED-FP4-K ACTIVE (hd=%d, Bq=%d, promote=%.2f, seq_kv=%d)",
                     head_dim, Bq, (d_promote != nullptr) ? promote_budget : 0.0f, seq_kv);
    }

    MxPagedKVArgs pkv;
    pkv.k_data = k_data;
    pkv.k_scales = k_scales;
    pkv.v_data = v_data;
    pkv.v_scales = v_scales;
    pkv.block_table = block_table;
    pkv.block_size = block_size;

    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(MX_WARP_SIZE, MX_NUM_WARPS);

#define LAUNCH_FMHA_MXFP4_PAGED(BQ, PR)                                                                  \
    do {                                                                                                 \
        cudaError_t attr_err =                                                                           \
            cudaFuncSetAttribute(fmha_sm120_mxfp4_kernel<BQ, 128, true, false, PR, true>,                \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));   \
        if (attr_err != cudaSuccess) {                                                                   \
            IMP_LOG_WARN("FMHA MXFP4 paged: cudaFuncSetAttribute failed Bq=%d smem=%zu: %s", BQ, smem,   \
                         cudaGetErrorString(attr_err));                                                  \
            return false;                                                                                \
        }                                                                                                \
        cudaFuncSetAttribute(fmha_sm120_mxfp4_kernel<BQ, 128, true, false, PR, true>,                    \
                             cudaFuncAttributePreferredSharedMemoryCarveout,                             \
                             cudaSharedmemCarveoutMaxShared);                                            \
        fmha_sm120_mxfp4_kernel<BQ, 128, true, false, PR, true><<<grid, block, smem, stream>>>(          \
            reinterpret_cast<const half*>(Q.data), k_fresh, v_fresh, reinterpret_cast<half*>(O.data),    \
            batch_size, seq_q, seq_kv, n_heads, n_kv_heads, scale, causal, sliding_window, softcap,      \
            q_offset, /*d_kmean=*/nullptr, d_promote, n_kv_tiles, pkv);                                  \
        IMP_CUDA_CHECK_LAUNCH();                                                                         \
    } while (0)

    // Promote is always instantiated: current-chunk tiles ride the promoted
    // (exact) machinery even at budget 0 (d_promote == nullptr).
    if (Bq == 128) {
        LAUNCH_FMHA_MXFP4_PAGED(128, true);
    } else if (Bq == 64) {
        LAUNCH_FMHA_MXFP4_PAGED(64, true);
    } else {
        LAUNCH_FMHA_MXFP4_PAGED(32, true);
    }
#undef LAUNCH_FMHA_MXFP4_PAGED
    return true;
}

}  // namespace imp
