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
#include "core/logging.h"
#include "quant/fp8_utils.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

namespace imp {

// =============================================================================
// Constants
// =============================================================================

static constexpr int MX_WARP_SIZE     = 32;
static constexpr int MX_NUM_WARPS     = 8;
static constexpr int MX_BLOCK_THREADS = MX_WARP_SIZE * MX_NUM_WARPS; // 256
static constexpr int MX_Bkv           = 64;  // KV tile columns

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
        return (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f)
             + (a >= 2.5f)  + (a >= 3.5f)  + (a >= 5.0f);
    };
    uint8_t sign0 = (v0 < 0.0f) ? 1u : 0u;
    uint8_t code0 = (sign0 << 3) | quant_abs(fabsf(v0));
    uint8_t sign1 = (v1 < 0.0f) ? 1u : 0u;
    uint8_t code1 = (sign1 << 3) | quant_abs(fabsf(v1));
    return (code1 << 4) | code0;
#endif
}

// =============================================================================
// Kernel template
// =============================================================================

// UseBlockScaleMma=false: legacy kind::f8f6f4.m16n8k32 (2× K-chunks, padded regs).
// UseBlockScaleMma=true:  new kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//                         (merges 2 legacy K-chunks per issue, half the MMA count).
// Scale=1.0 uniform (sfa=sfb=0x38383838) keeps the post-MMA manual per-row scaling
// path identical to the legacy kernel — only the MMA instruction changes.
template <int Bq, int HD, bool UseBlockScaleMma = false>
__global__ void __launch_bounds__(MX_BLOCK_THREADS, 1)
fmha_sm120_mxfp4_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int   batch_size,
    int   seq_q,
    int   seq_kv,
    int   n_heads,
    int   n_kv_heads,
    float scale,
    bool  causal,
    int   sliding_window,
    float softcap)
{
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
    const int tile_q     = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx  = batch_head / n_heads;
    const int head_idx   = batch_head % n_heads;
    const int kv_head    = head_idx / (n_heads / n_kv_heads);

    const int tid     = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / MX_WARP_SIZE;
    const int lane_id = tid % MX_WARP_SIZE;
    const int q_start = tile_q * Bq;

    const int sm_row  = tid / TPR;
    const int sm_lane = tid % TPR;

    const int64_t q_row_stride  = (int64_t)n_heads    * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;

    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q  * q_row_stride
                          + (int64_t)q_start   * q_row_stride
                          + (int64_t)head_idx  * head_dim;
    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride
                          + (int64_t)kv_head   * head_dim;
    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride
                          + (int64_t)kv_head   * head_dim;
    half* O_ptr       = O + (int64_t)batch_idx * seq_q  * q_row_stride
                          + (int64_t)q_start   * q_row_stride
                          + (int64_t)head_idx  * head_dim;

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

    uint8_t* Q_fp4    = reinterpret_cast<uint8_t*>(smem);
    float*   q_scales = reinterpret_cast<float*>(Q_fp4 + Bq * hd_half_padded);
    // KV_buf is aligned to 16 bytes for vectorized loads
    char*    KV_raw   = reinterpret_cast<char*>(q_scales + Bq);
    // Align KV_raw to 16 bytes
    KV_raw = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(KV_raw) + 15) & ~15ULL);
    uint8_t* KV_fp4   = reinterpret_cast<uint8_t*>(KV_raw);  // K as FP4
    half*    KV_fp16  = reinterpret_cast<half*>(KV_raw);      // V as FP16 (same slot)
    float*   k_scales = reinterpret_cast<float*>(KV_raw + Bkv * head_dim * sizeof(half));
    float*   S_tile   = k_scales + Bkv;
    float*   O_acc    = S_tile + Bq * Bkv;
    float*   row_m    = O_acc + Bq * head_dim;
    float*   row_l    = row_m + Bq;
    // Per-row UE4M3 scales for blockscale-MMA sfa/sfb operands (only populated
    // when UseBlockScaleMma=true; uninitialized otherwise). Placed at the tail.
    uint8_t* q_scales_fp8 = reinterpret_cast<uint8_t*>(row_l + Bq);
    uint8_t* k_scales_fp8 = q_scales_fp8 + Bq;

    // Pre-compute sqrt(attention_scale) to absorb into Q and K scales (Opt 3).
    // S_true[i,j] = q_scale[i] * k_scale[j] * mma[i,j], and we want the result
    // pre-multiplied by attention_scale.  Split sqrt evenly: q_scales *= sqrt_scale,
    // k_scales *= sqrt_scale, so the product gives q*k*scale automatically.
    const float sqrt_scale = sqrtf(scale);

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

        // Absmax from shared memory (fast L1 reads)
        {
            const int r = sm_row;
            float local_max = 0.0f;
            if (r < Bq) {
                for (int d = sm_lane; d < head_dim; d += TPR)
                    local_max = fmaxf(local_max, fabsf(__half2float(Q_stage[r * head_dim + d])));
            }
            #pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
            if (sm_lane == 0 && r < Bq) {
                if constexpr (UseBlockScaleMma) {
                    // Blockscale path: encode per-row scale × sqrt(attention_scale) as
                    // UE4M3, store both the dequanted float (for self-consistent quant)
                    // and the byte (for sfa operand).
                    float raw = local_max / 6.0f;
                    uint8_t ue4m3 = float_to_fp8_e4m3(raw * sqrt_scale);
                    q_scales_fp8[r] = ue4m3;
                    // Use dequanted value divided by sqrt_scale (so that inv_scale =
                    // sqrt_scale / q_scales[r] = 1/dequanted_raw_scale).
                    float dequant = fp8_e4m3_to_float_fast(ue4m3);
                    q_scales[r] = dequant;  // store sqrt_scale*raw_dq (MMA sfa quantity)
                } else {
                    q_scales[r] = local_max / 6.0f * sqrt_scale;  // legacy
                }
            }
        }
        __syncthreads();

        // Quantize from shared → Q_fp4 (vectorized: 8 halves = 4 bytes/iter, uint32 store)
        {
            // hd_half ≥ 4 for all supported HDs (32, 48, 64, 128). Stride hd_half_padded
            // is always 4-byte aligned (hd_half + 4 bytes pad).
            const int total_packed_u32 = (Bq * hd_half) / 4;
            for (int idx = tid; idx < total_packed_u32; idx += MX_BLOCK_THREADS) {
                int r = idx / (hd_half / 4);
                int b4 = idx % (hd_half / 4);     // which 4-byte chunk in this row
                int d = b4 * 8;                   // starting half index
                float inv_scale = (q_scales[r] > 0.0f) ? (sqrt_scale / q_scales[r]) : 0.0f;
                const half* src = &Q_stage[r * head_dim + d];
                // Load 8 halves via 2× half2 (4 bytes each)
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
                *reinterpret_cast<uint32_t*>(
                    &Q_fp4[r * hd_half_padded + b4 * 4]) = packed;
            }
        }
    } else {
        // Fallback: 2-pass global reads (for Bq=32, HD>64)
        {
            const int r = sm_row;
            float local_max = 0.0f;
            if (r < Bq && q_start + r < seq_q) {
                for (int d = sm_lane; d < head_dim; d += TPR)
                    local_max = fmaxf(local_max, fabsf(__half2float(Q_ptr[(int64_t)r * q_row_stride + d])));
            }
            #pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
            if (sm_lane == 0 && r < Bq) {
                if constexpr (UseBlockScaleMma) {
                    float raw = local_max / 6.0f;
                    uint8_t ue4m3 = float_to_fp8_e4m3(raw * sqrt_scale);
                    q_scales_fp8[r] = ue4m3;
                    q_scales[r] = fp8_e4m3_to_float_fast(ue4m3);
                } else {
                    q_scales[r] = local_max / 6.0f * sqrt_scale;
                }
            }
        }
        __syncthreads();

        {
            const int total_packed = Bq * hd_half;
            for (int idx = tid; idx < total_packed; idx += MX_BLOCK_THREADS) {
                int r = idx / hd_half;
                int d_byte = idx % hd_half;
                int d = d_byte * 2;
                if (q_start + r < seq_q) {
                    float inv_scale = (q_scales[r] > 0.0f) ? (sqrt_scale / q_scales[r]) : 0.0f;
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
    if (tid < Bq) { row_m[tid] = -FLT_MAX; row_l[tid] = 0.0f; }
    bool first_kv_iter = true;  // skip O_acc rescale on first tile (l_old=0 → rescale=0)
    __syncthreads();

    // ---- KV tile loop bounds ----
    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv,
                           causal, sliding_window, first_kv_tile, num_kv_tiles);

    // FP4 MMA tiling: m16n8k32 E2M1
    constexpr int FP4_K = MX_MMA_K;                // 32
    constexpr int FP4_K_BYTES = FP4_K / 2;         // 16 bytes per k-chunk (packed nibbles)
    const int hd_chunks_fp4 = head_dim / FP4_K;    // k-loop iterations (same count as FP8)
    const int s_row_tiles = Bq / MX_MMA_M;
    const int s_col_tiles_half = Bkv / MX_MMA_N;   // each m16n8 tile
    const int s_total_tiles = s_row_tiles * s_col_tiles_half;

    // FP16 WMMA for P·V
    const int o_row_tiles   = Bq / MX_WMMA_M;
    const int o_col_tiles   = head_dim / MX_WMMA_N;
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks     = Bkv / MX_WMMA_K;

    // ================================================================
    // Main KV tile loop
    // ================================================================
    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * Bkv;

        // ---- Quantize K tile to FP4 (Opt 1: shared-memory staging) ----
        if constexpr (can_stage_in_stile) {
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
                        *dst = *src;
                    } else {
                        *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
            }
            __syncthreads();

            // Absmax from shared (fast L1 reads)
            {
                const int r = sm_row;
                float local_max = 0.0f;
                if (r < Bkv) {
                    for (int d = sm_lane; d < head_dim; d += TPR)
                        local_max = fmaxf(local_max, fabsf(__half2float(K_stage[r * head_dim + d])));
                }
                #pragma unroll
                for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
                if (sm_lane == 0 && r < Bkv) {
                    if constexpr (UseBlockScaleMma) {
                        float raw = local_max / 6.0f;
                        uint8_t ue4m3 = float_to_fp8_e4m3(raw * sqrt_scale);
                        k_scales_fp8[r] = ue4m3;
                        k_scales[r] = fp8_e4m3_to_float_fast(ue4m3);
                    } else {
                        k_scales[r] = local_max / 6.0f * sqrt_scale;
                    }
                }
            }
            __syncthreads();

            // Quantize from shared → KV_fp4 (vectorized: 8 halves = 4 bytes/iter)
            {
                const int total_packed_u32 = (Bkv * hd_half) / 4;
                for (int idx = tid; idx < total_packed_u32; idx += MX_BLOCK_THREADS) {
                    int r = idx / (hd_half / 4);
                    int b4 = idx % (hd_half / 4);
                    int d = b4 * 8;
                    float inv_scale = (k_scales[r] > 0.0f) ? (sqrt_scale / k_scales[r]) : 0.0f;
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
                    *reinterpret_cast<uint32_t*>(
                        &KV_fp4[r * hd_half_padded + b4 * 4]) = packed;
                }
            }
            __syncthreads();
        } else {
            // Fallback: 2-pass global reads
            {
                const int r = sm_row;
                float local_max = 0.0f;
                if (r < Bkv && kv_start + r < seq_kv) {
                    for (int d = sm_lane; d < head_dim; d += TPR)
                        local_max = fmaxf(local_max, fabsf(__half2float(K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d])));
                }
                #pragma unroll
                for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
                if (sm_lane == 0 && r < Bkv) {
                    if constexpr (UseBlockScaleMma) {
                        float raw = local_max / 6.0f;
                        uint8_t ue4m3 = float_to_fp8_e4m3(raw * sqrt_scale);
                        k_scales_fp8[r] = ue4m3;
                        k_scales[r] = fp8_e4m3_to_float_fast(ue4m3);
                    } else {
                        k_scales[r] = local_max / 6.0f * sqrt_scale;
                    }
                }
            }
            __syncthreads();

            {
                const int total_packed = Bkv * hd_half;
                for (int idx = tid; idx < total_packed; idx += MX_BLOCK_THREADS) {
                    int r = idx / hd_half;
                    int d_byte = idx % hd_half;
                    int d = d_byte * 2;
                    if (kv_start + r < seq_kv) {
                        float inv_scale = (k_scales[r] > 0.0f) ? (sqrt_scale / k_scales[r]) : 0.0f;
                        float v0 = __half2float(K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]) * inv_scale;
                        float v1 = __half2float(K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d + 1]) * inv_scale;
                        KV_fp4[r * hd_half_padded + d_byte] = pack_fp4_pair(v0, v1);
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
        for (int tile_idx = warp_id; tile_idx < s_total_tiles; tile_idx += MX_NUM_WARPS) {
            int ri = tile_idx / s_col_tiles_half;
            int ci = tile_idx % s_col_tiles_half;

            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

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
                for (int k = 0; k < k_pairs; k++) {
                    const int k0 = 2 * k;
                    const int k1 = k0 + 1;
                    const uint8_t* q_base0 = Q_fp4 + ri * MX_MMA_M * hd_half_padded + k0 * FP4_K_BYTES;
                    const uint8_t* q_base1 = Q_fp4 + ri * MX_MMA_M * hd_half_padded + k1 * FP4_K_BYTES;
                    uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                        q_base0 + group_id * hd_half_padded + byte_offset);
                    uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                        q_base0 + (group_id + 8) * hd_half_padded + byte_offset);
                    uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                        q_base1 + group_id * hd_half_padded + byte_offset);
                    uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                        q_base1 + (group_id + 8) * hd_half_padded + byte_offset);

                    const uint8_t* k_base0 = KV_fp4 + ci * MX_MMA_N * hd_half_padded + k0 * FP4_K_BYTES;
                    const uint8_t* k_base1 = KV_fp4 + ci * MX_MMA_N * hd_half_padded + k1 * FP4_K_BYTES;
                    uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                        k_base0 + group_id * hd_half_padded + byte_offset);
                    uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                        k_base1 + group_id * hd_half_padded + byte_offset);

                    // Real block-scale: load per-row UE4M3 scales from SMEM and replicate
                    // to uint32 (all 4 scales per thread cover the same row's k-groups).
                    // SFA layout: thread T's sfa → row m_sfa = (T/4) + (T%2)*8.
                    // SFB layout: thread T's sfb → col n_sfb = T/4.
                    const int m_sfa = (lane_id / 4) + (lane_id % 2) * 8;
                    const int n_sfb = lane_id / 4;
                    uint8_t qs_byte = q_scales_fp8[ri * MX_MMA_M + m_sfa];
                    uint8_t ks_byte = k_scales_fp8[ci * MX_MMA_N + n_sfb];
                    uint32_t sfa = (uint32_t)qs_byte
                                 | ((uint32_t)qs_byte << 8)
                                 | ((uint32_t)qs_byte << 16)
                                 | ((uint32_t)qs_byte << 24);
                    uint32_t sfb = (uint32_t)ks_byte
                                 | ((uint32_t)ks_byte << 8)
                                 | ((uint32_t)ks_byte << 16)
                                 | ((uint32_t)ks_byte << 24);
                    constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
#if __CUDA_ARCH__ >= 1200
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0, %1, %2, %3},"
                        "{%4, %5, %6, %7},"
                        "{%8, %9},"
                        "{%10, %11, %12, %13},"
                        "{%14},"
                        "{%15, %16},"
                        "{%17},"
                        "{%18, %19};\n"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                          "r"(sfa), "h"(bidA), "h"(tidA),
                          "r"(sfb), "h"(bidB), "h"(tidB));
#endif
                }
            } else {
            for (int k = 0; k < hd_chunks_fp4; k++) {
                // Load A fragment: a0=row[groupID], a1=row[groupID+8], a2=a3=0
                uint32_t a0, a1;
                {
                    const uint8_t* q_base = Q_fp4 + ri * MX_MMA_M * hd_half_padded + k * FP4_K_BYTES;
                    a0 = *reinterpret_cast<const uint32_t*>(
                        q_base + group_id * hd_half_padded + byte_offset);
                    a1 = *reinterpret_cast<const uint32_t*>(
                        q_base + (group_id + 8) * hd_half_padded + byte_offset);
                }
                uint32_t a2 = 0, a3 = 0;  // padding for uniform register encoding

                // Load B fragment: b0=col[groupID], b1=0
                uint32_t b0;
                {
                    const uint8_t* k_base = KV_fp4 + ci * MX_MMA_N * hd_half_padded + k * FP4_K_BYTES;
                    b0 = *reinterpret_cast<const uint32_t*>(
                        k_base + group_id * hd_half_padded + byte_offset);
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
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(d0), "f"(d1), "f"(d2), "f"(d3));
#endif
            }
            }

            // Store 16×8 result to S_tile with fused scale + mask (Opt 2).
            // Fuses: per-row scale correction, attention_scale (pre-absorbed via
            // sqrt_scale in q/k_scales), softcap, causal mask, sliding window.
            // Eliminates the separate apply_score_masks pass + __syncthreads.
            {
                int base_row = ri * MX_MMA_M;
                int base_col = ci * MX_MMA_N;
                int r0 = (lane_id / 4) % 8;
                int c0 = (lane_id % 4) * 2;

                // Scale correction: in legacy mode, q_scales/k_scales include
                // sqrt(attention_scale), and their product applies the full scale
                // factor. In real blockscale mode, HW already applied per-row
                // sfa/sfb during MMA, so val is already the attention score.
                float qs0, qs1, ks0, ks1;
                if constexpr (UseBlockScaleMma) {
                    qs0 = qs1 = ks0 = ks1 = 1.0f;  // HW already scaled
                } else {
                    qs0 = q_scales[base_row + r0];
                    qs1 = q_scales[base_row + r0 + 8];
                    ks0 = k_scales[base_col + c0];
                    ks1 = k_scales[base_col + c0 + 1];
                }

                // Compute global Q/K positions for masking
                int gq0 = q_start + base_row + r0;
                int gq1 = q_start + base_row + r0 + 8;
                int gk0 = kv_start + base_col + c0;
                int gk1 = kv_start + base_col + c0 + 1;

                // Inline score computation + masking for all 4 output elements
                #define FUSED_STORE(val, gq, gk, qs, ks, idx) do { \
                    float s = (val) * (qs) * (ks); \
                    if (softcap > 0.0f) s = softcap * tanhf(s / softcap); \
                    if ((gq) >= seq_q || (gk) >= seq_kv) s = -FLT_MAX; \
                    else if (causal && (gq) < (gk)) s = -FLT_MAX; \
                    else if (sliding_window > 0 && ((gq) - (gk)) >= sliding_window) s = -FLT_MAX; \
                    S_tile[idx] = s; \
                } while (0)

                FUSED_STORE(d0, gq0, gk0, qs0, ks0, (base_row + r0) * Bkv + base_col + c0);
                FUSED_STORE(d1, gq0, gk1, qs0, ks1, (base_row + r0) * Bkv + base_col + c0 + 1);
                FUSED_STORE(d2, gq1, gk0, qs1, ks0, (base_row + r0 + 8) * Bkv + base_col + c0);
                FUSED_STORE(d3, gq1, gk1, qs1, ks1, (base_row + r0 + 8) * Bkv + base_col + c0 + 1);
                #undef FUSED_STORE
            }
        }
        __syncthreads();

        // ============================================================
        // Prefetch V tile via cp.async (overlaps with softmax below).
        // KV_fp16 and S_tile are separate SMEM regions, so V writes
        // don't conflict with softmax reads/writes on S_tile.
        // ============================================================
        {
            constexpr int CHUNK_HALVES = 8;  // 16 bytes = 8 halves per cp.async
            const int total_chunks = (Bkv * head_dim) / CHUNK_HALVES;
            for (int c = tid; c < total_chunks; c += MX_BLOCK_THREADS) {
                int elem = c * CHUNK_HALVES;
                int r = elem / head_dim;
                int d = elem % head_dim;
                if (kv_start + r < seq_kv) {
                    cp_async_ca_16(
                        &KV_fp16[r * head_dim + d],
                        &V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                } else {
                    // cp.async can't zero — use regular store
                    reinterpret_cast<float4*>(&KV_fp16[r * head_dim + d])[0] =
                        make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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

            // Step 3: Exp + sum
            float partial_sum = 0.0f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    float p = __expf(S_tile[r * Bkv + c] - m_new);
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
            if (sm_lane == 0 && row_valid) { row_m[r] = m_new; row_l[r] = l_new; }

            // Step 5: Rescale O accumulator (skip on first tile: l_old=0 → rescale=0)
            if (!first_kv_iter) {
                float rescale = (l_old > 0.0f) ? (alpha * l_old / l_new) : 0.0f;
                if (row_valid) {
                    for (int d = sm_lane; d < head_dim; d += TPR)
                        O_acc[r * head_dim + d] *= rescale;
                }
            }

            // Step 6: Normalize + float→half for P
            float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR)
                    SP_half[r * Bkv + c] = __float2half(S_tile[r * Bkv + c] * spv);
            } else if (r < Bq) {
                for (int c = sm_lane; c < Bkv; c += TPR)
                    SP_half[r * Bkv + c] = __float2half(0.0f);
            }
        }

        // Wait for V prefetch to complete, then sync all SMEM writes (P + V)
        cp_async_wait_group<0>();
        __syncthreads();

        // ============================================================
        // Phase 3: O_acc += P · V using FP16 WMMA (m16n16k16)
        // ============================================================
        {
            half* P_half = reinterpret_cast<half*>(S_tile);
            for (int tile_idx = warp_id; tile_idx < o_total_tiles; tile_idx += MX_NUM_WARPS) {
                int ri = tile_idx / o_col_tiles;
                int di = tile_idx % o_col_tiles;

                wmma::fragment<wmma::accumulator, MX_WMMA_M, MX_WMMA_N, MX_WMMA_K, float> o_frag;
                wmma::load_matrix_sync(o_frag,
                    O_acc + ri * MX_WMMA_M * head_dim + di * MX_WMMA_N,
                    head_dim, wmma::mem_row_major);

                for (int k = 0; k < pv_chunks; k++) {
                    wmma::fragment<wmma::matrix_a, MX_WMMA_M, MX_WMMA_N, MX_WMMA_K,
                                   half, wmma::row_major> p_frag;
                    wmma::load_matrix_sync(p_frag,
                        P_half + ri * MX_WMMA_M * Bkv + k * MX_WMMA_K, Bkv);

                    wmma::fragment<wmma::matrix_b, MX_WMMA_M, MX_WMMA_N, MX_WMMA_K,
                                   half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        KV_fp16 + k * MX_WMMA_N * head_dim + di * MX_WMMA_N, head_dim);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(
                    O_acc + ri * MX_WMMA_M * head_dim + di * MX_WMMA_N,
                    o_frag, head_dim, wmma::mem_row_major);
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
            if (q_start + r >= seq_q) continue;
            float4 v = reinterpret_cast<const float4*>(O_acc)[vi];
            half2 lo = __float22half2_rn(make_float2(v.x, v.y));
            half2 hi = __float22half2_rn(make_float2(v.z, v.w));
            uint2 packed;
            packed.x = *reinterpret_cast<const uint32_t*>(&lo);
            packed.y = *reinterpret_cast<const uint32_t*>(&hi);
            *reinterpret_cast<uint2*>(
                &O_ptr[(int64_t)r * q_row_stride + (i % head_dim)]) = packed;
        }
    }
}

// =============================================================================
// Shared memory computation
// =============================================================================

static size_t compute_smem_mxfp4(int Bq, int Bkv, int head_dim) {
    size_t q_fp4    = (size_t)Bq * (head_dim / 2 + 4);           // Q packed FP4 (padded stride)
    size_t q_scales = (size_t)Bq * sizeof(float);                // Q row scales
    size_t align    = 16;                                         // alignment padding
    size_t kv_buf   = (size_t)Bkv * head_dim * sizeof(half);     // KV (FP4 K or FP16 V)
    size_t k_scales = (size_t)Bkv * sizeof(float);               // K row scales
    size_t s_tile   = (size_t)Bq * Bkv * sizeof(float);          // S_tile
    size_t o_acc    = (size_t)Bq * head_dim * sizeof(float);     // O_acc
    size_t softmax  = 2 * (size_t)Bq * sizeof(float);            // row_m + row_l
    size_t scales_fp8 = (size_t)(Bq + Bkv);                      // UE4M3 per-row scales (blockscale)
    return q_fp4 + q_scales + align + kv_buf + k_scales + s_tile + o_acc + softmax + scales_fp8;
}

// =============================================================================
// Host launcher
// =============================================================================

bool fmha_sm120_mxfp4_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, int sliding_window, float softcap,
    cudaStream_t stream,
    bool use_blockscale)
{
    if (Q.dtype != DType::FP16) return false;
    // Blockscale MMA operates on K=64 per issue; head_dim must be a multiple of 64.
    // head_dim=96 (Gemma-class) is a multiple of 32 but NOT 64 → fall back to legacy.
    if (use_blockscale && (static_cast<int>(Q.shape[3]) % 64 != 0)) {
        use_blockscale = false;
    }

    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q      = static_cast<int>(Q.shape[1]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int seq_kv     = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    if (seq_q == 0 || seq_kv == 0) return false;
    if (head_dim % MX_MMA_K != 0) return false;  // FP4 MMA needs head_dim % 32 == 0

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Select Bq: prefer larger tiles, fall back for larger HD
    int Bq;
    {
        size_t smem_128 = compute_smem_mxfp4(128, MX_Bkv, head_dim);
        size_t smem_64  = compute_smem_mxfp4(64,  MX_Bkv, head_dim);
        size_t smem_32  = compute_smem_mxfp4(32,  MX_Bkv, head_dim);
        if (smem_128 <= (size_t)max_smem) {
            Bq = 128;
        } else if (smem_64 <= (size_t)max_smem) {
            Bq = 64;
        } else if (smem_32 <= (size_t)max_smem) {
            Bq = 32;
        } else {
            IMP_LOG_DEBUG("FMHA MXFP4: no Bq fits smem (hd=%d, smem_32=%zu, max=%d)",
                          head_dim, smem_32, max_smem);
            return false;
        }
    }

    const size_t smem = compute_smem_mxfp4(Bq, MX_Bkv, head_dim);

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(MX_WARP_SIZE, MX_NUM_WARPS);

    IMP_LOG_DEBUG("FMHA MXFP4 sm120: B=%d Sq=%d Skv=%d nh=%d nkv=%d hd=%d Bq=%d Bkv=%d smem=%zu "
                  "causal=%d sw=%d softcap=%.1f",
                  batch_size, seq_q, seq_kv, n_heads, n_kv_heads, head_dim,
                  Bq, MX_Bkv, smem, causal, sliding_window, softcap);

    #define LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, BS) do { \
        cudaError_t attr_err = cudaFuncSetAttribute( \
            fmha_sm120_mxfp4_kernel<BQ, HD, BS>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, \
            static_cast<int>(smem)); \
        if (attr_err != cudaSuccess) { \
            IMP_LOG_WARN("FMHA MXFP4: cudaFuncSetAttribute failed Bq=%d HD=%d bs=%d smem=%zu: %s", \
                         BQ, HD, (int)BS, smem, cudaGetErrorString(attr_err)); \
            return false; \
        } \
        cudaFuncSetAttribute(fmha_sm120_mxfp4_kernel<BQ, HD, BS>, \
            cudaFuncAttributePreferredSharedMemoryCarveout, \
            cudaSharedmemCarveoutMaxShared); \
        fmha_sm120_mxfp4_kernel<BQ, HD, BS><<<grid, block, smem, stream>>>( \
            reinterpret_cast<const half*>(Q.data), \
            reinterpret_cast<const half*>(K.data), \
            reinterpret_cast<const half*>(V.data), \
            reinterpret_cast<half*>(O.data), \
            batch_size, seq_q, seq_kv, \
            n_heads, n_kv_heads, \
            scale, causal, sliding_window, softcap); \
    } while (0)

    #define LAUNCH_FMHA_MXFP4(BQ, HD) do { \
        if (use_blockscale) LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, true); \
        else                LAUNCH_FMHA_MXFP4_IMPL(BQ, HD, false); \
    } while (0)

    if (Bq == 128) {
        switch (head_dim) {
            case 64:  LAUNCH_FMHA_MXFP4(128, 64);  return true;
            case 96:  LAUNCH_FMHA_MXFP4(128, 96);  return true;
            case 128: LAUNCH_FMHA_MXFP4(128, 128); return true;
            case 256: LAUNCH_FMHA_MXFP4(128, 256); return true;
            default: break;
        }
    } else if (Bq == 64) {
        switch (head_dim) {
            case 64:  LAUNCH_FMHA_MXFP4(64, 64);   return true;
            case 96:  LAUNCH_FMHA_MXFP4(64, 96);   return true;
            case 128: LAUNCH_FMHA_MXFP4(64, 128);  return true;
            case 256: LAUNCH_FMHA_MXFP4(64, 256);  return true;
            default: break;
        }
    } else {
        switch (head_dim) {
            case 64:  LAUNCH_FMHA_MXFP4(32, 64);   return true;
            case 96:  LAUNCH_FMHA_MXFP4(32, 96);   return true;
            case 128: LAUNCH_FMHA_MXFP4(32, 128);  return true;
            case 256: LAUNCH_FMHA_MXFP4(32, 256);  return true;
            default: break;
        }
    }

    #undef LAUNCH_FMHA_MXFP4
    #undef LAUNCH_FMHA_MXFP4_IMPL
    return false;
}

} // namespace imp
