#ifndef IMP_COMPUTE_MMQ_Q8_IMMA_INTERNAL_CUH
#define IMP_COMPUTE_MMQ_Q8_IMMA_INTERNAL_CUH

// Shared internals for the INT8 IMMA prefill GEMM family (sm_120a), split out of
// mmq_q8_imma.cu (recompile-blast-radius gate).
// Tile constants, cp.async primitives, cross-TU kernel declarations; definitions live in
// mmq_q8_imma_q4k/_q6k/_q51.cu, dispatch in mmq_q8_imma.cu. Keep byte-identical.

#include <mutex>
#include <unordered_map>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

constexpr int kBN = 128;
constexpr int kBK = 64;   // 2 sub-blocks per K-step
constexpr int kPad = 16;  // smem row pad (bytes): 80-B stride = conflict-free rl-lane access
constexpr int kRow = kBK + kPad;
constexpr int kStages = 2;
constexpr int kThreads = 256;

__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 16 : 0;  // src-size 0 → zero-fill (OOB M-tail rows)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(glob),
                 "r"(src_size));
}
__device__ __forceinline__ void cp_async_ca_8(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(glob),
                 "r"(src_size));
}
__device__ __forceinline__ void cp_async_ca_4(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(glob),
                 "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

constexpr int kQlRow = 64 + 16;
constexpr int kQhRow = 32 + 16;

// Cross-TU template kernel declarations (definitions + explicit instantiations
// live in the per-format .cu files). External linkage so the dispatch in
// mmq_q8_imma.cu can launch / cudaFuncSetAttribute them.
// SPLITK (dense only): gridDim.z = K-split, fp32 partials to split_out[z][M][N], ks_per_split 128-K steps
// each.
template <int BM, bool BETA1, bool SPLITK = false>
__global__ void mmq_imma_q4k_raw_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                                        const float* __restrict__ x_rowsum, const uint8_t* __restrict__ Wq4k,
                                        __half* __restrict__ out, int M, int N, int K,
                                        const int32_t* __restrict__ expert_offsets, size_t w_stride_blocks,
                                        float* __restrict__ split_out = nullptr, int ks_per_split = 0);

// Reads the GGUF 210-B blocks in place (aligned words + funnel shift).
template <int BM, bool BETA1>
__global__ void mmq_imma_q6k_raw_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                                        const uint8_t* __restrict__ Wq6k, __half* __restrict__ out, int M,
                                        int N, int K, const int32_t* __restrict__ expert_offsets,
                                        size_t w_stride_blocks);

// Dense Q6_K at M <= 32 (mmq_q8_imma_q6k_k256.cu): BM=32, one super-block per step, cp.async staging.
// SPLITK: gridDim.z = K-split, fp32 partials to split_out[z][M][N], sb_per_split super-blocks each.
template <bool BETA1, bool SPLITK>
__global__ void mmq_imma_q6k_k256_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                                         const uint8_t* __restrict__ Wq6k, __half* __restrict__ out, int M,
                                         int N, int K, float* __restrict__ split_out, int sb_per_split);
// Per stage: A [32][272] s8, A scales [32][8] half, B 210-B block windows [128][60 words].
constexpr int kQ6kK256ARow = 256 + 16;
constexpr int kQ6kK256BRowW = 60;  // 60 % 32 = 28: rl-lane word reads conflict-free
constexpr size_t q6k_k256_stage_bytes() {
    return (32 * kQ6kK256ARow) + (32 * 8 * sizeof(__half)) + (static_cast<size_t>(kBN) * kQ6kK256BRowW * 4);
}
constexpr size_t q6k_k256_smem_bytes() { return kStages * q6k_k256_stage_bytes(); }

template <int BM, bool BETA1>
__global__ void mmq_imma_q51_raw_kernel(const int8_t* __restrict__ X_s8,
                                        const __half* __restrict__ x_scale,
                                        const float* __restrict__ x_rowsum,
                                        const uint8_t* __restrict__ Wq51, __half* __restrict__ out,
                                        int M, int N, int K,
                                        const int32_t* __restrict__ expert_offsets,
                                        size_t w_stride_blocks);

template <int BM, bool BETA1>
__global__ void mmq_imma_q5k_raw_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                                        const float* __restrict__ x_rowsum, const uint8_t* __restrict__ Wq5k,
                                        __half* __restrict__ out, int M, int N, int K,
                                        const int32_t* __restrict__ expert_offsets, size_t w_stride_blocks);

// Q5_K raw kernel dynamic smem: A tile, qs + qh rows (32 B + 16 pad), 16-B headers, per-row A scale
// and rowsum, (alpha, beta) pairs; kStages each.
constexpr size_t q5k_smem_bytes(int BM) {
    return static_cast<size_t>(kStages) *
           (static_cast<size_t>(BM) * kRow + 2 * static_cast<size_t>(kBN) * (32 + 16) +
            static_cast<size_t>(kBN) * 16 + static_cast<size_t>(BM) * 2 * sizeof(__half) +
            static_cast<size_t>(BM) * 2 * sizeof(float) + static_cast<size_t>(kBN) * 2 * sizeof(__half2));
}

// Q4_K raw kernel: 128-K step (4 sub-blocks, 3 barriers per 16 MMAs per warp, #2441).
constexpr int kQ4kBK = 128;
constexpr int kQ4kSub = kQ4kBK / 32;
constexpr int kQ4kRow = kQ4kBK + kPad;  // 144-B stride: rl-lane A reads conflict-free
constexpr int kQ4kQRow = 64 + 16;       // two 32-B nibble groups + pad
constexpr size_t q4k_smem_bytes(int BM) {
    return static_cast<size_t>(kStages) *
           (static_cast<size_t>(BM) * kQ4kRow + static_cast<size_t>(kBN) * kQ4kQRow +
            static_cast<size_t>(kBN) * 16 + static_cast<size_t>(BM) * kQ4kSub * sizeof(float) +
            static_cast<size_t>(kBN) * kQ4kSub * sizeof(__half2) +
            static_cast<size_t>(BM) * kQ4kSub * sizeof(__half));
}

// Q8_0 dense plane GEMM, 128-K step (mmq_q8_imma_k128.cu). bm 32 or 128, beta 0 or 1;
// false = shape or smem opt-in unsupported, nothing launched.
[[nodiscard]] bool launch_q8_k128(const int8_t* xs8, const __half* xscale, const int8_t* w_qs,
                                  const __half* w_sc, __half* out, int M, int N, int K, int bm, bool beta1,
                                  dim3 grid, cudaStream_t stream);

constexpr size_t q6k_smem_bytes(int BM) {
    return static_cast<size_t>(kStages) *
           (static_cast<size_t>(BM) * kRow + static_cast<size_t>(kBN) * kQlRow +
            static_cast<size_t>(kBN) * kQhRow + static_cast<size_t>(kBN) * 8 +
            static_cast<size_t>(BM) * 4);
}

// State owned by mmq_q8_imma_scratch.cu, declared here (not file-static) because
// mmq_q8_imma.cu's dispatch reads scratch pointers directly for kernel args.
// Weight caches are model-resident direct allocations (T1, A7 step 6 will move them);
// activation triple + split-K slice come from the T2 arena, carrying their generation
// (A7 step 8 / AUDIT B13).
struct WeightPlanes {
    int8_t* qs = nullptr;
    __half* sc = nullptr;  // interleaved (alpha, beta) [N][K/32][2]
    int N = 0;             // total rows (ne x per-expert rows for MoE)
    int K = 0;
};

struct ActScratch {
    int8_t* xs8 = nullptr;
    __half* xscale = nullptr;
    float* xrowsum = nullptr;
    size_t cap_mk = 0;
    size_t cap_msubs = 0;
    uint64_t gen = 0;
};

struct SplitKScratch {
    float* buf = nullptr;
    size_t cap = 0;
    uint64_t gen = 0;
};

extern std::mutex g_imma_mtx;
// Planned budget for the per-weight prefill caches and what they hold now
// (mmq_q8_imma_scratch.cu, #1899). Both are read under g_imma_mtx.
extern size_t g_imma_plane_budget;
extern size_t g_imma_plane_used;
extern bool g_imma_plane_budget_hit;
extern std::unordered_map<const void*, WeightPlanes> g_imma_weights;
extern ActScratch g_imma_act;
extern SplitKScratch g_imma_splitk;

[[nodiscard]] bool imma_stream_capturing(cudaStream_t stream);
[[nodiscard]] bool imma_ensure_weight(const void* src, int N, int K, cudaStream_t stream, bool capturing);
[[nodiscard]] bool imma_ensure_act(int M, int K, bool capturing);
[[nodiscard]] bool imma_ensure_splitk(size_t floats, bool capturing);
void imma_quantize_act(const __half* x, int M, int K, cudaStream_t stream);
// True if the last imma_quantize_act wrote (x, M, K) on `stream` into the current scratch.
[[nodiscard]] bool imma_act_holds(const __half* x, int M, int K, cudaStream_t stream);
void imma_act_forget();
// Thread-local one-shot: this thread's next imma_quantize_act skips if imma_act_holds. Only a
// caller that knows x is unchanged since that quantize may set it (#2469).
void imma_act_reuse_next(bool on);

// Dynamic smem opt-in of one kernel; false (logged) when the driver refuses.
[[nodiscard]] bool imma_smem_opt_in(const void* kernel, size_t bytes);
// Fixed-order sum of `used` fp32 slices of g_imma_splitk into out[total] (beta1: out += sum).
void imma_splitk_finalize(int used, int total, __half* out, bool beta1, cudaStream_t stream);

// K-splits of a dense M <= 32 raw-kernel grid of `steps` K-steps: about two waves of CTAs, from N and K
// only (rows of different M keep their k order). n_tiles * S <= 384 keeps N * S under kExecImmaSplitkBytes.
inline int imma_dense_splits(int N, int steps) {
    const int n_tiles = (N + kBN - 1) / kBN;
    int s = 1;
    while (s < 16 && n_tiles * s * 2 <= 384 && steps / (s * 2) >= 2)
        s *= 2;
    return s;
}

// Q4_K raw launch (mmq_q8_imma_q4k.cu): dense M <= 32 on BM=32 tiles (per row bit-identical to BM=128),
// K-split when allow_split. Q6_K dense M <= 32 (mmq_q8_imma_q6k_k256.cu). False = smem opt-in failed.
[[nodiscard]] bool launch_q4k(const uint8_t* w4, __half* out_f16, int M, int N, int K, dim3 grid,
                              bool small_m, const int32_t* d_offsets, float beta, bool allow_split,
                              bool capturing, cudaStream_t stream);
[[nodiscard]] bool launch_q6k_k256(const uint8_t* w6, __half* out_f16, int M, int N, int K, float beta,
                                   bool capturing, cudaStream_t stream);
// Dense M <= 32 on GGUF blocks in place (mmq_q8_imma_gguf_smallm.cu), qkind 5 Q8_0, 6 IQ4_NL, 7 IQ4_XS,
// 8 Q5_K: K % 256 == 0, w aligned to the format's copy chunk. False = unknown qkind or smem opt-in failed.
// Dense Q8_0 (qkind 5) / IQ4_XS (7) above 32 rows (mmq_q8_imma_gguf_bm128.cu): BM=128 tiles, K % 256 == 0.
// False = other qkind, nothing launched.
[[nodiscard]] bool launch_gguf_bm128(int qkind, const uint8_t* w, __half* out_f16, int M, int N, int K,
                                     float beta, cudaStream_t stream);
[[nodiscard]] bool launch_gguf_small_m(int qkind, const uint8_t* w, __half* out_f16, int M, int N, int K,
                                       float beta, bool capturing, cudaStream_t stream);

// BM=32 raw-kernel epilogue, rows row_lo and row_lo + 8 of the warp tile: fp32 partials (SPLITK) or
// fp16 (+= when BETA1).
template <bool BETA1, bool SPLITK, int kNF>
__device__ __forceinline__ void imma_small_m_store(const float (&acc)[kNF][4], __half* __restrict__ C,
                                                   float* __restrict__ Cs, int row_lo, int col0, int M,
                                                   int N) {
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf) {
        const int col = col0 + (nf * 8);
        if (col + 1 >= N)
            continue;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row_lo + (h * 8);
            if (row >= M)
                continue;
            const size_t o = (static_cast<size_t>(row) * N) + col;
            if constexpr (SPLITK) {
                *reinterpret_cast<float2*>(&Cs[o]) = make_float2(acc[nf][2 * h], acc[nf][(2 * h) + 1]);
            } else {
                __half2* p = reinterpret_cast<__half2*>(&C[o]);
                const __half2 v = __floats2half2_rn(acc[nf][2 * h], acc[nf][(2 * h) + 1]);
                *p = BETA1 ? __hadd2(*p, v) : v;
            }
        }
    }
}

}  // namespace imp

#endif  // IMP_COMPUTE_MMQ_Q8_IMMA_INTERNAL_CUH
