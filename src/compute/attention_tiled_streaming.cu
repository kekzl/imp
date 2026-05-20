#include "compute/attention_tiled_streaming.h"
#include "core/logging.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

namespace {

// 1 producer + 7 consumers = 8 warps × 32 threads = 256 threads/CTA.
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kProducerWarp = 0;

// MMA tile dimensions (m16n8k16 FP16).
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

// Bkv per hd. Br baked into kernel template.
template <int HD>
constexpr int default_Bkv() {
    return (HD <= 128) ? 64 : 32;
}

// Br per hd. Picked in §2 of the spec.
template <int HD>
constexpr int default_Br() {
    if constexpr (HD == 64)  return 128;
    else if constexpr (HD == 96)  return 96;
    else if constexpr (HD == 128) return 64;
    else if constexpr (HD == 256) return 32;
    else if constexpr (HD == 512) return 32;
    else return -1;  // SFINAE-ish: unsupported.
}

// HD chunk size for hd=512 chunked path.
constexpr int kHDChunkBytes = 128 * 2;  // 128 halves = 256 B
constexpr int kHDChunkHalves = 128;

}  // namespace

namespace {

__device__ __forceinline__ void cp_async_16(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(glob));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(s), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(s));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE_%=;\n"
        "bra WAIT_%=;\n"
        "DONE_%=:\n"
        "}\n"
        :: "r"(s), "r"(phase));
}

// ldmatrix x4 (loads 4 fragments, 16x16 halves, into 4 32-bit regs per lane).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

// mma.sync.m16n8k16 FP16 in/out (acc FP32). D += A·B.
__device__ __forceinline__ void mma_m16n8k16_f16(
        float (&d)[4],
        const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float redux_max_f32(float x) {
    float result;
    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;\n"
                 : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float redux_add_f32(float x) {
    float result;
    asm volatile("redux.sync.add.f32 %0, %1, 0xffffffff;\n"
                 : "=f"(result) : "f"(x));
    return result;
}

}  // namespace

template <int Br, int HD>
__global__ void __launch_bounds__(kThreads, 1)
attention_tiled_streaming_kernel(
        const __half* __restrict__ Q,
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        __half* __restrict__ O,
        int seq_q, int seq_kv,
        int n_heads, int n_kv_heads,
        float scale, bool causal,
        int sliding_window, float softcap, int q_offset) {
    constexpr int Bkv = default_Bkv<HD>();

    // Suppress unused-parameter warnings for params used in later tasks.
    (void)V; (void)causal; (void)sliding_window; (void)softcap; (void)q_offset;
    (void)seq_kv; (void)Bkv;

    // Block coordinates: x=row-block, y=head, z=batch.
    const int row_block = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int kv_head = head / (n_heads / n_kv_heads);
    (void)kv_head; (void)K;  // K is unused until iter loop (Task 6).

    const int q_row0 = row_block * Br;
    if (q_row0 >= seq_q) return;

    const int tid = threadIdx.x;

    // ------------------------------------------------------------------
    // Shared memory layout
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) uint8_t smem_raw[];

    __half* Q_smem = reinterpret_cast<__half*>(smem_raw);
    __half* K_smem[2];                          // double-buffered
    K_smem[0] = Q_smem + Br * HD;
    K_smem[1] = K_smem[0] + Bkv * HD;
    __half* V_smem = K_smem[1] + Bkv * HD;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(V_smem + Bkv * HD);
    (void)K_smem; (void)V_smem;  // unused until iter loop.

    // mbar layout: [Q_ready, K_ready[0], K_ready[1], V_ready,
    //               QKt_done, V_consumed]
    if (tid == 0) {
        mbar_init(&mbar[0], 1);         // Q_ready
        mbar_init(&mbar[1], 1);         // K_ready[0]
        mbar_init(&mbar[2], 1);         // K_ready[1]
        mbar_init(&mbar[3], 1);         // V_ready
        mbar_init(&mbar[4], 7);         // QKt_done
        mbar_init(&mbar[5], 7);         // V_consumed
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Q load: one-time. All 256 threads cooperate.
    // ------------------------------------------------------------------
    const __half* Q_gmem = Q
        + static_cast<size_t>(batch) * seq_q * n_heads * HD
        + static_cast<size_t>(q_row0) * n_heads * HD
        + static_cast<size_t>(head) * HD;

    constexpr int kHalvesPerChunk = 8;          // 16 bytes per cp.async
    constexpr int kQChunks = (Br * HD) / kHalvesPerChunk;
    for (int c = tid; c < kQChunks; c += kThreads) {
        int elem = c * kHalvesPerChunk;
        int r = elem / HD;
        int d = elem % HD;
        const __half* src = Q_gmem + static_cast<size_t>(r) * n_heads * HD + d;
        cp_async_16(&Q_smem[r * HD + d], src);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    if (tid == 0) mbar_arrive(&mbar[0]);

    // Real iteration loop lands in Task 6. For now: just return so the
    // launcher path doesn't UB. (Note: returning here is fine; the kernel
    // hasn't written O yet so the test will FAIL on the correctness check —
    // expected baseline state for Task 5.)
    (void)Q_gmem;  // silence unused after we exit early.
    (void)O; (void)scale;
}

bool attention_tiled_streaming_prefill(const Tensor& Q, const Tensor& K,
                                       const Tensor& V, Tensor& O, float scale,
                                       bool causal, int sliding_window,
                                       float softcap, int q_offset,
                                       cudaStream_t stream) {
    // v1: only hd=128 supported at this task. Other hds bail to cuBLAS.
    if (Q.qtype != QType::F16 || K.qtype != QType::F16 || V.qtype != QType::F16)
        return false;
    if (Q.ndim != 4) return false;
    const int batch = static_cast<int>(Q.shape[0]);
    const int seq_q = static_cast<int>(Q.shape[1]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int seq_kv = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    if (seq_q == 0 || seq_kv == 0) return false;
    if (head_dim != 128) return false;       // expanding in Task 7+

    constexpr int Br = 64;
    constexpr int HD = 128;
    constexpr int Bkv = 64;

    // Smem: Q + K_dbuf + V + 6 mbarriers.
    const size_t smem_bytes =
          Br * HD * sizeof(__half)
        + 2 * Bkv * HD * sizeof(__half)
        + Bkv * HD * sizeof(__half)
        + 6 * sizeof(uint64_t);

    cudaFuncSetAttribute(
        attention_tiled_streaming_kernel<Br, HD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));

    dim3 grid((seq_q + Br - 1) / Br, n_heads, batch);
    attention_tiled_streaming_kernel<Br, HD><<<grid, kThreads, smem_bytes, stream>>>(
        static_cast<const __half*>(Q.data),
        static_cast<const __half*>(K.data),
        static_cast<const __half*>(V.data),
        static_cast<__half*>(O.data),
        seq_q, seq_kv, n_heads, n_kv_heads,
        scale, causal, sliding_window, softcap, q_offset);

    if (cudaGetLastError() != cudaSuccess) return false;
    (void)scale; // referenced for compile, kernel will use later
    return true;
}

}  // namespace imp
