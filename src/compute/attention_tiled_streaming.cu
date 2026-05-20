#include "compute/attention_tiled_streaming.h"
#include "core/logging.h"
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

bool attention_tiled_streaming_prefill(const Tensor& Q, const Tensor& K,
                                       const Tensor& V, Tensor& O, float scale,
                                       bool causal, int sliding_window,
                                       float softcap, int q_offset,
                                       cudaStream_t stream) {
    // Stub: returns false so the dispatcher falls back to cuBLAS.
    // Real implementation lands in subsequent tasks.
    (void)Q; (void)K; (void)V; (void)O; (void)scale; (void)causal;
    (void)sliding_window; (void)softcap; (void)q_offset; (void)stream;
    return false;
}

}  // namespace imp
