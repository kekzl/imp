// =============================================================================
// attention_mxf4nvf4_probe.cu -- Feasibility probe for mxf4nvf4.block_scale MMA
// =============================================================================
//
// Tests whether the SageAttention3-style hardware block-scale MMA compiles
// and runs on imp's sm_120f + CUDA 13.2 toolchain. Does NOT integrate into
// the real attention path — this is only a compile + link gate.
//
// Upgrade target (vs existing kind::f8f6f4.m16n8k32):
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//
// Reference: SageAttention3, thu-ml/SageAttention, cute_extension.h
// =============================================================================

#include "bench/attention_mxf4nvf4_probe.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace imp {

// Minimal kernel: runs one block-scaled MMA instance with canned inputs.
// Not correctness-validated against a reference — only exercises the PTX.
//
// Inputs:
//   a[4]  — NVFP4 A operand (m16k64 tile: 16*64*4 bit = 128 bytes = 4 uint32 per lane)
//   b[2]  — NVFP4 B operand (single m16n8k64 sub-tile: 8*64*4 bit = 64 bytes = 2 uint32 per lane)
//   sfa   — FP8 UE4M3 A scale packed in uint32
//   sfb   — FP8 UE4M3 B scale packed in uint32
// Output:
//   d[4]  — FP32 accumulator, written to global out
//
__global__ void probe_mxf4nvf4_blockscale_kernel(const uint32_t* __restrict__ a_in,  // [4]
                                                 const uint32_t* __restrict__ b_in,  // [2]
                                                 uint32_t sfa_in, uint32_t sfb_in,
                                                 float* __restrict__ d_out)  // [4]
{
    // Each lane holds its fragment of the operands (CUTLASS convention).
    uint32_t a0 = a_in[0];
    uint32_t a1 = a_in[1];
    uint32_t a2 = a_in[2];
    uint32_t a3 = a_in[3];
    uint32_t b0 = b_in[0];
    uint32_t b1 = b_in[1];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    constexpr uint16_t tidA = 0;
    constexpr uint16_t bidA = 0;
    constexpr uint16_t bidB = 0;
    constexpr uint16_t tidB0 = 0;

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
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sfa_in), "h"(bidA), "h"(tidA), "r"(sfb_in), "h"(bidB), "h"(tidB0));
#else
    // Host-side or pre-sm_120 stub so the file compiles everywhere.
    d0 = static_cast<float>(a0 & 0xFF);
    d1 = static_cast<float>(b0 & 0xFF);
    d2 = static_cast<float>(sfa_in & 0xFF);
    d3 = static_cast<float>(sfb_in & 0xFF);
#endif

    if (threadIdx.x == 0) {
        d_out[0] = d0;
        d_out[1] = d1;
        d_out[2] = d2;
        d_out[3] = d3;
    }
}

// Probe variant that writes the per-thread accumulator of thread 0 to out.
// Used for numerical assertion tests where the expected output is known.
__global__ void probe_mxf4nvf4_blockscale_dump_kernel(const uint32_t* __restrict__ a_in,
                                                      const uint32_t* __restrict__ b_in, uint32_t sfa_in,
                                                      uint32_t sfb_in, float* __restrict__ d_out) {
    uint32_t a0 = a_in[0];
    uint32_t a1 = a_in[1];
    uint32_t a2 = a_in[2];
    uint32_t a3 = a_in[3];
    uint32_t b0 = b_in[0];
    uint32_t b1 = b_in[1];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    constexpr uint16_t tidA = 0;
    constexpr uint16_t bidA = 0;
    constexpr uint16_t bidB = 0;
    constexpr uint16_t tidB0 = 0;

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
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sfa_in), "h"(bidA), "h"(tidA), "r"(sfb_in), "h"(bidB), "h"(tidB0));
#endif

    if (threadIdx.x == 0) {
        d_out[0] = d0;
        d_out[1] = d1;
        d_out[2] = d2;
        d_out[3] = d3;
    }
}

bool probe_mxf4nvf4_allzero_a(cudaStream_t stream, float out_d[4]) {
    // A = all-zero E2M1 nibbles → dequant(A) = 0 for every element.
    // D = C + dequant(A) * dequant(B) * sf_a * sf_b, with C=0 and the
    // A term zero, must collapse to exactly 0.
    static constexpr uint32_t kAZero[4] = {0u, 0u, 0u, 0u};
    // B with arbitrary non-zero content — output must stay 0 regardless.
    static constexpr uint32_t kB[2] = {0x55555555u, 0x55555555u};
    // Scale factors = 1.0 in FP8 UE4M3 (0x38 per byte = exp=7, man=0).
    static constexpr uint32_t kSFA = 0x38383838u;
    static constexpr uint32_t kSFB = 0x38383838u;

    uint32_t* d_a = nullptr;
    uint32_t* d_b = nullptr;
    float* d_out = nullptr;

    if (cudaMalloc(&d_a, sizeof(kAZero)) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_b, sizeof(kB)) != cudaSuccess) {
        cudaFree(d_a);
        return false;
    }
    if (cudaMalloc(&d_out, 4 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return false;
    }

    cudaMemcpyAsync(d_a, kAZero, sizeof(kAZero), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, kB, sizeof(kB), cudaMemcpyHostToDevice, stream);

    probe_mxf4nvf4_blockscale_dump_kernel<<<1, 32, 0, stream>>>(d_a, d_b, kSFA, kSFB, d_out);

    cudaError_t launch_err = cudaGetLastError();
    cudaMemcpyAsync(out_d, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaError_t sync_err = cudaGetLastError();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return (launch_err == cudaSuccess) && (sync_err == cudaSuccess);
}

// Host-callable probe: runs one MMA, returns true on kernel launch success.
// Does NOT validate numerical correctness.
bool probe_mxf4nvf4_blockscale(cudaStream_t stream) {
    static constexpr uint32_t kA[4] = {0x44444444u, 0x44444444u, 0x44444444u, 0x44444444u};
    static constexpr uint32_t kB[2] = {0x44444444u, 0x44444444u};
    static constexpr uint32_t kSFA = 0x38383838u;  // ~1.0 in FP8 UE4M3
    static constexpr uint32_t kSFB = 0x38383838u;

    uint32_t* d_a = nullptr;
    uint32_t* d_b = nullptr;
    float* d_out = nullptr;

    if (cudaMalloc(&d_a, sizeof(kA)) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_b, sizeof(kB)) != cudaSuccess) {
        cudaFree(d_a);
        return false;
    }
    if (cudaMalloc(&d_out, 4 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return false;
    }

    cudaMemcpyAsync(d_a, kA, sizeof(kA), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, kB, sizeof(kB), cudaMemcpyHostToDevice, stream);

    probe_mxf4nvf4_blockscale_kernel<<<1, 32, 0, stream>>>(d_a, d_b, kSFA, kSFB, d_out);

    cudaError_t launch_err = cudaGetLastError();
    cudaStreamSynchronize(stream);
    cudaError_t sync_err = cudaGetLastError();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return (launch_err == cudaSuccess) && (sync_err == cudaSuccess);
}

}  // namespace imp
