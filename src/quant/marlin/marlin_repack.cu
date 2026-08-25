// Marlin weight preparation for imp's plain NVFP4 layout: transpose the
// packed nibbles into GPTQ orientation, run the vendored Marlin repack
// kernel, and process the FP8 micro-scales into the shifted "S0E5M3" format
// the FP4 Marlin kernel dequantizes against (port of
// vllm marlin_utils_fp4.prepare_fp4_layer_for_marlin, FP16-activation case:
// scale_factor = 1, global scale = tensor_scale * 2^7).

#include "marlin_repack_kernel.cuh"
#include "marlin_w4a16.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace imp {
namespace marlin_w4a16 {

void marlin_ensure_func_attrs();  // marlin_gemm.cu

namespace {

// [N, cols] u32 -> [cols, N] u32 tiled transpose (init-time only).
__global__ void transpose_u32_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n_rows,
                                     int cols) {
    __shared__ uint32_t tile[32][33];
    int c = blockIdx.x * 32 + threadIdx.x;
    int r = blockIdx.y * 32 + threadIdx.y;
    if (r < n_rows && c < cols)
        tile[threadIdx.y][threadIdx.x] = in[(size_t)r * cols + c];
    __syncthreads();
    int tc = blockIdx.y * 32 + threadIdx.x;  // output col = input row
    int tr = blockIdx.x * 32 + threadIdx.y;  // output row = input col
    if (tr < cols && tc < n_rows)
        out[(size_t)tr * n_rows + tc] = tile[threadIdx.x][threadIdx.y];
}

float fp8_e4m3_to_float(uint8_t v) {
    int sign = (v >> 7) ? -1 : 1;
    int exp = (v >> 3) & 0xF;
    int mant = v & 0x7;
    if (exp == 0)
        return sign * (mant / 8.0f) * 0.015625f;  // 2^-6
    return sign * (1.0f + mant / 8.0f) * std::pow(2.0f, exp - 7);
}

}  // namespace

bool prepare(const void* d_packed, const void* d_micro_scales, float tensor_scale, int N, int K,
             MarlinWeight& out, cudaStream_t stream) {
    out = MarlinWeight{};
    if (!shape_supported(N, K) || N % MARLIN_NAMESPACE_NAME::tile_n_size != 0 || K % 16 != 0)
        return false;
    marlin_ensure_func_attrs();

    const size_t qweight_bytes = (size_t)K * N / 2;
    const size_t groups = (size_t)K / 16;
    const size_t scale_bytes = groups * N;

    // --- weights: transpose [N, K/8]u32 -> [K/8, N]u32, then Marlin repack ---
    void* d_transposed = nullptr;
    if (cudaMalloc(&d_transposed, qweight_bytes) != cudaSuccess)
        return false;
    if (cudaMalloc(&out.qweight, qweight_bytes) != cudaSuccess) {
        cudaFree(d_transposed);
        out = MarlinWeight{};
        return false;
    }
    {
        const int cols = K / 8;
        dim3 grid((cols + 31) / 32, (N + 31) / 32);
        dim3 block(32, 32);
        transpose_u32_kernel<<<grid, block, 0, stream>>>(static_cast<const uint32_t*>(d_packed),
                                                         static_cast<uint32_t*>(d_transposed), N, cols);
    }
    {
        int dev = 0, sms = 0, max_shared_mem = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        auto kernel = MARLIN_NAMESPACE_NAME::gptq_marlin_repack_kernel<MARLIN_NAMESPACE_NAME::repack_threads,
                                                                       4, false, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
        kernel<<<sms, MARLIN_NAMESPACE_NAME::repack_threads, max_shared_mem, stream>>>(
            static_cast<const uint32_t*>(d_transposed), nullptr, static_cast<uint32_t*>(out.qweight), K, N);
    }

    // --- scales: [N, K/16] fp8 -> permuted + processed [K/16, N] bytes ---
    std::vector<uint8_t> h_src(scale_bytes);
    if (cudaMemcpyAsync(h_src.data(), d_micro_scales, scale_bytes, cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaFree(d_transposed);
        release(out);
        return false;
    }
    cudaFree(d_transposed);

    // marlin_permute_scales: transpose to (K/16, N), then within each flat row
    // of 64 apply scale_perm[i*8+j] = i + 8*j.
    std::vector<uint16_t> s_half(groups * N);
    for (size_t n = 0; n < (size_t)N; n++)
        for (size_t g = 0; g < groups; g++)
            s_half[g * N + n] = __half_as_ushort(__float2half(fp8_e4m3_to_float(h_src[n * groups + g])));

    std::vector<uint16_t> s_perm(groups * N);
    const size_t rows64 = groups * N / 64;
    for (size_t r = 0; r < rows64; r++)
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                s_perm[r * 64 + i * 8 + j] = s_half[r * 64 + i + 8 * j];

    // nvfp4_marlin_process_scales (a_dtype = fp16 => scale_factor = 1):
    // quads [0,1,2,3] <- [0,2,1,3]; v = half(s) * 2^7; v < 2 -> 0;
    // byte = high byte of (bits(v) << 1).
    std::vector<uint8_t> s_out(groups * N);
    for (size_t q = 0; q < groups * N / 4; q++) {
        const int gather[4] = {0, 2, 1, 3};
        for (int i = 0; i < 4; i++) {
            __half h = __ushort_as_half(s_perm[q * 4 + gather[i]]);
            __half v = __hmul(h, __float2half(128.0f));
            if (__half2float(v) < 2.0f)
                v = __float2half(0.0f);
            uint16_t bits = static_cast<uint16_t>(__half_as_ushort(v) << 1);
            s_out[q * 4 + i] = static_cast<uint8_t>(bits >> 8);
        }
    }

    if (cudaMalloc(&out.scales, scale_bytes) != cudaSuccess ||
        cudaMemcpy(out.scales, s_out.data(), scale_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        release(out);
        return false;
    }

    // --- global scale: tensor_scale * 2^(exponent_bias_fp16 - 7) = ts * 128 ---
    const float gs = tensor_scale * 128.0f;
    if (cudaMalloc(&out.d_global_scale, sizeof(float)) != cudaSuccess ||
        cudaMemcpy(out.d_global_scale, &gs, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        release(out);
        return false;
    }

    out.N = N;
    out.K = K;
    return true;
}

void release(MarlinWeight& w) {
    if (w.qweight)
        cudaFree(w.qweight);
    if (w.scales)
        cudaFree(w.scales);
    if (w.d_global_scale)
        cudaFree(w.d_global_scale);
    w = MarlinWeight{};
}

}  // namespace marlin_w4a16
}  // namespace imp
