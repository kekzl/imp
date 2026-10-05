#include "memory/ssm_snapshot_int8.h"

#include "core/logging.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace imp {

namespace {

template <typename T>
__device__ __forceinline__ float load_f(const T* p);
template <>
__device__ __forceinline__ float load_f<float>(const float* p) {
    return *p;
}
template <>
__device__ __forceinline__ float load_f<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}
template <>
__device__ __forceinline__ float load_f<__half>(const __half* p) {
    return __half2float(*p);
}

template <typename T>
__device__ __forceinline__ void store_f(T* p, float v);
template <>
__device__ __forceinline__ void store_f<float>(float* p, float v) {
    *p = v;
}
template <>
__device__ __forceinline__ void store_f<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ void store_f<__half>(__half* p, float v) {
    *p = __float2half_rn(v);
}

// One warp per (layer, row): absmax over row_len, scale = absmax / 127, q = rint(x / scale).
template <typename T>
__global__ void snapshot_int8_encode_kernel(const char* __restrict__ slab, char* __restrict__ packed,
                                            int64_t rows, int row_len, size_t slab_layer, size_t slab_conv,
                                            size_t pk_layer, size_t pk_conv, size_t pk_hq, int n_layers) {
    const int64_t warp_id = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (warp_id >= rows * n_layers)
        return;
    const int layer = static_cast<int>(warp_id / rows);
    const int64_t row = warp_id % rows;
    const T* src = reinterpret_cast<const T*>(slab + layer * slab_layer + slab_conv) + row * row_len;
    int8_t* q = reinterpret_cast<int8_t*>(packed + layer * pk_layer + pk_conv) + row * row_len;
    float* scale = reinterpret_cast<float*>(packed + layer * pk_layer + pk_conv + pk_hq) + row;
    float amax = 0.0f;
    for (int i = lane; i < row_len; i += 32)
        amax = fmaxf(amax, fabsf(load_f(src + i)));
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int i = lane; i < row_len; i += 32)
        q[i] = static_cast<int8_t>(__float2int_rn(fminf(fmaxf(load_f(src + i) * inv, -127.0f), 127.0f)));
    if (lane == 0)
        *scale = s;
}

template <typename T>
__global__ void snapshot_int8_decode_kernel(const char* __restrict__ packed, char* __restrict__ slab,
                                            int64_t rows, int row_len, size_t slab_layer, size_t slab_conv,
                                            size_t pk_layer, size_t pk_conv, size_t pk_hq, int n_layers) {
    const int64_t warp_id = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (warp_id >= rows * n_layers)
        return;
    const int layer = static_cast<int>(warp_id / rows);
    const int64_t row = warp_id % rows;
    T* dst = reinterpret_cast<T*>(slab + layer * slab_layer + slab_conv) + row * row_len;
    const int8_t* q = reinterpret_cast<const int8_t*>(packed + layer * pk_layer + pk_conv) + row * row_len;
    const float s = *(reinterpret_cast<const float*>(packed + layer * pk_layer + pk_conv + pk_hq) + row);
    for (int i = lane; i < row_len; i += 32)
        store_f(dst + i, static_cast<float>(q[i]) * s);
}

template <bool kEncode>
bool run(const void* in, void* out, const SsmStateGeometry& g, cudaStream_t stream) {
    const SsmSnapshotInt8Layout l = ssm_snapshot_int8_layout(g);
    if (l.total == 0 || in == nullptr || out == nullptr)
        return false;
    const size_t slab_layer = ssm_bytes_per_layer(g);
    const size_t slab_conv = ssm_conv_bytes_per_layer(g);
    const size_t slab_layers = slab_layer * static_cast<size_t>(g.n_ssm_layers);
    // conv windows: one strided copy over every layer; the per-slot tail: one copy.
    const char* src = static_cast<const char*>(in);
    char* dst = static_cast<char*>(out);
    const size_t src_pitch = kEncode ? slab_layer : l.per_layer;
    const size_t dst_pitch = kEncode ? l.per_layer : slab_layer;
    if (cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, l.conv, static_cast<size_t>(g.n_ssm_layers),
                          cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
        return false;
    if (l.extra > 0) {
        const size_t src_off = kEncode ? slab_layers : l.per_layer * g.n_ssm_layers;
        const size_t dst_off = kEncode ? l.per_layer * g.n_ssm_layers : slab_layers;
        if (cudaMemcpyAsync(dst + dst_off, src + src_off, g.extra_bytes_per_slot, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess)
            return false;
    }
    const int64_t warps = l.rows * g.n_ssm_layers;
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((warps * 32 + threads - 1) / threads);
#define IMP_SNAP_LAUNCH(T)                                                                           \
    do {                                                                                             \
        if (kEncode)                                                                                 \
            snapshot_int8_encode_kernel<T>                                                           \
                <<<blocks, threads, 0, stream>>>(src, dst, l.rows, l.row_len, slab_layer, slab_conv, \
                                                 l.per_layer, l.conv, l.h_q, g.n_ssm_layers);        \
        else                                                                                         \
            snapshot_int8_decode_kernel<T>                                                           \
                <<<blocks, threads, 0, stream>>>(src, dst, l.rows, l.row_len, slab_layer, slab_conv, \
                                                 l.per_layer, l.conv, l.h_q, g.n_ssm_layers);        \
    } while (0)
    if (g.h_dtype == QType::BF16)
        IMP_SNAP_LAUNCH(__nv_bfloat16);
    else if (g.h_dtype == QType::F16)
        IMP_SNAP_LAUNCH(__half);
    else
        IMP_SNAP_LAUNCH(float);
#undef IMP_SNAP_LAUNCH
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool ssm_snapshot_int8_encode(const void* slab, void* packed, const SsmStateGeometry& g,
                              cudaStream_t stream) {
    return run<true>(slab, packed, g, stream);
}

bool ssm_snapshot_int8_decode(const void* packed, void* slab, const SsmStateGeometry& g,
                              cudaStream_t stream) {
    return run<false>(packed, slab, g, stream);
}

}  // namespace imp
