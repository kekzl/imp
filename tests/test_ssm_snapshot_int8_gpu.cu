// Packed recurrent snapshot round trip on the GPU (#2419): conv window and tail exact, h within
// half an int8 step of its group's absmax.
#include "memory/ssm_snapshot_int8.h"

#include <gtest/gtest.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using imp::QType;
using imp::SsmStateGeometry;

namespace {

template <typename T>
float to_f(T v);
template <>
float to_f<float>(float v) {
    return v;
}
template <>
float to_f<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
T from_f(float v);
template <>
float from_f<float>(float v) {
    return v;
}
template <>
__nv_bfloat16 from_f<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
void round_trip(QType dtype) {
    SsmStateGeometry g{3, 96, 4, 4, 8, 64, dtype, 300};
    const size_t slab_bytes = imp::ssm_bytes_per_slot(g);
    const auto l = imp::ssm_snapshot_int8_layout(g);
    ASSERT_GT(l.total, 0u);
    std::vector<uint8_t> slab(slab_bytes, 0);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t layer = imp::ssm_bytes_per_layer(g), conv = imp::ssm_conv_bytes_per_layer(g);
    const size_t n_h = static_cast<size_t>(g.n_heads) * g.head_dim * g.state_size;
    for (int L = 0; L < g.n_ssm_layers; ++L) {
        for (size_t b = 0; b < conv; ++b)
            slab[L * layer + b] = static_cast<uint8_t>(byte(rng));
        T* h = reinterpret_cast<T*>(slab.data() + L * layer + conv);
        for (size_t i = 0; i < n_h; ++i)  // magnitudes spread over 1e-3 .. 1e2 per group
            h[i] = from_f<T>(nd(rng) * std::pow(10.0f, static_cast<float>((i / 32) % 6) - 3.0f));
    }
    const size_t tail = layer * g.n_ssm_layers;
    for (size_t b = 0; b < g.extra_bytes_per_slot; ++b)
        slab[tail + b] = static_cast<uint8_t>(byte(rng));

    void *d_slab = nullptr, *d_pack = nullptr, *d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_slab, slab_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pack, l.total), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, slab_bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_slab, slab.data(), slab_bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_out, 0xAB, slab_bytes), cudaSuccess);
    ASSERT_TRUE(imp::ssm_snapshot_int8_encode(d_slab, d_pack, g, nullptr));
    ASSERT_TRUE(imp::ssm_snapshot_int8_decode(d_pack, d_out, g, nullptr));
    std::vector<uint8_t> out(slab_bytes);
    ASSERT_EQ(cudaMemcpy(out.data(), d_out, slab_bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_slab);
    cudaFree(d_pack);
    cudaFree(d_out);

    for (int L = 0; L < g.n_ssm_layers; ++L) {
        EXPECT_EQ(std::memcmp(out.data() + L * layer, slab.data() + L * layer, conv), 0)
            << "conv layer " << L;
        const T* a = reinterpret_cast<const T*>(slab.data() + L * layer + conv);
        const T* b = reinterpret_cast<const T*>(out.data() + L * layer + conv);
        for (size_t grp = 0; grp < n_h / 32; ++grp) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i)
                amax = std::max(amax, std::fabs(to_f(a[grp * 32 + i])));
            for (int i = 0; i < 32; ++i) {
                const float err = std::fabs(to_f(a[grp * 32 + i]) - to_f(b[grp * 32 + i]));
                // half an int8 step, plus the BF16 rounding of the decoded value
                const float tol = amax / 254.0f * 1.02f + std::fabs(to_f(a[grp * 32 + i])) / 128.0f + 1e-30f;
                ASSERT_LE(err, tol) << "layer " << L << " group " << grp << " i " << i;
            }
        }
    }
    EXPECT_EQ(std::memcmp(out.data() + tail, slab.data() + tail, g.extra_bytes_per_slot), 0) << "tail";
}

}  // namespace

TEST(SsmSnapshotInt8Gpu, RoundTripF32) { round_trip<float>(QType::F32); }
TEST(SsmSnapshotInt8Gpu, RoundTripBf16) { round_trip<__nv_bfloat16>(QType::BF16); }
