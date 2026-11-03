// gemm_nvfp4_tile_swiglu_quant (gemm.swiglu_quant_epilogue): packed FP4 and SfAtom bytes of the down input
// equal gemm_nvfp4_tile (up, fp16) + swiglu_quantize_fp16_to_nvfp4_cutlass, per tile plan (data-parallel
// 128x64 / 128x128 / 128x256, stream-K uniform / hybrid / split), ragged M, both act-scale modes, 2^-9 clamp
// blocks.

#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm_tile.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

using namespace imp;

namespace {

__device__ __forceinline__ float hash_unit(size_t i, uint32_t seed) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return (static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f;
}

// Every 5th row (from row 2) at amplitude tiny: its SwiGLU blocks fall under 2^-9 x 6 x act scale (clamp).
__global__ void fill_rows(half* p, int rows, int cols, uint32_t seed, float amp, float tiny) {
    const size_t n = static_cast<size_t>(rows) * cols;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(i / cols);
        p[i] = __float2half(((r % 5) == 2 ? tiny : amp) * hash_unit(i, seed));
    }
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t bytes = 0;
    explicit Dev(size_t b) : bytes(b) { EXPECT_EQ(cudaMalloc(&p, std::max<size_t>(b, 1)), cudaSuccess); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<uint8_t> host() const {
        std::vector<uint8_t> h(bytes);
        EXPECT_EQ(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
        return h;
    }
};

class Nvfp4TileSwigluQuant : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(Nvfp4TileSwigluQuant, BytesEqualUnfusedPath) {
    const auto [M, N, K] = GetParam();
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    Dev<half> w(size_t(N) * K * 2), x(size_t(M) * K * 2), gate(size_t(M) * N * 2), up(size_t(M) * N * 2),
        so(size_t(M) * N * 2);
    fill_rows<<<1024, 256, 0, s>>>(w.p, N, K, 0x1234u, 0.08f, 0.08f);
    fill_rows<<<1024, 256, 0, s>>>(x.p, M, K, 0x9876u, 2.0f, 2.0f);
    fill_rows<<<1024, 256, 0, s>>>(gate.p, M, N, 0x5151u, 4.0f, 2e-3f);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    int64_t wshape[2] = {N, K};
    Tensor wt(w.p, QType::F16, 2, wshape, true);
    NvFP4QuantResult qr;
    CutlassNvFP4Weight cw;
    quantize_fp16_to_nvfp4(wt, qr, s);
    convert_nvfp4_to_cutlass(qr, cw, s);
    Dev<uint8_t> act(size_t(M) * K / 2), act_sf(cutlass_nvfp4_sf_size(M, K));
    cudaMemsetAsync(act_sf.p, 0, act_sf.bytes, s);
    quantize_fp16_to_nvfp4_cutlass(x.p, act.p, act_sf.p, M, K, s);
    const size_t ws_bytes = std::max<size_t>(gemm_tile_streamk_workspace_bytes(), 4096 + (size_t(170) << 16));
    Dev<uint8_t> ws(ws_bytes);
    const size_t sf_bytes = cutlass_nvfp4_sf_size(M, N);
    Dev<uint8_t> pk_ref(size_t(M) * N / 2), sf_ref(sf_bytes), pk_own(size_t(M) * N / 2), sf_own(sf_bytes);

    int clamp_blocks = 0;
    for (const bool with_ws : {false, true}) {
        for (const float as : {1.0f, 0.37f}) {  // no act scale / a static checkpoint act scale
            for (auto* b : {&pk_ref, &sf_ref, &pk_own, &sf_own})
                cudaMemsetAsync(b->p, 0xA5, b->bytes, s);  // rows past M: untouched in both
            void* wsp = with_ws ? ws.p : nullptr;
            const size_t wsb = with_ws ? ws_bytes : 0;
            ASSERT_TRUE(gemm_nvfp4_tile(act.p, act_sf.p, cw.data, cw.scale_factors, cw.tensor_scale, up.p, M,
                                        N, K, s, -1, wsp, wsb));
            swiglu_quantize_fp16_to_nvfp4_cutlass(gate.p, up.p, so.p, pk_ref.p, sf_ref.p, M, N, s, as);
            ASSERT_TRUE(gemm_nvfp4_tile_swiglu_quant(act.p, act_sf.p, cw.data, cw.scale_factors,
                                                     cw.tensor_scale, gate.p, pk_own.p, sf_own.p, as, M, N, K,
                                                     s, wsp, wsb));
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            const auto pr = pk_ref.host(), po = pk_own.host(), sr = sf_ref.host(), so_ = sf_own.host();
            size_t bad_pk = 0, bad_sf = 0;
            for (size_t i = 0; i < pr.size(); ++i)
                bad_pk += pr[i] != po[i];
            for (size_t i = 0; i < sr.size(); ++i) {
                bad_sf += sr[i] != so_[i];
                clamp_blocks += sr[i] == 0x01;  // UE4M3 2^-9
            }
            EXPECT_EQ(bad_pk, 0u) << "M=" << M << " N=" << N << " K=" << K << " ws=" << with_ws
                                  << " as=" << as;
            EXPECT_EQ(bad_sf, 0u) << "M=" << M << " N=" << N << " K=" << K << " ws=" << with_ws
                                  << " as=" << as;
        }
    }
    EXPECT_GT(clamp_blocks, 0) << "no block reached the 2^-9 clamp";
    free_cutlass_nvfp4_weight(cw);
    cudaFree(qr.packed_data);
    cudaFree(qr.micro_scales);
    cudaStreamDestroy(s);
}

// Plans at auto pick, with / without workspace: 128x64 split (200x1024), hybrid stream-K (256x4096), uniform
// stream-K (512x12288), data-parallel 128x128 / 128x256 (1000x6144, 2048x12288); K 576 = partial last stage,
// N 1040 = a partial last N tile.
INSTANTIATE_TEST_SUITE_P(Shapes, Nvfp4TileSwigluQuant,
                         ::testing::Values(std::make_tuple(33, 1024, 512), std::make_tuple(200, 1024, 1024),
                                           std::make_tuple(129, 1040, 576), std::make_tuple(256, 4096, 512),
                                           std::make_tuple(512, 12288, 1024),
                                           std::make_tuple(1000, 6144, 512),
                                           std::make_tuple(2048, 12288, 1024)));

}  // namespace
