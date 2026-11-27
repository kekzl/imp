// gemm_nvfp4_tile_swiglu_quant: packed FP4 and SfAtom bytes of the down input
// equal gemm_nvfp4_tile (up, fp16) + swiglu_quantize_fp16_to_nvfp4_sfatom, per tile plan (data-parallel
// 128x64 / 128x128 / 128x256, stream-K uniform / hybrid / split), ragged M, both act-scale modes, 2^-9 clamp
// blocks.

#include "compute/nvfp4_sfatom_quant.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm_tile.h"
#include "compute/nvfp4_sfatom_owned.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <utility>
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
    NvFP4Owned qr;
    NvFP4SfAtomOwned cw;
    quantize_fp16_to_nvfp4(wt, qr, s);
    convert_nvfp4_to_sfatom(qr, cw, s);
    Dev<uint8_t> act(size_t(M) * K / 2), act_sf(nvfp4_sfatom_size(M, K));
    cudaMemsetAsync(act_sf.p, 0, act_sf.bytes, s);
    quantize_fp16_to_nvfp4_sfatom(x.p, act.p, act_sf.p, M, K, s);
    const size_t ws_bytes = std::max<size_t>(gemm_tile_streamk_workspace_bytes(), 4096 + (size_t(170) << 16));
    Dev<uint8_t> ws(ws_bytes);
    const size_t sf_bytes = nvfp4_sfatom_size(M, N);
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
            swiglu_quantize_fp16_to_nvfp4_sfatom(gate.p, up.p, so.p, pk_ref.p, sf_ref.p, M, N, s, as);
            ASSERT_TRUE(gemm_nvfp4_tile_swiglu_quant({.a_data = act.p,
                                                      .a_sf = act_sf.p,
                                                      .w_data = cw.data,
                                                      .w_sf = cw.scale_factors,
                                                      .alpha = cw.tensor_scale,
                                                      .gate_fp16 = gate.p,
                                                      .q_packed = pk_own.p,
                                                      .q_sf = sf_own.p,
                                                      .q_act_scale = as,
                                                      .M = M,
                                                      .N = N,
                                                      .K = K,
                                                      .stream = s,
                                                      .workspace = wsp,
                                                      .workspace_bytes = wsb}));
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            const auto pr = pk_ref.host(), po = pk_own.host(), sr = sf_ref.host(), so_ = sf_own.host();
            size_t bad_pk = 0, bad_sf = 0;
            std::map<std::pair<int, int>, int> bad_at;  // (row / 128, col / 64) -> mismatched packed bytes
            int r_lo = M, r_hi = -1;
            for (size_t i = 0; i < pr.size(); ++i) {
                if (pr[i] == po[i])
                    continue;
                ++bad_pk;
                const int r = static_cast<int>(i / (N / 2)), c = static_cast<int>(i % (N / 2)) * 2;
                ++bad_at[{r / 128, c / 64}];
                r_lo = std::min(r_lo, r);
                r_hi = std::max(r_hi, r);
            }
            for (const auto& [t, n] : bad_at)  // B-136: which tile the mismatches sit in
                std::printf("B-136 mismatch ws=%d as=%.2f tile m=%d box n=%d bytes=%d rows %d..%d\n", with_ws,
                            as, t.first, t.second, n, r_lo, r_hi);
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
    cw = NvFP4SfAtomOwned{};
    qr = NvFP4Owned{};
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

__global__ void count_ne(const uint32_t* a, const uint32_t* b, size_t n, unsigned int* bad) {
    unsigned int c = 0;
    for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        c += a[i] != b[i];
    if (c)
        atomicAdd(bad, c);
}

// B-136: the plans of BytesEqualUnfusedPath/5 (1000x6144x512) relaunched on the same inputs, bytes against
// the first launch. Arms: tile GEMM D with the workspace (uniform stream-K) / without (data-parallel), fused
// packed output with / without. Without the proxy fence before a stream-K stage release, 18-65 of 20000 fused
// relaunches differed in one split tile. IMP_B136_REPS relaunches per arm.
TEST(Nvfp4TileRepeat, RelaunchesAreByteIdentical) {
    const int M = 1000, N = 6144, K = 512;
    const char* env = std::getenv("IMP_B136_REPS");
    const int reps = env ? std::atoi(env) : 20000;
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    Dev<half> w(size_t(N) * K * 2), x(size_t(M) * K * 2), gate(size_t(M) * N * 2);
    fill_rows<<<1024, 256, 0, s>>>(w.p, N, K, 0x1234u, 0.08f, 0.08f);
    fill_rows<<<1024, 256, 0, s>>>(x.p, M, K, 0x9876u, 2.0f, 2.0f);
    fill_rows<<<1024, 256, 0, s>>>(gate.p, M, N, 0x5151u, 4.0f, 2e-3f);
    int64_t wshape[2] = {N, K};
    Tensor wt(w.p, QType::F16, 2, wshape, true);
    NvFP4Owned qr;
    NvFP4SfAtomOwned cw;
    quantize_fp16_to_nvfp4(wt, qr, s);
    convert_nvfp4_to_sfatom(qr, cw, s);
    Dev<uint8_t> act(size_t(M) * K / 2), act_sf(nvfp4_sfatom_size(M, K));
    cudaMemsetAsync(act_sf.p, 0, act_sf.bytes, s);
    quantize_fp16_to_nvfp4_sfatom(x.p, act.p, act_sf.p, M, K, s);
    const size_t ws_bytes = std::max<size_t>(gemm_tile_streamk_workspace_bytes(), 4096 + (size_t(170) << 16));
    Dev<uint8_t> ws(ws_bytes), ref(size_t(M) * N * 2), cur(size_t(M) * N * 2), sf(nvfp4_sfatom_size(M, N));
    Dev<unsigned int> bad(sizeof(unsigned int) * 4 * reps);
    cudaMemsetAsync(bad.p, 0, bad.bytes, s);
    const auto launch = [&](int arm, uint8_t* out) {
        void* wsp = arm == 0 || arm == 2 ? ws.p : nullptr;
        const size_t wsb = wsp != nullptr ? ws_bytes : 0;
        if (arm >= 2)
            return gemm_nvfp4_tile_swiglu_quant({.a_data = act.p,
                                                 .a_sf = act_sf.p,
                                                 .w_data = cw.data,
                                                 .w_sf = cw.scale_factors,
                                                 .alpha = cw.tensor_scale,
                                                 .gate_fp16 = gate.p,
                                                 .q_packed = out,
                                                 .q_sf = sf.p,
                                                 .q_act_scale = 1.0f,
                                                 .M = M,
                                                 .N = N,
                                                 .K = K,
                                                 .stream = s,
                                                 .workspace = wsp,
                                                 .workspace_bytes = wsb});
        return gemm_nvfp4_tile(act.p, act_sf.p, cw.data, cw.scale_factors, cw.tensor_scale, out, M, N, K, s,
                               -1, wsp, wsb);
    };
    for (int arm = 0; arm < 4; ++arm) {
        const size_t words = (arm >= 2 ? size_t(M) * N / 2 : size_t(M) * N * 2) / 4;
        ASSERT_TRUE(launch(arm, ref.p));
        for (int i = 0; i < reps; ++i) {
            ASSERT_TRUE(launch(arm, cur.p));
            count_ne<<<256, 256, 0, s>>>(reinterpret_cast<const uint32_t*>(ref.p),
                                         reinterpret_cast<const uint32_t*>(cur.p), words,
                                         bad.p + (arm * reps) + i);
        }
    }
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    std::vector<unsigned int> h(size_t(4) * reps);
    ASSERT_EQ(cudaMemcpy(h.data(), bad.p, bad.bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (int arm = 0; arm < 4; ++arm) {
        int launches = 0;
        for (int i = 0; i < reps; ++i)
            launches += h[(arm * reps) + i] != 0;
        std::printf("B-136 arm %d: %d of %d relaunches differ\n", arm, launches, reps);
        EXPECT_EQ(launches, 0) << "arm " << arm;
    }
    cw = NvFP4SfAtomOwned{};
    qr = NvFP4Owned{};
    cudaStreamDestroy(s);
}

}  // namespace
