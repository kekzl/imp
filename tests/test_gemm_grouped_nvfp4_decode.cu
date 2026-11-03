// gemm_grouped_nvfp4_decode (16-row grouped NVFP4 MoE GEMM for decode rows): kF16 bit-equal to
// gemm_grouped_nvfp4_tile and within tolerance of the fp64 reference; the fused gate+up+act+quantize
// epilogues bit-equal to tile(gate) + tile(up) + fused_act_quantize (packed bytes and UE4M3 scales of every
// valid row); plus a decode-routing timing table on Gemma-4 / Qwen3-30B expert shapes (informational).

#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_grouped_nvfp4_decode.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "core/tensor.h"
#include "model/model_config.h"
#include "nvfp4_fp64_ref.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace imp {
namespace {

__global__ void fill_kernel(half* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half(amp * ((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f));
    }
}

// ne experts of one projection [N, K]: packed FP4 + SfAtom SF slabs, device pointer arrays, alpha.
struct Weights {
    int ne = 0, N = 0, K = 0;
    size_t stride = 0, sf_stride = 0;
    void *b = nullptr, *sf = nullptr;
    const void** d_b = nullptr;
    const void** d_sf = nullptr;
    float* d_alpha = nullptr;
    std::vector<float> alpha;

    void build(int ne_, int n, int k, uint32_t seed, float amp, cudaStream_t s) {
        ne = ne_;
        N = n;
        K = k;
        stride = size_t(N) * K / 2;
        sf_stride = cutlass_nvfp4_sf_size(N, K);
        cudaMalloc(&b, stride * ne);
        cudaMalloc(&sf, sf_stride * ne);
        half* w = nullptr;
        cudaMalloc(&w, size_t(N) * K * sizeof(half));
        alpha.resize(ne);
        for (int e = 0; e < ne; ++e) {
            fill_kernel<<<256, 256, 0, s>>>(w, size_t(N) * K, seed + e,
                                            amp * static_cast<float>(1 + (e % 4)));
            int64_t ws[2] = {N, K};
            Tensor wt(w, QType::F16, 2, ws, true);
            NvFP4QuantResult q;
            quantize_fp16_to_nvfp4(wt, q, s);
            CutlassNvFP4Weight cw;
            convert_nvfp4_to_cutlass(q, cw, s);
            cudaMemcpyAsync(static_cast<char*>(b) + e * stride, cw.data, stride, cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(static_cast<char*>(sf) + e * sf_stride, cw.scale_factors, sf_stride,
                            cudaMemcpyDeviceToDevice, s);
            alpha[e] = cw.tensor_scale;
            cudaStreamSynchronize(s);
            free_cutlass_nvfp4_weight(cw);
            free_nvfp4_result(q);
        }
        cudaFree(w);
        set_ptrs(b, sf);
    }
    // Pointer arrays into (base, sf_base), e.g. a copy of the slabs at another address.
    void set_ptrs(const void* base, const void* sf_base) {
        std::vector<const void*> hb(ne), hs(ne);
        for (int e = 0; e < ne; ++e) {
            hb[e] = static_cast<const char*>(base) + e * stride;
            hs[e] = static_cast<const char*>(sf_base) + e * sf_stride;
        }
        if (d_b == nullptr) {
            cudaMalloc(&d_b, ne * sizeof(void*));
            cudaMalloc(&d_sf, ne * sizeof(void*));
            cudaMalloc(&d_alpha, ne * sizeof(float));
        }
        cudaMemcpy(d_b, hb.data(), ne * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sf, hs.data(), ne * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_alpha, alpha.data(), ne * sizeof(float), cudaMemcpyHostToDevice);
    }
    void release() {
        for (void* p :
             {b, sf, static_cast<void*>(d_b), static_cast<void*>(d_sf), static_cast<void*>(d_alpha)})
            cudaFree(p);
    }
};

// Expert-sorted activations [M_total, K] as NVFP4 + SfAtom slabs (the gate/up input of the MoE path).
struct Acts {
    int ne = 0, K = 0, M_total = 0;
    std::vector<int> M_per, off;
    std::vector<int64_t> sfa_off;
    void *a = nullptr, *a_sf = nullptr;
    int32_t *d_M = nullptr, *d_off = nullptr;
    int64_t* d_sfa_off = nullptr;

    void build(const std::vector<int>& m_per, int k, cudaStream_t s) {
        M_per = m_per;
        ne = static_cast<int>(m_per.size());
        K = k;
        off.assign(ne + 1, 0);
        sfa_off.assign(ne + 1, 0);
        for (int e = 0; e < ne; ++e) {
            off[e + 1] = off[e] + M_per[e];
            sfa_off[e + 1] = sfa_off[e] + static_cast<int64_t>(cutlass_nvfp4_sf_size(M_per[e], K));
        }
        M_total = off[ne];
        half* x = nullptr;
        cudaMalloc(&x, size_t(M_total) * K * sizeof(half));
        fill_kernel<<<1024, 256, 0, s>>>(x, size_t(M_total) * K, 0x77u, 1.0f);
        cudaMalloc(&a, size_t(M_total) * K / 2 + 16);
        cudaMalloc(&a_sf, sfa_off[ne] + 16);
        cudaMemsetAsync(a_sf, 0, sfa_off[ne] + 16, s);
        for (int e = 0; e < ne; ++e)
            if (M_per[e] > 0)
                quantize_fp16_to_nvfp4_cutlass(x + size_t(off[e]) * K,
                                               static_cast<char*>(a) + size_t(off[e]) * K / 2,
                                               static_cast<char*>(a_sf) + sfa_off[e], M_per[e], K, s);
        cudaMalloc(&d_M, ne * sizeof(int32_t));
        cudaMalloc(&d_off, (ne + 1) * sizeof(int32_t));
        cudaMalloc(&d_sfa_off, (ne + 1) * sizeof(int64_t));
        cudaMemcpy(d_M, M_per.data(), ne * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_off, off.data(), (ne + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sfa_off, sfa_off.data(), (ne + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaStreamSynchronize(s);
        cudaFree(x);
    }
    void release() {
        for (void* p :
             {a, a_sf, static_cast<void*>(d_M), static_cast<void*>(d_off), static_cast<void*>(d_sfa_off)})
            cudaFree(p);
    }
};

GroupedNvfp4DeviceArgs tile_args(const Acts& x, const Weights& w, void* d) {
    GroupedNvfp4DeviceArgs g{};
    g.d_M_per = x.d_M;
    g.d_expert_offsets = x.d_off;
    g.d_sfa_offsets = x.d_sfa_off;
    g.d_alpha = w.d_alpha;
    g.base_A_packed = x.a;
    g.base_A_sf = x.a_sf;
    g.d_B_ptrs = w.d_b;
    g.d_SFB_ptrs = w.d_sf;
    g.base_D = d;
    return g;
}

GroupedNvfp4DecodeArgs dec_args(const Acts& x, const Weights& up, const Weights* gate, void* d, void* d_sf,
                                bool own_sfa_offsets) {
    GroupedNvfp4DecodeArgs g{};
    g.d_M_per = x.d_M;
    g.d_expert_offsets = x.d_off;
    g.d_sfa_offsets = own_sfa_offsets ? nullptr : x.d_sfa_off;
    g.base_A_packed = x.a;
    g.base_A_sf = x.a_sf;
    g.d_B_ptrs = up.d_b;
    g.d_SFB_ptrs = up.d_sf;
    g.d_alpha = up.d_alpha;
    if (gate != nullptr) {
        g.d_G_ptrs = gate->d_b;
        g.d_SFG_ptrs = gate->d_sf;
        g.d_galpha = gate->d_alpha;
    }
    g.base_D = d;
    g.base_D_sf = d_sf;
    return g;
}

// T decode tokens, each routed to top_k distinct experts (skewed popularity), as rows per expert.
std::vector<int> decode_routing(int ne, int tokens, int top_k, uint32_t seed) {
    std::mt19937 gen(seed);
    std::vector<double> wgt(ne);
    for (int e = 0; e < ne; ++e)
        wgt[e] = 1.0 / std::pow(1.0 + ((e * 37) % ne), 0.6);
    std::vector<int> m(ne, 0);
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> w = wgt;
        for (int k = 0; k < top_k; ++k) {
            std::discrete_distribution<int> d(w.begin(), w.end());
            const int e = d(gen);
            ++m[e];
            w[e] = 0.0;
        }
    }
    return m;
}

class GroupedNvfp4DecodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&s_);
        gemm_grouped_nvfp4_decode_prewarm();
    }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

std::vector<half> fetch(const void* d, size_t n) {
    std::vector<half> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost);
    return h;
}

struct Shape {
    int N, K;
};

}  // namespace

// kF16 rows of 1..32 per expert (one and two M tiles, empty experts), decode routing over 128 experts, the
// model down shapes (Gemma-4 K 704 ends on a half stage): bit-equal to the tile kernel, within 1e-2 of fp64.
TEST_F(GroupedNvfp4DecodeTest, F16BitEqualToTileAndFp64) {
    const std::vector<std::vector<int>> routings = {{1, 0, 7, 16, 17, 32, 3, 0, 9},
                                                    decode_routing(128, 16, 8, 3),
                                                    decode_routing(128, 4, 8, 5)};
    for (const Shape sh : {Shape{384, 320}, Shape{2816, 704}, Shape{2048, 768}})
        for (size_t ri = 0; ri < routings.size(); ++ri) {
            const auto& m = routings[ri];
            if (sh.N > 1000 && ri == 0)
                continue;
            Acts x;
            x.build(m, sh.K, s_);
            Weights w;
            w.build(x.ne, sh.N, sh.K, 0x100u, 0.05f, s_);
            const size_t n_out = size_t(x.M_total) * sh.N;
            void *d_tile = nullptr, *d_dec = nullptr;
            cudaMalloc(&d_tile, n_out * sizeof(half));
            cudaMalloc(&d_dec, n_out * sizeof(half));
            cudaMemset(d_tile, 0xff, n_out * sizeof(half));
            cudaMemset(d_dec, 0xff, n_out * sizeof(half));
            ASSERT_TRUE(gemm_grouped_nvfp4_tile(x.ne, sh.N, sh.K, x.M_total, tile_args(x, w, d_tile), s_));
            ASSERT_TRUE(gemm_grouped_nvfp4_decode(x.ne, sh.N, sh.K, x.M_total, MoeDecodeEpi::kF16,
                                                  dec_args(x, w, nullptr, d_dec, nullptr, false), s_));
            ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
            {
                const std::vector<half> ht = fetch(d_tile, n_out), hd = fetch(d_dec, n_out);
                size_t mism = 0;
                for (size_t i = 0; i < n_out; ++i)
                    mism += __half_as_ushort(ht[i]) != __half_as_ushort(hd[i]) ? 1 : 0;
                EXPECT_EQ(mism, 0u) << "N=" << sh.N << " K=" << sh.K << " routing " << ri;
            }
            nvfp4_ref::GroupedHost h;
            h.ne = x.ne;
            h.N = sh.N;
            h.K = sh.K;
            h.offsets = x.off;
            h.sfa_off.assign(x.sfa_off.begin(), x.sfa_off.end() - 1);
            h.alpha = w.alpha;
            h.a.resize(size_t(x.M_total) * sh.K / 2);
            h.a_sf.resize(static_cast<size_t>(x.sfa_off[x.ne]));
            cudaMemcpy(h.a.data(), x.a, h.a.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(h.a_sf.data(), x.a_sf, h.a_sf.size(), cudaMemcpyDeviceToHost);
            h.b.resize(x.ne);
            h.b_sf.resize(x.ne);
            for (int e = 0; e < x.ne; ++e) {
                h.b[e].resize(w.stride);
                h.b_sf[e].resize(w.sf_stride);
                if (x.M_per[e] == 0)
                    continue;
                cudaMemcpy(h.b[e].data(), static_cast<const char*>(w.b) + e * w.stride, w.stride,
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(h.b_sf[e].data(), static_cast<const char*>(w.sf) + e * w.sf_stride, w.sf_stride,
                           cudaMemcpyDeviceToHost);
            }
            const nvfp4_ref::GroupedErr err = nvfp4_ref::check_grouped(h, fetch(d_dec, n_out), 1e-2);
            std::fprintf(stderr, "[decode f16] N=%d K=%d routing %zu rows %d: bad %zu max_rel %.3e\n", sh.N,
                         sh.K, ri, x.M_total, err.bad, err.max_rel);
            EXPECT_EQ(err.checked, n_out);
            EXPECT_EQ(err.bad, 0u);
            cudaFree(d_tile);
            cudaFree(d_dec);
            w.release();
            x.release();
        }
}

// Gated (SwiGLU, GeGLU) and ReLU^2 epilogues against the 4-launch path they replace: tile(gate), tile(up),
// fused_act_quantize. Every valid row's packed bytes and UE4M3 scales must match bit for bit. up_amp 4e-5
// puts most down-input blocks under the unit-scale UE4M3 floor (scale < 2^-9: both clamp to 2^-9, never write
// 0).
TEST_F(GroupedNvfp4DecodeTest, ActEpilogueBitEqualToUnfusedPath) {
    struct Case {
        MoeDecodeEpi epi;
        FFNActivation act;
        int N, K;
        float up_amp = 0.08f;
    };
    for (const Case c : {Case{MoeDecodeEpi::kGeglu, FFNActivation::GEGLU, 704, 2816},
                         Case{MoeDecodeEpi::kSwiglu, FFNActivation::SWIGLU, 768, 2048},
                         Case{MoeDecodeEpi::kRelu2, FFNActivation::RELU_SQR, 1856, 2688},
                         Case{MoeDecodeEpi::kSwiglu, FFNActivation::SWIGLU, 128, 320},
                         Case{MoeDecodeEpi::kGeglu, FFNActivation::GEGLU, 704, 2816, 4e-5f},
                         Case{MoeDecodeEpi::kSwiglu, FFNActivation::SWIGLU, 768, 2048, 4e-5f}})
        for (int tokens : {1, 4, 16, 32}) {
            const bool gated = c.epi != MoeDecodeEpi::kRelu2;
            Acts x;
            x.build(decode_routing(128, tokens, 8, 11 + tokens), c.K, s_);
            Weights up, gate;
            up.build(x.ne, c.N, c.K, 0x500u, c.up_amp, s_);
            if (gated)
                gate.build(x.ne, c.N, c.K, 0x900u, 0.08f, s_);
            const size_t n_mid = size_t(x.M_total) * c.N;
            half *g16 = nullptr, *u16 = nullptr;
            cudaMalloc(&g16, n_mid * sizeof(half));
            cudaMalloc(&u16, n_mid * sizeof(half));
            if (gated)
                ASSERT_TRUE(gemm_grouped_nvfp4_tile(x.ne, c.N, c.K, x.M_total, tile_args(x, gate, g16), s_));
            ASSERT_TRUE(gemm_grouped_nvfp4_tile(x.ne, c.N, c.K, x.M_total, tile_args(x, up, u16), s_));
            // Reference down input: SfAtom slabs at compute_sfa_offsets_device(K = N).
            int64_t* d_ref_off = nullptr;
            uint8_t** d_ref_bases = nullptr;
            size_t ref_sf_bytes = 0;
            std::vector<size_t> m16_off(x.ne + 1, 0);  // M16 layout: ceil(M_e / 16) tiles x N/64 atoms x 64 B
            for (int e = 0; e < x.ne; ++e) {
                ref_sf_bytes += cutlass_nvfp4_sf_size(x.M_per[e], c.N);
                m16_off[e + 1] = m16_off[e] + (size_t((x.M_per[e] + 15) / 16) * ((c.N + 63) / 64) * 64);
            }
            const size_t act_bytes = gemm_grouped_nvfp4_decode_act_bytes(x.ne, x.M_total, c.N);
            const size_t sf_at = gemm_grouped_nvfp4_decode_act_sf_offset(x.M_total, c.N);
            ASSERT_LE(sf_at + m16_off[x.ne], act_bytes);
            ASSERT_LE(act_bytes, n_mid * sizeof(half));  // fits the FP16 up buffer it replaces
            void *ref_packed = nullptr, *ref_sf = nullptr, *dec = nullptr;
            cudaMalloc(&d_ref_off, (x.ne + 1) * sizeof(int64_t));
            cudaMalloc(&d_ref_bases, x.ne * sizeof(uint8_t*));
            cudaMalloc(&ref_packed, size_t(x.M_total) * c.N / 2);
            cudaMalloc(&ref_sf, ref_sf_bytes + 16);
            cudaMalloc(&dec, act_bytes);
            cudaMemset(ref_sf, 0, ref_sf_bytes + 16);
            cudaMemset(dec, 0, act_bytes);
            compute_sfa_offsets_device(x.d_M, d_ref_off, x.ne, c.N, s_, d_ref_bases, ref_sf);
            fused_act_quantize_fp16_to_nvfp4_cutlass_moe(gated ? g16 : nullptr, u16, ref_packed, d_ref_bases,
                                                         x.d_off, x.M_total, c.N, x.ne, c.act, s_);
            ASSERT_TRUE(gemm_grouped_nvfp4_decode(x.ne, c.N, c.K, x.M_total, c.epi,
                                                  dec_args(x, up, gated ? &gate : nullptr, dec,
                                                           static_cast<char*>(dec) + sf_at, false),
                                                  s_));
            ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
            std::vector<uint8_t> rp(size_t(x.M_total) * c.N / 2), dp(rp.size()), rs(ref_sf_bytes),
                ds(act_bytes - sf_at);
            std::vector<int64_t> roff(x.ne + 1);
            cudaMemcpy(rp.data(), ref_packed, rp.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(dp.data(), dec, dp.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(rs.data(), ref_sf, ref_sf_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(ds.data(), static_cast<char*>(dec) + sf_at, ds.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(roff.data(), d_ref_off, (x.ne + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);
            size_t bad_p = 0, bad_s = 0, nonzero = 0, sub_floor = 0;
            for (size_t i = 0; i < rp.size(); ++i) {
                bad_p += rp[i] != dp[i] ? 1 : 0;
                nonzero += rp[i] != 0 ? 1 : 0;
            }
            const int n_k_tiles = (c.N + 63) / 64;
            for (int e = 0; e < x.ne; ++e)
                for (int r = 0; r < x.M_per[e]; ++r)
                    for (int kg = 0; kg < c.N / 16; ++kg) {
                        const size_t o = static_cast<size_t>(roff[e]) +
                                         nvfp4_ref::sfatom_offset(r, kg, n_k_tiles);
                        bad_s += rs[o] != ds[m16_off[e] + gemm_grouped_nvfp4_decode_m16_sf(r, kg, c.N)] ? 1
                                                                                                        : 0;
                        sub_floor += rs[o] <= 0x01 ? 1 : 0;  // byte 0x01 = 2^-9, the floor
                    }
            // Down projection on each input: tile GEMM on the reference (SfAtom) vs decode GEMM on the M16
            // output.
            size_t bad_d = 0;
            if (gemm_grouped_nvfp4_decode_supports(x.ne, c.K, c.N, MoeDecodeEpi::kF16)) {
                Weights dn;
                dn.build(x.ne, c.K, c.N, 0xd00u, 0.05f, s_);
                Acts ref_in = x, dec_in = x;
                ref_in.a = ref_packed;
                ref_in.a_sf = ref_sf;
                ref_in.d_sfa_off = d_ref_off;
                dec_in.a = dec;
                dec_in.a_sf = static_cast<char*>(dec) + sf_at;
                const size_t n_out = size_t(x.M_total) * c.K;
                void *o_ref = nullptr, *o_dec = nullptr;
                cudaMalloc(&o_ref, n_out * sizeof(half));
                cudaMalloc(&o_dec, n_out * sizeof(half));
                ASSERT_TRUE(
                    gemm_grouped_nvfp4_tile(x.ne, c.K, c.N, x.M_total, tile_args(ref_in, dn, o_ref), s_));
                ASSERT_TRUE(gemm_grouped_nvfp4_decode(x.ne, c.K, c.N, x.M_total, MoeDecodeEpi::kF16,
                                                      dec_args(dec_in, dn, nullptr, o_dec, nullptr, true),
                                                      s_));
                ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
                const std::vector<half> hr = fetch(o_ref, n_out), hd = fetch(o_dec, n_out);
                for (size_t i = 0; i < n_out; ++i)
                    bad_d += __half_as_ushort(hr[i]) != __half_as_ushort(hd[i]) ? 1 : 0;
                cudaFree(o_ref);
                cudaFree(o_dec);
                dn.release();
            }
            std::fprintf(
                stderr,
                "[decode act] epi %d N=%d K=%d tokens %d rows %d: packed bad %zu / %zu (nonzero %zu), "
                "sf bad %zu (at the 2^-9 floor %zu), down bad %zu\n",
                static_cast<int>(c.epi), c.N, c.K, tokens, x.M_total, bad_p, rp.size(), nonzero, bad_s,
                sub_floor, bad_d);
            EXPECT_GT(nonzero, rp.size() / ((c.up_amp < 0.01f) ? 8 : 2));
            if (c.up_amp < 0.01f)
                EXPECT_GT(sub_floor, size_t(x.M_total) * c.N / 16 / 4) << "tiny case reaches no floor block";
            EXPECT_EQ(bad_p, 0u);
            EXPECT_EQ(bad_s, 0u);
            EXPECT_EQ(bad_d, 0u);
            for (void* p : {static_cast<void*>(g16), static_cast<void*>(u16), static_cast<void*>(d_ref_off),
                            static_cast<void*>(d_ref_bases), ref_packed, ref_sf, dec})
                cudaFree(p);
            up.release();
            if (gated)
                gate.release();
            x.release();
        }
}

// Decode routing (tokens x top-8 over 128 experts), weights rotated over 4 copies per projection (L2 holds
// one layer, not four). Unfused: tile gate + tile up + sfa prep + act-quantize + tile down; fused: decode
// gate_up + decode down. Per-layer us, informational.
TEST_F(GroupedNvfp4DecodeTest, Bench) {
    struct M {
        const char* name;
        int d, eff;
        MoeDecodeEpi epi;
        FFNActivation act;
    };
    constexpr int kCopies = 4;
    std::fprintf(stderr, "%-8s %4s %5s | %7s %7s %7s %7s %7s | %7s %7s %7s\n", "model", "tok", "rows", "gate",
                 "up", "actq", "down", "sum", "gate_up", "down", "sum");
    for (const M mm : {M{"gemma4", 2816, 704, MoeDecodeEpi::kGeglu, FFNActivation::GEGLU},
                       M{"qwen3moe", 2048, 768, MoeDecodeEpi::kSwiglu, FFNActivation::SWIGLU}}) {
        Weights g, u, dn;
        g.build(128, mm.eff, mm.d, 0x10u, 0.05f, s_);
        u.build(128, mm.eff, mm.d, 0x20u, 0.05f, s_);
        dn.build(128, mm.d, mm.eff, 0x30u, 0.05f, s_);
        // kCopies slabs per projection; copy c of every projection lives at its own address.
        std::vector<void*> slabs;
        auto copies = [&](Weights& w) {
            std::vector<std::pair<void*, void*>> v;
            for (int c = 0; c < kCopies; ++c) {
                void *b = nullptr, *sf = nullptr;
                cudaMalloc(&b, w.stride * w.ne);
                cudaMalloc(&sf, w.sf_stride * w.ne);
                cudaMemcpy(b, w.b, w.stride * w.ne, cudaMemcpyDeviceToDevice);
                cudaMemcpy(sf, w.sf, w.sf_stride * w.ne, cudaMemcpyDeviceToDevice);
                slabs.push_back(b);
                slabs.push_back(sf);
                v.emplace_back(b, sf);
            }
            return v;
        };
        const auto cg = copies(g), cu = copies(u), cd = copies(dn);
        for (int tokens : {4, 16, 32, 64}) {
            Acts x;
            x.build(decode_routing(128, tokens, 8, 100 + tokens), mm.d, s_);
            const int rows = x.M_total;
            void *g16 = nullptr, *u16 = nullptr, *dpk = nullptr, *dsf = nullptr, *out = nullptr,
                 *dec = nullptr;
            int64_t* d_off2 = nullptr;
            uint8_t** d_bases = nullptr;
            const size_t act_bytes = gemm_grouped_nvfp4_decode_act_bytes(128, rows, mm.eff);
            cudaMalloc(&g16, size_t(rows) * mm.eff * 2);
            cudaMalloc(&u16, size_t(rows) * mm.eff * 2);
            cudaMalloc(&dpk, size_t(rows) * mm.eff / 2 + 16);
            cudaMalloc(&dsf, act_bytes);
            cudaMalloc(&dec, act_bytes);
            cudaMalloc(&out, size_t(rows) * mm.d * 2);
            cudaMalloc(&d_off2, 129 * sizeof(int64_t));
            cudaMalloc(&d_bases, 128 * sizeof(void*));
            Acts down_in;  // the unfused down GEMM's A view: dpk / dsf at the prep offsets
            down_in = x;
            down_in.a = dpk;
            down_in.a_sf = dsf;
            down_in.d_sfa_off = d_off2;
            down_in.K = mm.eff;
            Acts fused_in = x;  // fused down input: dec packed + in-kernel SF offsets
            fused_in.a = dec;
            fused_in.a_sf = static_cast<char*>(dec) + gemm_grouped_nvfp4_decode_act_sf_offset(rows, mm.eff);
            fused_in.K = mm.eff;
            auto timed = [&](auto&& fn) {
                for (int i = 0; i < 3 * kCopies; ++i)
                    fn(i % kCopies);
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                constexpr int kIters = 40;
                cudaEventRecord(e0, s_);
                for (int i = 0; i < kIters; ++i)
                    fn(i % kCopies);
                cudaEventRecord(e1, s_);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * ms / kIters;
            };
            auto at = [&](Weights& w, const std::pair<void*, void*>& c) {
                w.set_ptrs(c.first, c.second);  // host copy: outside the timed region via the warm loop order
            };
            // Pointer arrays per copy prepared up front (set_ptrs syncs).
            std::vector<Weights> wg(kCopies, g), wu(kCopies, u), wd(kCopies, dn);
            for (int c = 0; c < kCopies; ++c) {
                for (Weights* w : {&wg[c], &wu[c], &wd[c]}) {
                    w->d_b = nullptr;
                    w->d_sf = nullptr;
                    w->d_alpha = nullptr;
                }
                at(wg[c], cg[c]);
                at(wu[c], cu[c]);
                at(wd[c], cd[c]);
            }
            const double t_g = timed([&](int c) {
                (void)gemm_grouped_nvfp4_tile(128, mm.eff, mm.d, rows, tile_args(x, wg[c], g16), s_);
            });
            const double t_u = timed([&](int c) {
                (void)gemm_grouped_nvfp4_tile(128, mm.eff, mm.d, rows, tile_args(x, wu[c], u16), s_);
            });
            const double t_a = timed([&](int) {
                compute_sfa_offsets_device(x.d_M, d_off2, 128, mm.eff, s_, d_bases, dsf);
                bzero_sfa_active(dsf, d_off2, 128, act_bytes, s_);
                fused_act_quantize_fp16_to_nvfp4_cutlass_moe(g16, u16, dpk, d_bases, x.d_off, rows, mm.eff,
                                                             128, mm.act, s_);
            });
            const double t_d = timed([&](int c) {
                (void)gemm_grouped_nvfp4_tile(128, mm.d, mm.eff, rows, tile_args(down_in, wd[c], out), s_);
            });
            const double t_gu = timed([&](int c) {
                (void)gemm_grouped_nvfp4_decode(
                    128, mm.eff, mm.d, rows, mm.epi,
                    dec_args(x, wu[c], &wg[c], dec,
                             static_cast<char*>(dec) + gemm_grouped_nvfp4_decode_act_sf_offset(rows, mm.eff),
                             false),
                    s_);
            });
            const double t_fd = timed([&](int c) {
                (void)gemm_grouped_nvfp4_decode(128, mm.d, mm.eff, rows, MoeDecodeEpi::kF16,
                                                dec_args(fused_in, wd[c], nullptr, out, nullptr, true), s_);
            });
            ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
            int active = 0;
            for (int v : x.M_per)
                active += v > 0 ? 1 : 0;
            std::fprintf(stderr,
                         "%-8s %4d %5d | %7.1f %7.1f %7.1f %7.1f %7.1f | %7.1f %7.1f %7.1f  (experts %d)\n",
                         mm.name, tokens, rows, t_g, t_u, t_a, t_d, t_g + t_u + t_a + t_d, t_gu, t_fd,
                         t_gu + t_fd, active);
            for (int c = 0; c < kCopies; ++c)
                for (Weights* w : {&wg[c], &wu[c], &wd[c]}) {
                    cudaFree(static_cast<void*>(w->d_b));
                    cudaFree(static_cast<void*>(w->d_sf));
                    cudaFree(w->d_alpha);
                }
            for (void* p :
                 {g16, u16, dpk, dsf, out, dec, static_cast<void*>(d_off2), static_cast<void*>(d_bases)})
                cudaFree(p);
            x.release();
        }
        for (void* p : slabs)
            cudaFree(p);
        g.release();
        u.release();
        dn.release();
    }
}

}  // namespace imp
