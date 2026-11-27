// gemm_grouped_nvfp4_decode (16-row grouped NVFP4 MoE GEMM for decode rows): kF16 bit-equal to
// gemm_grouped_nvfp4_tile and within tolerance of the fp64 reference; the fused gate+up+act+quantize
// epilogues bit-equal to tile(gate) + tile(up) + fused_act_quantize (packed bytes and UE4M3 scales of every
// valid row). Timing cases: tests/bench/test_gemm_grouped_nvfp4_decode_bench.cu.

#include "compute/nvfp4_sfatom_quant.h"
#include "compute/gemm_grouped_nvfp4_decode.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "core/tensor.h"
#include "model/model_config.h"
#include "nvfp4_fp64_ref.h"
#include "compute/nvfp4_sfatom_owned.h"
#include "quant/nvfp4_owned.h"
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

#include "test_gemm_grouped_nvfp4_decode_util.cuh"
namespace imp {
namespace {

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
                ref_sf_bytes += nvfp4_sfatom_size(x.M_per[e], c.N);
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
            fused_act_quantize_fp16_to_nvfp4_sfatom_moe(gated ? g16 : nullptr, u16, ref_packed, d_ref_bases,
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

}  // namespace imp
