// Timing cases of tests/test_gemm_grouped_nvfp4_decode.cu, built into test-bench (make test-bench), not into
// the default GPU run.
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_grouped_nvfp4_decode.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "core/tensor.h"
#include "model/model_config.h"
#include "nvfp4_fp64_ref.h"
#include "compute/cutlass_nvfp4_owned.h"
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
            // Unfused down input is SfAtom (128-row slabs per active expert), not the decode M16 layout:
            // sized by act_bytes the act quantize overran it and the down tile faulted, misaligned address
            // (Gemma-4, 4 tokens: 163344 B needed, 35200 held).
            size_t sfa_bytes = 16;
            for (int v : x.M_per)
                sfa_bytes += cutlass_nvfp4_sf_size(v, mm.eff);
            cudaMalloc(&g16, size_t(rows) * mm.eff * 2);
            cudaMalloc(&u16, size_t(rows) * mm.eff * 2);
            cudaMalloc(&dpk, size_t(rows) * mm.eff / 2 + 16);
            cudaMalloc(&dsf, sfa_bytes);
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
                float ms = 0.0f;
                // A device fault fails here: cudaEventElapsedTime after a sticky error segfaults in the WSL
                // driver.
                const cudaError_t sync = cudaEventSynchronize(e1);
                if (sync != cudaSuccess)
                    ADD_FAILURE() << "kernel fault in the timed loop: " << cudaGetErrorString(sync);
                else
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
                bzero_sfa_active(dsf, d_off2, 128, sfa_bytes, s_);
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
