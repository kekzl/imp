// Tile-shape bench for the CUTLASS 3.x NVFP4 grouped GEMM on the Qwen3.6-35B
// MoE prefill geometry (256 experts, gate/up N=512 K=2048, down N=2048
// K=512, ~16 rows per expert at pp512). The shipped 128x128x128 cooperative
// tile pads a 16-row expert to 128 rows (8x the MMA work) and holds one CTA
// per SM; roofline run 1d5b9230 reads the class at 55% of DRAM bandwidth.
// Local instantiations below try smaller N tiles, a deeper K tile and the
// pingpong schedule against the production entry point (the builder rejects
// M=64 tiles: the SF atom is 128 rows). Weights total 134 MB per launch, so every launch reads DRAM.
// GPU required - skips without one. Decision data, not a gate.

#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"
#include "scoped_engine_arena.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

// ---- local grouped instantiation, parameterised by tile + schedule ----
template <class TileShape, class Schedule>
struct GroupedVariant {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using ElemIn = cutlass::float_e2m1_t;
    using ElementA = cutlass::nv_float4_t<ElemIn>;
    using ElementB = cutlass::nv_float4_t<ElemIn>;
    using ElementD = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, Shape<_1, _1, _1>,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementC, cutlass::layout::RowMajor*,
        8, ElementD, cutlass::layout::RowMajor*, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, cutlass::layout::RowMajor*,
        32, ElementB, cutlass::layout::ColumnMajor*, 32, float, TileShape, Shape<_1, _1, _1>,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename Epilogue::SharedStorage))>,
        Schedule>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
    using BlkCfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using ElemSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
    using UShape = typename ProblemShape::UnderlyingProblemShape;

    Gemm gemm;
    bool initialized = false;
    void* workspace = nullptr;
    size_t workspace_sz = 0;
    // device staging (struct of arrays)
    std::vector<char> host_buf;
    char* d_base = nullptr;
    size_t d_base_sz = 0;

    bool run(int ne, const int* M, int N, int K, const void* const* pA, const void* const* pSFA,
             const void* const* pB, const void* const* pSFB, void* const* pD, const float* alpha,
             cudaStream_t stream) {
        auto al = [](size_t x) { return (x + 127) & ~size_t(127); };
        const size_t n = ne;
        size_t o_shape = 0, o_stA = al(n * sizeof(UShape)), o_stB = al(o_stA + n * sizeof(StrideA)),
               o_stC = al(o_stB + n * sizeof(StrideB)), o_stD = al(o_stC + n * sizeof(StrideC)),
               o_lSFA = al(o_stD + n * sizeof(StrideD)), o_lSFB = al(o_lSFA + n * sizeof(LayoutSFA)),
               o_pA = al(o_lSFB + n * sizeof(LayoutSFB)), o_pB = al(o_pA + n * 8), o_pSFA = al(o_pB + n * 8),
               o_pSFB = al(o_pSFA + n * 8), o_pC = al(o_pSFB + n * 8), o_pD = al(o_pC + n * 8),
               o_alpha = al(o_pD + n * 8), o_aPtr = al(o_alpha + n * sizeof(float)),
               total = al(o_aPtr + n * 8);
        if (total > d_base_sz) {
            if (d_base)
                cudaFree(d_base);
            cudaMalloc(&d_base, total);
            d_base_sz = total;
        }
        host_buf.assign(total, 0);
        char* h = host_buf.data();
        for (int i = 0; i < ne; ++i) {
            reinterpret_cast<UShape*>(h + o_shape)[i] = {M[i], N, K};
            reinterpret_cast<StrideA*>(h + o_stA)[i] = cutlass::make_cute_packed_stride(StrideA{},
                                                                                        {M[i], K, 1});
            reinterpret_cast<StrideB*>(h + o_stB)[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
            reinterpret_cast<StrideC*>(h + o_stC)[i] = cutlass::make_cute_packed_stride(StrideC{},
                                                                                        {M[i], N, 1});
            reinterpret_cast<StrideD*>(h + o_stD)[i] = cutlass::make_cute_packed_stride(StrideD{},
                                                                                        {M[i], N, 1});
            reinterpret_cast<LayoutSFA*>(h + o_lSFA)[i] = BlkCfg::tile_atom_to_shape_SFA(
                make_shape(M[i], N, K, 1));
            reinterpret_cast<LayoutSFB*>(h + o_lSFB)[i] = BlkCfg::tile_atom_to_shape_SFB(
                make_shape(M[i], N, K, 1));
            reinterpret_cast<const void**>(h + o_pA)[i] = pA[i];
            reinterpret_cast<const void**>(h + o_pB)[i] = pB[i];
            reinterpret_cast<const void**>(h + o_pSFA)[i] = pSFA[i];
            reinterpret_cast<const void**>(h + o_pSFB)[i] = pSFB[i];
            reinterpret_cast<const void**>(h + o_pC)[i] = pD[i];
            reinterpret_cast<void**>(h + o_pD)[i] = pD[i];
            reinterpret_cast<float**>(h + o_aPtr)[i] = reinterpret_cast<float*>(d_base + o_alpha) + i;
        }
        std::memcpy(h + o_alpha, alpha, n * sizeof(float));
        cudaMemcpyAsync(d_base, h, total, cudaMemcpyHostToDevice, stream);

        typename Gemm::Arguments args;
        decltype(args.epilogue.thread) fusion{};
        fusion.alpha = 0.f;
        fusion.beta = 0.f;
        fusion.alpha_ptr_array = reinterpret_cast<float**>(d_base + o_aPtr);
        fusion.dAlpha = {cute::_0{}, cute::_0{}, 1};
        fusion.dBeta = {cute::_0{}, cute::_0{}, 0};
        cutlass::KernelHardwareInfo hw;
        hw.device_id = 0;
        hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
        typename Gemm::GemmKernel::TileSchedulerArguments sched;
        args = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {ne, reinterpret_cast<UShape*>(d_base + o_shape), reinterpret_cast<UShape*>(h + o_shape)},
            {reinterpret_cast<const typename Gemm::ElementA**>(d_base + o_pA),
             reinterpret_cast<StrideA*>(d_base + o_stA),
             reinterpret_cast<const typename Gemm::ElementB**>(d_base + o_pB),
             reinterpret_cast<StrideB*>(d_base + o_stB), reinterpret_cast<const ElemSF**>(d_base + o_pSFA),
             reinterpret_cast<LayoutSFA*>(d_base + o_lSFA), reinterpret_cast<const ElemSF**>(d_base + o_pSFB),
             reinterpret_cast<LayoutSFB*>(d_base + o_lSFB)},
            {fusion, reinterpret_cast<const typename Gemm::ElementC**>(d_base + o_pC),
             reinterpret_cast<StrideC*>(d_base + o_stC), reinterpret_cast<ElementD**>(d_base + o_pD),
             reinterpret_cast<StrideD*>(d_base + o_stD)},
            hw,
            sched};
        if (gemm.can_implement(args) != cutlass::Status::kSuccess)
            return false;
        size_t need = Gemm::get_workspace_size(args);
        if (need > workspace_sz) {
            if (workspace)
                cudaFree(workspace);
            cudaMalloc(&workspace, need ? need : 1);
            workspace_sz = need;
        }
        cutlass::Status st = initialized ? gemm.update(args, workspace)
                                         : gemm.initialize(args, workspace, stream);
        if (st != cutlass::Status::kSuccess)
            return false;
        initialized = true;
        return gemm.run(stream) == cutlass::Status::kSuccess;
    }
};

using Coop = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using V128x64 = GroupedVariant<Shape<_128, _64, _128>, Coop>;
using V128x64ping = GroupedVariant<Shape<_128, _64, _128>, Ping>;
using V128x128k256 = GroupedVariant<Shape<_128, _128, _256>, Coop>;
using V128x128ping = GroupedVariant<Shape<_128, _128, _128>, Ping>;

// ---- operands ----
struct Experts {
    int ne = 0, N = 0, K = 0;
    std::vector<imp::NvFP4QuantResult> q;
    std::vector<imp::CutlassNvFP4Weight> w;
    std::vector<const void*> pB, pSFB, pA, pSFA;
    std::vector<void*> pD, dA, dSFA;
    std::vector<float> alpha;
    std::vector<int> M;
    half* d_out = nullptr;
    // plain-layout twins for the v2 grouped kernel: expert slab + scales
    imp::NvFP4QuantResult xq_plain{};
    uint8_t* d_w_slab = nullptr;
    uint8_t* d_s_slab = nullptr;
    size_t w_stride = 0, s_stride = 0;
    float* d_w_ts = nullptr;
    int* d_offsets = nullptr;
    int2* d_work = nullptr;
    int work_cap = 0;

    void build(int ne_, int N_, int K_, int m_per, cudaStream_t stream) {
        ne = ne_, N = N_, K = K_;
        q.resize(ne), w.resize(ne);
        std::vector<half> h_w((size_t)N * K);
        void* d_w = nullptr;
        cudaMalloc(&d_w, h_w.size() * sizeof(half));
        for (int e = 0; e < ne; ++e) {
            for (size_t i = 0; i < h_w.size(); ++i)
                h_w[i] = __float2half((float)((int)((i * 7 + e * 13 + 1) % 13) - 6) * 0.05f);
            cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(half), cudaMemcpyHostToDevice);
            int64_t shp[2] = {N, K};
            imp::Tensor t(d_w, imp::QType::F16, 2, shp, true);
            imp::quantize_fp16_to_nvfp4(t, q[e], stream);
            imp::convert_nvfp4_to_cutlass(q[e], w[e], stream);
            cudaStreamSynchronize(stream);
            pB.push_back(w[e].data);
            pSFB.push_back(w[e].scale_factors);
            alpha.push_back(w[e].tensor_scale);
            M.push_back(m_per);
        }
        cudaFree(d_w);
        // activations: per-expert packed + SfAtom slabs
        std::vector<half> h_x((size_t)m_per * K);
        for (size_t i = 0; i < h_x.size(); ++i)
            h_x[i] = __float2half((float)((int)((i * 11 + 3) % 17) - 8) * 0.05f);
        half* d_x = nullptr;
        cudaMalloc(&d_x, h_x.size() * sizeof(half));
        cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMalloc(reinterpret_cast<void**>(&d_out), (size_t)ne * m_per * N * sizeof(half));
        for (int e = 0; e < ne; ++e) {
            void *a = nullptr, *sfa = nullptr;
            size_t sf_bytes = imp::cutlass_nvfp4_sf_size(m_per, K);
            cudaMalloc(&a, (size_t)m_per * K / 2);
            cudaMalloc(&sfa, sf_bytes);
            cudaMemsetAsync(sfa, 0, sf_bytes, stream);
            imp::quantize_fp16_to_nvfp4_cutlass(d_x, a, sfa, m_per, K, stream);
            dA.push_back(a), dSFA.push_back(sfa);
            pA.push_back(a), pSFA.push_back(sfa);
            pD.push_back(d_out + (size_t)e * m_per * N);
        }
        cudaStreamSynchronize(stream);
        // plain twins: one activation matrix [ne*m_per, K] (rows expert-sorted)
        {
            std::vector<half> h_all((size_t)ne * m_per * K);
            for (int e = 0; e < ne; ++e)
                std::copy(h_x.begin(), h_x.end(), h_all.begin() + (size_t)e * m_per * K);
            half* d_all = nullptr;
            cudaMalloc(&d_all, h_all.size() * sizeof(half));
            cudaMemcpy(d_all, h_all.data(), h_all.size() * sizeof(half), cudaMemcpyHostToDevice);
            int64_t shp[2] = {(int64_t)ne * m_per, K};
            imp::Tensor t(d_all, imp::QType::F16, 2, shp, true);
            imp::quantize_fp16_to_nvfp4(t, xq_plain, stream);
            cudaStreamSynchronize(stream);
            cudaFree(d_all);
            std::vector<float> ts(ne);
            std::vector<int> off(ne + 1);
            w_stride = (size_t)N * K / 2;
            s_stride = (size_t)N * K / 16;
            cudaMalloc(reinterpret_cast<void**>(&d_w_slab), w_stride * ne);
            cudaMalloc(reinterpret_cast<void**>(&d_s_slab), s_stride * ne);
            for (int e = 0; e < ne; ++e) {
                cudaMemcpy(d_w_slab + e * w_stride, q[e].packed_data, w_stride, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_s_slab + e * s_stride, q[e].micro_scales, s_stride, cudaMemcpyDeviceToDevice);
                ts[e] = q[e].tensor_scale;
                off[e] = e * m_per;
            }
            off[ne] = ne * m_per;
            cudaMalloc(reinterpret_cast<void**>(&d_w_ts), ne * sizeof(float));
            cudaMalloc(reinterpret_cast<void**>(&d_offsets), (ne + 1) * sizeof(int));
            work_cap = imp::gemm_nvfp4_smallm_v2_grouped_work_cap(ne * m_per, ne);
            cudaMalloc(reinterpret_cast<void**>(&d_work), work_cap * sizeof(int2));
            cudaMemcpy(d_w_ts, ts.data(), ne * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_offsets, off.data(), (ne + 1) * sizeof(int), cudaMemcpyHostToDevice);
        }
        cudaFree(d_x);
    }
    void release() {
        imp::free_nvfp4_result(xq_plain);
        cudaFree(d_w_slab);
        cudaFree(d_s_slab);
        cudaFree(d_w_ts);
        cudaFree(d_offsets);
        cudaFree(d_work);
        for (int e = 0; e < ne; ++e) {
            imp::free_cutlass_nvfp4_weight(w[e]);
            imp::free_nvfp4_result(q[e]);
            cudaFree(dA[e]);
            cudaFree(dSFA[e]);
        }
        cudaFree(d_out);
    }
};

class GroupedTileBench : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no GPU";
        arena_ = std::make_unique<imp::ScopedEngineArena>(64ull << 20);
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        if (stream_)
            cudaStreamDestroy(stream_);
        imp::gemm_grouped_3x_nvfp4_cleanup();
        arena_.reset();
    }
    std::unique_ptr<imp::ScopedEngineArena> arena_;
    cudaStream_t stream_ = nullptr;

    template <class F>
    float time_us(F&& launch) {
        for (int i = 0; i < 5; ++i)
            launch();
        cudaStreamSynchronize(stream_);
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        constexpr int kIters = 20;
        cudaEventRecord(t0, stream_);
        for (int i = 0; i < kIters; ++i)
            launch();
        cudaEventRecord(t1, stream_);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        return ms * 1000.0f / kIters;
    }
};

float max_abs_diff(const half* d_a, const half* d_b, size_t n) {
    std::vector<half> a(n), b(n);
    cudaMemcpy(a.data(), d_a, n * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_b, n * sizeof(half), cudaMemcpyDeviceToHost);
    float m = 0.f, ref = 0.f;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(__half2float(a[i]) - __half2float(b[i])));
        ref = std::max(ref, std::fabs(__half2float(b[i])));
    }
    return m / std::max(ref, 1e-3f);
}

}  // namespace

TEST_F(GroupedTileBench, Qwen36MoeShapes) {
    struct S {
        int N, K, m_per;
        const char* name;
    };
    const S shapes[] = {{512, 2048, 16, "gate/up pp512"},
                        {2048, 512, 16, "down pp512"},
                        {512, 2048, 32, "gate/up pp1024"},
                        {2048, 512, 32, "down pp1024"}};
    V128x64 v128x64;
    V128x64ping v128x64ping;
    V128x128k256 v128k256;
    V128x128ping v128ping;
    for (const auto& s : shapes) {
        const int ne = 256;
        Experts ex;
        ex.build(ne, s.N, s.K, s.m_per, stream_);
        const size_t out_n = (size_t)ne * s.m_per * s.N;
        half* ref = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&ref), out_n * sizeof(half));

        // burn ~1 s so clocks are up (clocks idle while the experts were built)
        {
            cudaEvent_t w0, w1;
            cudaEventCreate(&w0);
            cudaEventCreate(&w1);
            cudaEventRecord(w0, stream_);
            float ms = 0.f;
            while (ms < 1000.f) {
                for (int i = 0; i < 20; ++i)
                    imp::gemm_grouped_cutlass_3x_nvfp4(ne, ex.M.data(), s.N, s.K, ex.pA.data(),
                                                       ex.pSFA.data(), ex.pB.data(), ex.pSFB.data(),
                                                       ex.pD.data(), ex.alpha.data(), stream_);
                cudaEventRecord(w1, stream_);
                cudaEventSynchronize(w1);
                cudaEventElapsedTime(&ms, w0, w1);
            }
            cudaEventDestroy(w0);
            cudaEventDestroy(w1);
        }
        float t_prod = time_us([&] {
            imp::gemm_grouped_cutlass_3x_nvfp4(ne, ex.M.data(), s.N, s.K, ex.pA.data(), ex.pSFA.data(),
                                               ex.pB.data(), ex.pSFB.data(), ex.pD.data(), ex.alpha.data(),
                                               stream_);
        });
        cudaMemcpyAsync(ref, ex.d_out, out_n * sizeof(half), cudaMemcpyDeviceToDevice, stream_);
        cudaStreamSynchronize(stream_);

        auto arm = [&](const char* tag, auto& v) {
            bool ok = v.run(ne, ex.M.data(), s.N, s.K, ex.pA.data(), ex.pSFA.data(), ex.pB.data(),
                            ex.pSFB.data(), ex.pD.data(), ex.alpha.data(), stream_);
            cudaStreamSynchronize(stream_);
            if (!ok || cudaGetLastError() != cudaSuccess) {
                printf("[grouped-tile] %-16s %-12s: DECLINED\n", s.name, tag);
                return;
            }
            float t = time_us([&] {
                v.run(ne, ex.M.data(), s.N, s.K, ex.pA.data(), ex.pSFA.data(), ex.pB.data(), ex.pSFB.data(),
                      ex.pD.data(), ex.alpha.data(), stream_);
            });
            cudaStreamSynchronize(stream_);
            float rel = max_abs_diff(ex.d_out, ref, out_n);
            printf("[grouped-tile] %-16s %-12s: %.1f us (prod 128x128x128 %.1f us, %+.1f%%)  max_rel=%.2e\n",
                   s.name, tag, t, t_prod, 100.f * (t - t_prod) / t_prod, rel);
            EXPECT_LT(rel, 2e-2f) << tag;
        };
        const double floor_us = (double)ne * s.N * s.K / 2 / 1792e9 * 1e6;
        printf("[grouped-tile] %-16s prod 128x128x128: %.1f us  (weight floor %.1f us, %.0f%%)\n", s.name,
               t_prod, floor_us, 100.0 * floor_us / t_prod);
        arm("128x64x128", v128x64);
        arm("128x64 ping", v128x64ping);
        arm("128x128x256", v128k256);
        arm("128x128 ping", v128ping);
        for (int mt : {32, 64, 128}) {
            // v2 grouped: plain layouts, per-tensor activation scale, so the
            // output differs from the CUTLASS arm by the activation quant
            // (per-expert SfAtom, tensor scale 1); the dense v2 kernel on the
            // same operands is the exact reference.
            char tag[32];
            snprintf(tag, sizeof(tag), "v2 grouped mt%d", mt);
            bool ok = imp::gemm_nvfp4_smallm_v2_grouped(ne, ex.d_w_slab, ex.w_stride, ex.d_s_slab, ex.s_stride,
                                                        ex.d_w_ts, ex.xq_plain.packed_data,
                                                        ex.xq_plain.micro_scales, ex.xq_plain.tensor_scale,
                                                        ex.d_offsets, ex.d_work, ex.work_cap, ex.d_out, s.N,
                                                        s.K, mt, stream_);
            cudaStreamSynchronize(stream_);
            if (!ok || cudaGetLastError() != cudaSuccess) {
                printf("[grouped-tile] %-16s %-16s: DECLINED\n", s.name, tag);
                continue;
            }
            float t = time_us([&] {
                imp::gemm_nvfp4_smallm_v2_grouped(ne, ex.d_w_slab, ex.w_stride, ex.d_s_slab, ex.s_stride,
                                                  ex.d_w_ts, ex.xq_plain.packed_data, ex.xq_plain.micro_scales,
                                                  ex.xq_plain.tensor_scale, ex.d_offsets, ex.d_work,
                                                  ex.work_cap, ex.d_out, s.N, s.K, mt, stream_);
            });
            cudaStreamSynchronize(stream_);
            half* y_dense = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&y_dense), out_n * sizeof(half));
            cudaMemset(y_dense, 0, out_n * sizeof(half));
            for (int e = 0; e < ne; ++e) {
                imp::NvFP4QuantResult xv = ex.xq_plain;  // view: rows of expert e
                xv.packed_data = static_cast<uint8_t*>(ex.xq_plain.packed_data) + (size_t)e * s.m_per * s.K / 2;
                xv.micro_scales = static_cast<uint8_t*>(ex.xq_plain.micro_scales) + (size_t)e * s.m_per * s.K / 16;
                ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4_tuned(ex.q[e], xv, y_dense + (size_t)e * s.m_per * s.N,
                                                               s.m_per, s.N, s.K, nullptr, stream_,
                                                               /*accumulate=*/false, /*stages=*/6,
                                                               /*stripes=*/1));
            }
            cudaStreamSynchronize(stream_);
            float rel_dense = max_abs_diff(ex.d_out, y_dense, out_n);
            cudaFree(y_dense);
            printf("[grouped-tile] %-16s %-16s: %.1f us (prod 128x128x128 %.1f us, %+.1f%%)  vs dense v2=%.2e\n",
                   s.name, tag, t, t_prod, 100.f * (t - t_prod) / t_prod, rel_dense);
            EXPECT_LT(rel_dense, 1e-3f) << tag << " differs from the dense v2 kernel on the same operands";
        }
        cudaFree(ref);
        ex.release();
    }
}
