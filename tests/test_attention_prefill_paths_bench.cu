// =============================================================================
// test_attention_prefill_paths_bench.cu — Säule 1+2 of Track E gating bench
// =============================================================================
//
// 6 model-classes × 7 seq-lens = 42 measurement points per path × 2 paths.
// Each test prints a row. Read the full matrix in CI log; aggregate is written
// by `tools/analysis/attention_prefill_bench_summary.py` after a full run.
//
// Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before invoking gtest for stable cuBLAS
// algo selection (per bench_methodology_2026_05_15).
// =============================================================================

#include "bench/attention_prefill_paths_bench.h"

#include <gtest/gtest.h>

#include <cctype>
#include <cstdio>
#include <string>

namespace {

struct ModelShape {
    const char* name;
    int n_heads;
    int n_kv_heads;
    int head_dim;
};

const ModelShape kShapes[] = {
    {"Qwen3-dense",      32,  8, 128},   // Qwen3-4B/8B/30B-A3B
    {"Llama-3.2-3B",     24,  8, 128},
    {"Gemma-4-SWA",      32, 16, 256},   // hd=256 SWA layers
    {"Gemma-4-global",    8,  8, 512},   // hd=512 global layers (FMHA may bail)
    {"Qwen3-MHA",        32, 32, 128},   // MHA stress (gqa=1)
    {"Llama-3-70B",      64,  8, 128},   // larger nh
};
constexpr int kNumShapes = sizeof(kShapes) / sizeof(kShapes[0]);

const int kSeqLens[] = {128, 256, 512, 1024, 2048, 4096, 8192};
constexpr int kNumSeqs = sizeof(kSeqLens) / sizeof(kSeqLens[0]);

void run_and_print(const ModelShape& m, int seq) {
    imp::AttnPrefillBenchResult r{};
    bool ok = imp::attention_prefill_paths_bench(seq, m.n_heads, m.n_kv_heads,
                                                  m.head_dim, &r);
    ASSERT_TRUE(ok) << "bench fixture failed";

    auto fmt_ms = [](double ms) -> std::string {
        if (ms != ms) return "    nan";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%7.3f", ms);
        return buf;
    };
    auto fmt_gflops = [](double g) -> std::string {
        if (g <= 0.0) return "    --";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%6.0f", g);
        return buf;
    };

    double speedup = 0.0;
    if (r.cublas_ms == r.cublas_ms && r.fmha_ms == r.fmha_ms && r.fmha_ms > 0) {
        speedup = r.fmha_ms / r.cublas_ms;
    }

    std::printf(
        "ATTN_PREFILL_BENCH | %-16s | seq=%5d nh=%2d nkv=%2d hd=%3d "
        "| cuBLAS %s ms (%s GFLOPS, S=%lld MiB) "
        "| FMHA %s ms (%s GFLOPS) "
        "| FMHA/cuBLAS=%6.3fx\n",
        m.name, seq, m.n_heads, m.n_kv_heads, m.head_dim,
        fmt_ms(r.cublas_ms).c_str(), fmt_gflops(r.cublas_gflops).c_str(),
        r.cublas_s_workspace_bytes / (1024LL * 1024LL),
        fmt_ms(r.fmha_ms).c_str(), fmt_gflops(r.fmha_gflops).c_str(),
        speedup);
    std::fflush(stdout);
}

struct ParamPair { int shape_idx; int seq_idx; };

class AttnPrefillBench : public ::testing::TestWithParam<ParamPair> {};

TEST_P(AttnPrefillBench, Sweep) {
    auto p = GetParam();
    run_and_print(kShapes[p.shape_idx], kSeqLens[p.seq_idx]);
}

std::vector<ParamPair> all_params() {
    std::vector<ParamPair> v;
    v.reserve(kNumShapes * kNumSeqs);
    for (int s = 0; s < kNumShapes; ++s)
        for (int q = 0; q < kNumSeqs; ++q)
            v.push_back({s, q});
    return v;
}

std::string name_for(const ::testing::TestParamInfo<ParamPair>& info) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_seq%d",
                  kShapes[info.param.shape_idx].name,
                  kSeqLens[info.param.seq_idx]);
    for (char* p = buf; *p; ++p)
        if (!std::isalnum(static_cast<unsigned char>(*p))) *p = '_';
    return buf;
}

INSTANTIATE_TEST_SUITE_P(
    Matrix, AttnPrefillBench,
    ::testing::ValuesIn(all_params()),
    name_for);

}  // namespace
