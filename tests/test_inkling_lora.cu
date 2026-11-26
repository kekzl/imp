// Inkling2-Loop LoRA (B-111 S5): an adapter on attn.{wq_du, wk_dv, wv_dv, wo_ud} gives the forward of the
// merged weights W + scale * B A; a zero delta leaves it unchanged; other modules refuse the load.
// Needs IMP_TEST_MODEL_INKLING only (tests/inkling_fixture.h): 96 prompt rows in one call, 4 decode rows.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "exec/inkling_forward.h"
#include "exec/inkling_registry.h"
#include "inkling_fixture.h"
#include "lora/lora_adapter.h"

using namespace imp;
using namespace imp::inkling_test;

namespace {

constexpr int kRank = 8, kAlpha = 16;  // scale alpha / r = 2
constexpr int kPrompt = 96, kSteps = 4;
const char* const kModule[4] = {"wq_du", "wk_dv", "wv_dv", "wo_ud"};

const char* model_dir() { return std::getenv("IMP_TEST_MODEL_INKLING"); }

std::vector<int32_t> prompt() {
    std::vector<int32_t> ids;
    for (int i = 0; i < kPrompt + kSteps; ++i)
        ids.push_back(1000 + (i * 7919) % 50000);
    return ids;
}

// Deterministic adapter values in [-amp, amp], already rounded to fp16 (what the loader uploads).
float val(int a, int b, int salt, float amp) {
    const float v = amp * static_cast<float>(((a * 31 + b * 7 + salt * 5) % 13) - 6) / 6.0f;
    return __half2float(__float2half(v));
}

struct Pair {
    int layer, proj, N, K;
    std::vector<float> A, B;  // [r, K], [N, r]
};

// PEFT dir with F32 tensors; extra: a tensor name (no .lora_A/B suffix) written as one more pair.
std::string write_adapter(const std::vector<Pair>& pairs, const char* name, const std::string& extra = {}) {
    const std::string dir = std::string(testing::TempDir()) + name;
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/adapter_config.json")
        << "{\"r\": " << kRank << ", \"lora_alpha\": " << kAlpha << ", \"use_rslora\": false}";
    std::string hdr;
    std::vector<float> data;
    auto add = [&](const std::string& key, int d0, int d1, const std::vector<float>& v) {
        const size_t start = data.size() * sizeof(float);
        data.insert(data.end(), v.begin(), v.end());
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                      hdr.empty() ? "" : ",", key.c_str(), d0, d1, start, data.size() * sizeof(float));
        hdr += buf;
    };
    for (const Pair& p : pairs) {
        const std::string k = "base_model.model.model.llm.layers." + std::to_string(p.layer) + ".attn." +
                              kModule[p.proj];
        add(k + ".lora_A.weight", kRank, p.K, p.A);
        add(k + ".lora_B.weight", p.N, kRank, p.B);
    }
    if (!extra.empty()) {
        add(extra + ".lora_A.weight", kRank, 64, std::vector<float>(kRank * 64, 0.0f));
        add(extra + ".lora_B.weight", 64, kRank, std::vector<float>(64 * kRank, 0.0f));
    }
    std::string h = "{" + hdr + "}";
    while (h.size() % 8 != 0)
        h += ' ';
    const uint64_t len = h.size();
    std::ofstream f(dir + "/adapter_model.safetensors", std::ios::binary);
    f.write(reinterpret_cast<const char*>(&len), 8);
    f.write(h.data(), static_cast<std::streamsize>(h.size()));
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    return dir;
}

// q / k / v / o of a sliding layer (one block GEMM on its first pass), a full-attention layer and a looped
// one (split passes); zero_b: the delta is exactly 0.
std::vector<Pair> make_pairs(const InklingConfig& c, bool zero_b) {
    const int H = c.hidden, QD = c.n_heads * c.head_dim, KD = c.n_kv_heads * c.head_dim;
    std::vector<Pair> out;
    for (int layer : {0, 5, 7})
        for (int p = 0; p < 4; ++p) {
            Pair x{layer, p, p == 0 ? QD : p == 3 ? H : KD, p == 3 ? QD : H, {}, {}};
            for (int j = 0; j < kRank; ++j)
                for (int k = 0; k < x.K; ++k)
                    x.A.push_back(val(j, k, layer + p, 0.05f));
            for (int n = 0; n < x.N; ++n)
                for (int j = 0; j < kRank; ++j)
                    x.B.push_back(zero_b ? 0.0f : val(n, j, 3 * layer + p, 0.05f));
            out.push_back(std::move(x));
        }
    return out;
}

half* weight_of(const InklingWeights& w, const Pair& p) {
    const InklingLayer& ly = w.layers[static_cast<size_t>(p.layer)];
    return p.proj == 0 ? ly.wq : p.proj == 1 ? ly.wk : p.proj == 2 ? ly.wv : ly.wo;
}

// W += scale * B A in place (fp32 sum, one fp16 rounding); returns the original rows.
std::vector<half> merge(const InklingWeights& w, const Pair& p, float scale) {
    const size_t n = static_cast<size_t>(p.N) * p.K;
    std::vector<half> orig(n), merged(n);
    half* d = weight_of(w, p);
    cudaMemcpy(orig.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost);
    for (int r = 0; r < p.N; ++r)
        for (int k = 0; k < p.K; ++k) {
            float acc = 0.0f;
            for (int j = 0; j < kRank; ++j)
                acc += p.B[static_cast<size_t>(r) * kRank + j] * p.A[static_cast<size_t>(j) * p.K + k];
            const size_t i = static_cast<size_t>(r) * p.K + k;
            merged[i] = __float2half(__half2float(orig[i]) + scale * acc);
        }
    cudaMemcpy(d, merged.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    return orig;
}

// Final rows of the prompt in one call, then kSteps one-row calls; l null = no adapter.
std::vector<float> run(const InklingWeights& w, VRAMAllocator& va, const InklingLora* l) {
    InklingForward f;
    if (!f.init(w, va, kPrompt, 4096))
        return {};
    f.set_lora(l);
    return run_rows(f, w, prompt(), 0, {kPrompt, kPrompt + 1, kPrompt + 2, kPrompt + 3});
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        den += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

struct Scratch {
    float* p = nullptr;
    InklingLora l;
    explicit Scratch(const LoraAdapter* a) {
        const size_t plane = static_cast<size_t>(kPrompt) * kRank;
        if (cudaMalloc(&p, inkling_lora_scratch_bytes(kPrompt, kRank)) == cudaSuccess) {
            l.a = a;
            for (int i = 0; i < 3; ++i)
                l.t[i] = p + plane * i;
        }
    }
    ~Scratch() { cudaFree(p); }
};

}  // namespace

TEST(InklingLora, ZeroDeltaLeavesTheForwardUnchanged) {
    if (!model_dir())
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model_dir()));
    LoraAdapter a;
    a.set_targets(kInklingLoraTargets);
    ASSERT_TRUE(a.load(write_adapter(make_pairs(fx.w.cfg, true), "inkling_lora_zero"), fx.w.cfg.n_layers));
    Scratch sc(&a);
    ASSERT_NE(sc.p, nullptr);
    const std::vector<float> base = run(fx.w, fx.va, nullptr);
    const std::vector<float> none = run(fx.w, fx.va, &sc.l);  // 12 pairs, every delta 0
    InklingLora off;                                          // a wired runtime with no adapter active
    const std::vector<float> idle = run(fx.w, fx.va, &off);
    ASSERT_EQ(base.size(), static_cast<size_t>(kPrompt + kSteps) * fx.w.cfg.hidden);
    ASSERT_EQ(none.size(), base.size());
    ASSERT_EQ(idle.size(), base.size());
    size_t diff = 0, diff_idle = 0;
    for (size_t i = 0; i < base.size(); ++i) {
        diff += none[i] != base[i] ? 1 : 0;  // == : -0 and +0 are the same value
        diff_idle += idle[i] != base[i] ? 1 : 0;
    }
    std::printf("inkling lora: zero delta %zu of %zu values differ, no adapter %zu\n", diff, base.size(),
                diff_idle);
    EXPECT_EQ(diff, 0u);
    EXPECT_EQ(diff_idle, 0u);
}

TEST(InklingLora, DeltaMatchesTheMergedWeights) {
    if (!model_dir())
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model_dir()));
    const std::vector<Pair> pairs = make_pairs(fx.w.cfg, false);
    LoraAdapter a;
    a.set_targets(kInklingLoraTargets);
    ASSERT_TRUE(a.load(write_adapter(pairs, "inkling_lora_ramp"), fx.w.cfg.n_layers));
    std::string why;
    ASSERT_TRUE(a.check_dims(inkling_lora_dims(fx.w.cfg), &why)) << why;
    ASSERT_FLOAT_EQ(a.scale(), 2.0f);
    Scratch sc(&a);
    ASSERT_NE(sc.p, nullptr);
    const std::vector<float> base = run(fx.w, fx.va, nullptr);
    const std::vector<float> lora = run(fx.w, fx.va, &sc.l);
    std::vector<std::vector<half>> orig;
    for (const Pair& p : pairs)
        orig.push_back(merge(fx.w, p, a.scale()));
    const std::vector<float> merged = run(fx.w, fx.va, nullptr);
    for (size_t i = 0; i < pairs.size(); ++i)  // the fixture's weights serve the other tests
        cudaMemcpy(weight_of(fx.w, pairs[i]), orig[i].data(), orig[i].size() * sizeof(half),
                   cudaMemcpyHostToDevice);
    ASSERT_EQ(lora.size(), base.size());
    ASSERT_EQ(merged.size(), base.size());
    const double moved = rel(lora, base), off = rel(lora, merged);
    std::printf("inkling lora: rel. distance to the base %.4g, to the merged weights %.4g\n", moved, off);
    EXPECT_GT(moved, 1e-2) << "the adapter must move the forward";
    EXPECT_LT(off, 0.25 * moved) << "the deltas must give the merged weights' forward";
    EXPECT_EQ(run(fx.w, fx.va, nullptr), base) << "restored weights give the base forward";
}

TEST(InklingLora, AnAdapterOnOtherModulesRefusesTheLoad) {
    if (!model_dir())
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model_dir()));
    std::vector<Pair> q = make_pairs(fx.w.cfg, true);
    q.resize(1);
    const std::string dir = write_adapter(q, "inkling_lora_experts",
                                          "base_model.model.model.llm.layers.0.mlp.experts.w13_weight");
    LoraAdapter strict, lax;
    strict.set_targets(kInklingLoraTargets);
    EXPECT_FALSE(strict.load(dir, fx.w.cfg.n_layers));
    // Every other forward skips the experts pair and keeps the q pair (#522 v1).
    ASSERT_TRUE(lax.load(dir, fx.w.cfg.n_layers));
    EXPECT_NE(lax.get(0, LoraProj::Q), nullptr);
}
