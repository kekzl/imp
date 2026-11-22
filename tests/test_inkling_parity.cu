// Inkling2-Loop text forward against the transformers reference: residual stream after each of the 48
// layer applications and the last-token logits. Needs IMP_TEST_MODEL_INKLING (checkpoint directory) and
// IMP_TEST_INKLING_ORACLE (directory with <name>_prompt.json, <name>_hidden.npy, <name>_logits_last.npy).

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "compute/inkling_ops.h"
#include "exec/inkling_forward.h"
#include "memory/vram_allocator.h"
#include "model/inkling_weights.h"

using namespace imp;

namespace {

// NumPy v1 fp32 C-order array; shape from the header text.
std::vector<float> read_npy(const std::string& path, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> out;
    char magic[10];
    if (!f.read(magic, 10) || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        return out;
    const uint16_t hlen = static_cast<uint16_t>(static_cast<unsigned char>(magic[8]) |
                                                (static_cast<unsigned char>(magic[9]) << 8));
    std::string hdr(hlen, ' ');
    f.read(hdr.data(), hlen);
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("False") == std::string::npos)
        return out;
    const size_t lp = hdr.find('(', hdr.find("shape")), rp = hdr.find(')', lp);
    std::stringstream ss(hdr.substr(lp + 1, rp - lp - 1));
    std::string tok;
    size_t n = 1;
    while (std::getline(ss, tok, ','))
        if (tok.find_first_of("0123456789") != std::string::npos) {
            shape.push_back(std::stoll(tok));
            n *= static_cast<size_t>(shape.back());
        }
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(float)));
    return out;
}

std::vector<int32_t> read_ids(const std::string& path) {
    std::ifstream f(path);
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int32_t> ids;
    const size_t a = s.find('[', s.find("\"ids\"")), b = s.find(']', a);
    if (a == std::string::npos || b == std::string::npos)
        return ids;
    std::stringstream ss(s.substr(a + 1, b - a - 1));
    std::string tok;
    while (std::getline(ss, tok, ','))
        ids.push_back(std::stoi(tok));
    return ids;
}

// |a - b|^2 and |b|^2, accumulated.
struct Err {
    double num = 0.0, den = 0.0;
    void add(const float* a, const float* b, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            num += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
            den += static_cast<double>(b[i]) * b[i];
        }
    }
    double rel() const { return std::sqrt(num / (den + 1e-30)); }
};

struct Fixture {
    VRAMAllocator va;  // declared first: the weights free into it
    InklingWeights w;
    bool loaded = false;
    bool load(const char* dir) { return loaded = va.init() && load_inkling_weights(dir, w, va); }
};
Fixture& fixture() {
    static Fixture f;
    return f;
}

class InklingParity : public ::testing::TestWithParam<const char*> {};

TEST_P(InklingParity, ResidualStreamAndLogitsMatchTheReference) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const std::string base = std::string(oracle) + "/" + GetParam();
    const std::vector<int32_t> ids = read_ids(base + "_prompt.json");
    std::vector<int64_t> hs, ls;
    const std::vector<float> ref = read_npy(base + "_hidden.npy", hs);
    const std::vector<float> ref_logits = read_npy(base + "_logits_last.npy", ls);
    ASSERT_FALSE(ids.empty());
    ASSERT_EQ(hs.size(), 3u);
    const int T = static_cast<int>(ids.size()), H = fx.w.cfg.hidden;
    ASSERT_EQ(hs[0], fx.w.cfg.n_apps() + 1);
    ASSERT_EQ(hs[1], T);
    ASSERT_EQ(hs[2], H);
    ASSERT_EQ(ls[0], fx.w.cfg.vocab);

    const bool split = std::getenv("IMP_TEST_INKLING_SPLIT") != nullptr;
    InklingForward fwd;
    ASSERT_TRUE(fwd.init(fx.w, fx.va, T, 4096));
    int32_t* d_ids = nullptr;
    float* d_logits = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ids, ids.size() * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_logits, static_cast<size_t>(fx.w.cfg.vocab) * sizeof(float)), cudaSuccess);
    cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    // One call, then the same prompt as a prefix call plus single-token steps: both must match.
    for (int mode = 0; mode < (split ? 2 : 1); ++mode) {
        fwd.reset(nullptr);
        std::vector<Err> errs(static_cast<size_t>(hs[0]));
        int row0 = 0;
        std::vector<float> host;
        fwd.tap = [&](int app, const float* h, int n_tok) {
            // The reference's last entry is the normed final state, not layer 23's output.
            if (app == fx.w.cfg.n_apps() - 1)
                return;
            if (app == fx.w.cfg.n_apps())
                --app;
            host.resize(static_cast<size_t>(n_tok) * H);
            cudaMemcpy(host.data(), h, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
            const float* r = ref.data() + (static_cast<size_t>(app + 1) * T + row0) * H;
            errs[static_cast<size_t>(app + 1)].add(host.data(), r, host.size());
        };
        if (mode == 0) {
            ASSERT_TRUE(fwd.forward(d_ids, T, d_logits, nullptr));
        } else {
            const int head = T / 2;
            ASSERT_TRUE(fwd.forward(d_ids, head, d_logits, nullptr));
            for (row0 = head; row0 < T; ++row0)
                ASSERT_TRUE(fwd.forward(d_ids + row0, 1, d_logits, nullptr));
        }
        double worst = 0.0;
        for (size_t a = 0; a < errs.size(); ++a) {
            std::printf("mode %d app %2zu rel_err %.5f\n", mode, a, errs[a].rel());
            worst = std::max(worst, errs[a].rel());
        }
        // The reference rounds to bf16 after every op: 2 to 6 % by the last applications.
        EXPECT_LT(worst, 0.08) << "mode " << mode;
        std::vector<float> logits(static_cast<size_t>(fx.w.cfg.vocab));
        cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
        size_t am = 0, ar = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (logits[i] > logits[am])
                am = i;
            if (ref_logits[i] > ref_logits[ar])
                ar = i;
        }
        Err le;
        le.add(logits.data(), ref_logits.data(), logits.size());
        std::printf("mode %d top-1 %zu (%.3f), reference %zu (%.3f), logits rel_err %.5f\n", mode, am,
                    logits[am], ar, ref_logits[ar], le.rel());
        EXPECT_LT(le.rel(), 0.03) << "mode " << mode;
        EXPECT_EQ(am, ar) << "mode " << mode;
    }
    cudaFree(d_ids);
    cudaFree(d_logits);
}

// Greedy continuation, one token per forward call, against the reference's generate().
TEST_P(InklingParity, GreedyContinuationMatchesTheReference) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const std::string base = std::string(oracle) + "/" + GetParam();
    std::vector<int32_t> ids = read_ids(base + "_prompt.json");
    const std::vector<int32_t> want = read_ids(base + "_greedy.json");
    ASSERT_FALSE(ids.empty());
    ASSERT_FALSE(want.empty());
    const int T = static_cast<int>(ids.size()), V = fx.w.cfg.vocab;
    InklingForward fwd;
    ASSERT_TRUE(fwd.init(fx.w, fx.va, T, 4096));
    int32_t* d_ids = nullptr;
    float* d_logits = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ids, ids.size() * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_logits, static_cast<size_t>(V) * sizeof(float)), cudaSuccess);
    cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    ASSERT_TRUE(fwd.forward(d_ids, T, d_logits, nullptr));
    std::vector<float> logits(static_cast<size_t>(V));
    std::vector<int32_t> got;
    for (size_t i = 0; i < want.size(); ++i) {
        cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
        int32_t am = 0;
        for (int v = 1; v < V; ++v)
            if (logits[static_cast<size_t>(v)] > logits[static_cast<size_t>(am)])
                am = v;
        got.push_back(am);
        cudaMemcpy(d_ids, &am, sizeof am, cudaMemcpyHostToDevice);
        ASSERT_TRUE(fwd.forward(d_ids, 1, d_logits, nullptr));
    }
    size_t same = 0;
    while (same < want.size() && got[same] == want[same])
        ++same;
    std::printf("greedy: %zu of %zu tokens equal the reference\n", same, want.size());
    EXPECT_GE(same, want.size() / 2);
    cudaFree(d_ids);
    cudaFree(d_logits);
}

INSTANTIATE_TEST_SUITE_P(Prompts, InklingParity, ::testing::Values("sky", "paris"));

// Final state of every row after run() over ids[from..) at position from.
std::vector<float> run_rows(InklingForward& fwd, const InklingWeights& w, const std::vector<int32_t>& ids,
                            int from, const std::vector<int>& cuts) {
    const int T = static_cast<int>(ids.size()) - from, H = w.cfg.hidden;
    const size_t n = static_cast<size_t>(T) * H;
    int32_t* d_ids = nullptr;
    float* d_f = nullptr;
    half* d_h = nullptr;
    std::vector<float> last(n);
    if (cudaMalloc(&d_ids, static_cast<size_t>(T) * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_f, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_h, n * sizeof(half)) != cudaSuccess)
        return {};
    cudaMemcpy(d_ids, ids.data() + from, static_cast<size_t>(T) * sizeof(int32_t), cudaMemcpyHostToDevice);
    bool ok = true;
    // One call per stretch between the cuts (absolute positions).
    for (int at = 0, m = 0; at < T && ok; at += m) {
        m = T - at;
        for (int c : cuts)
            if (c > from + at && c - from - at < m)
                m = c - from - at;
        const size_t o = static_cast<size_t>(at) * H, mn = static_cast<size_t>(m) * H;
        inkling::embed_rows(w.embed, d_ids + at, d_f + o, m, H, nullptr);
        inkling::f32_to_f16(d_f + o, d_h + o, mn, nullptr);
        ok = fwd.run(d_h + o, m, from + at, nullptr, ids.data() + from + at);
    }
    inkling::f16_to_f32(d_h, d_f, n, nullptr);
    cudaMemcpy(last.data(), d_f, last.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ids);
    cudaFree(d_f);
    cudaFree(d_h);
    return ok ? last : std::vector<float>();
}

// A prompt that shares 280 tokens with the kept sequence resumes at the checkpoint at 256 and ends in
// the state a fresh run of it reaches.
TEST(InklingRewind, ResumedPrefixMatchesAFreshRun) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const std::vector<int32_t> sky = read_ids(std::string(oracle) + "/sky_prompt.json");
    const std::vector<int32_t> paris = read_ids(std::string(oracle) + "/paris_prompt.json");
    ASSERT_FALSE(sky.empty());
    ASSERT_FALSE(paris.empty());
    std::vector<int32_t> a, b;
    for (size_t i = 0; i < 300; ++i)
        a.push_back(sky[i % sky.size()]);
    b.assign(a.begin(), a.begin() + 280);
    for (size_t i = 0; i < 30; ++i)
        b.push_back(paris[i % paris.size()]);
    InklingForward kept, fresh, kept100, fresh100;
    ASSERT_TRUE(fresh100.init(fx.w, fx.va, 512, 4096));
    ASSERT_TRUE(kept100.init(fx.w, fx.va, 512, 4096));
    // Calls of 100 rows: the checkpoint at 256 falls inside the call [200, 300).
    ASSERT_FALSE(run_rows(kept100, fx.w, a, 0, {100, 200}).empty());
    ASSERT_EQ(kept100.rewind(b, nullptr), 256);
    const std::vector<float> got100 = run_rows(kept100, fx.w, b, 256, {});
    const std::vector<float> want100 = run_rows(fresh100, fx.w, b, 0, {100, 200, 256});
    ASSERT_TRUE(kept.init(fx.w, fx.va, 512, 4096));
    ASSERT_TRUE(fresh.init(fx.w, fx.va, 512, 4096));
    ASSERT_EQ(kept.rewind(a, nullptr), 0);
    ASSERT_FALSE(run_rows(kept, fx.w, a, 0, {}).empty());
    ASSERT_EQ(kept.rewind(b, nullptr), 256);
    // Both runs compute the same segments ([0, 256), [256, 310)): equal row counts, equal GEMM picks.
    const std::vector<float> got = run_rows(kept, fx.w, b, 256, {});
    const std::vector<float> want = run_rows(fresh, fx.w, b, 0, {});
    ASSERT_FALSE(got.empty());
    ASSERT_FALSE(want.empty());
    Err e;
    ASSERT_LT(got.size(), want.size());
    const float* tail = want.data() + (want.size() - got.size());
    e.add(got.data(), tail, got.size());
    std::printf("rewind: resumed rows rel. error %.3g against a fresh run\n", e.rel());
    EXPECT_LT(e.rel(), 2e-3);
    Err e100;
    ASSERT_EQ(got100.size(), got.size());
    ASSERT_EQ(want100.size(), want.size());
    e100.add(got100.data(), want100.data() + (want.size() - got.size()), got.size());
    std::printf("rewind: %.3g with the checkpoint inside a call\n", e100.rel());
    EXPECT_LT(e100.rel(), 2e-3);
    // The same prompt again: everything but the open segment is reused.
    EXPECT_EQ(kept.rewind(b, nullptr), 256);
    ASSERT_FALSE(run_rows(kept, fx.w, b, 256, {}).empty());
    // A prompt that differs in its first token starts over.
    std::vector<int32_t> c = b;
    c[0] = paris[1];
    EXPECT_EQ(kept.rewind(c, nullptr), 0);
}

}  // namespace
