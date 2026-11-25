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
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "compute/inkling_ops.h"
#include "exec/inkling_forward.h"
#include "inkling_fixture.h"
#include "memory/vram_allocator.h"
#include "model/inkling_weights.h"
#include "vision/inkling_vision.h"

using namespace imp;
using namespace imp::inkling_test;

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
    InklingForward kept, fresh, kept100;
    ASSERT_TRUE(kept100.init(fx.w, fx.va, 512, 4096));
    // One call of 310 rows: the checkpoint at 256 is taken inside it. Resuming there must give the
    // rows the call itself computed (a call's rows move by percents with the shapes of the calls before
    // it: a router near-tie flips an expert, so only equal call shapes compare).
    const std::vector<float> want100 = run_rows(kept100, fx.w, b, 0, {});
    ASSERT_EQ(kept100.rewind(b, nullptr), 256);
    const std::vector<float> got100 = run_rows(kept100, fx.w, b, 256, {});
    ASSERT_TRUE(kept.init(fx.w, fx.va, 512, 4096));
    ASSERT_TRUE(fresh.init(fx.w, fx.va, 512, 4096));
    ASSERT_EQ(kept.rewind(a, nullptr), 0);
    ASSERT_FALSE(run_rows(kept, fx.w, a, 0, {256}).empty());
    ASSERT_EQ(kept.rewind(b, nullptr), 256);
    // Both runs compute the same calls ([0, 256), [256, 310)): equal row counts, equal GEMM picks.
    const std::vector<float> got = run_rows(kept, fx.w, b, 256, {});
    const std::vector<float> want = run_rows(fresh, fx.w, b, 0, {256});
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

// Slot choice: a bound slot is never taken, the longest shared prefix wins; on a tie a slot without a
// checkpoint goes before one with, then the oldest binding.
TEST(InklingRewind, ARequestBindsToTheFreeSlotWithTheLongestPrefix) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const std::vector<int32_t> sky = read_ids(std::string(oracle) + "/sky_prompt.json");
    ASSERT_FALSE(sky.empty());
    std::vector<int32_t> a, other;
    for (size_t i = 0; i < 300; ++i)
        a.push_back(sky[i % sky.size()]);
    other.assign(a.rbegin(), a.rend());
    {
        InklingRuntime rt;
        for (int i = 0; i < 3; ++i) {
            rt.slots.push_back(std::make_unique<InklingForward>());
            ASSERT_TRUE(rt.slots.back()->init(fx.w, fx.va, 512, 4096));
        }
        rt.table = InklingSlotTable(3);
        // Three empty slots: bound in order.
        ASSERT_EQ(rt.bind(10, a), 0);
        ASSERT_EQ(rt.slots[0]->rewind(a, nullptr), 0);
        ASSERT_FALSE(run_rows(*rt.slots[0], fx.w, a, 0, {}).empty());
        EXPECT_EQ(rt.bind(11, other), 1);  // slot 0 is bound to 10
        EXPECT_EQ(rt.table.find(11), 1);
        // 10 and 11 end. The kept prompt again: slot 0 shares 256 positions, slots 1 and 2 none.
        rt.table.release(10);
        rt.table.release(11);
        EXPECT_EQ(rt.bind(12, a), 0);
        EXPECT_EQ(rt.table.find(10), -1);
        // Slot 0 bound: the never-bound slot 2 is older than slot 1.
        EXPECT_EQ(rt.bind(13, a), 2);
        // 12 and 13 end. No match anywhere: slot 1 holds the oldest binding now.
        rt.table.release(12);
        rt.table.release(13);
        EXPECT_EQ(rt.bind(14, other), 1);
        // 14 ends. Slot 0 holds the oldest binding but also the only checkpoint, so slot 2 goes.
        rt.table.release(14);
        EXPECT_EQ(rt.bind(16, other), 2);
        EXPECT_EQ(rt.bind(17, a), 0);
        EXPECT_EQ(rt.bind(18, a), 1);
        EXPECT_EQ(rt.bind(15, a), -1);  // every slot bound
    }
}

// Vision tower against the transformers reference (fp32): merged features of two synthetic images,
// one of them not a multiple of the 40-pixel token cell.
TEST(InklingVisionParity, MergedFeaturesMatchTheReference) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    {
        InklingVision vis;
        ASSERT_TRUE(vis.load(model, fx.va));
        for (const char* name : {"a", "b"}) {
            const std::string base = std::string(oracle) + "/vis_" + name;
            std::vector<int64_t> rs, fs;
            const std::vector<float> rgbf = read_npy(base + "_rgb.npy", rs);
            const std::vector<float> want = read_npy(base + "_features.npy", fs);
            ASSERT_EQ(rs.size(), 3u);
            ASSERT_EQ(fs.size(), 2u);
            const int h = static_cast<int>(rs[0]), w = static_cast<int>(rs[1]);
            int rows = 0, cols = 0;
            vis.grid(h, w, rows, cols);
            ASSERT_EQ(static_cast<int64_t>(rows) * cols, fs[0]);
            ASSERT_EQ(fs[1], vis.config().out);
            std::vector<uint8_t> rgb(rgbf.size());
            for (size_t i = 0; i < rgb.size(); ++i)
                rgb[i] = static_cast<uint8_t>(rgbf[i]);
            half* d_out = nullptr;
            float* d_f = nullptr;
            ASSERT_EQ(cudaMalloc(&d_out, want.size() * sizeof(half)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_f, want.size() * sizeof(float)), cudaSuccess);
            ASSERT_TRUE(vis.encode(rgb.data(), h, w, d_out, nullptr));
            inkling::f16_to_f32(d_out, d_f, want.size(), nullptr);
            std::vector<float> got(want.size());
            cudaMemcpy(got.data(), d_f, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
            Err e;
            e.add(got.data(), want.data(), want.size());
            std::printf("vision %s: %d x %d px, %d x %d tokens, features rel. error %.4f\n", name, w, h, cols,
                        rows, e.rel());
            EXPECT_LT(e.rel(), 0.01);
            cudaFree(d_out);
            cudaFree(d_f);
        }
    }
}

// Lookahead forward whose drafts come from a plan: position of the anchor -> tokens proposed after it.
InklingSpecHooks planned(const std::map<int, std::vector<int32_t>>& plan) {
    InklingSpecHooks h;
    h.on_rows = [](const float*, int, int, int, cudaStream_t) {};
    h.propose = [&plan](int32_t, const half*, int pos, int32_t* out, cudaStream_t) {
        const auto it = plan.find(pos);
        if (it == plan.end())
            return 0;
        std::copy(it->second.begin(), it->second.end(), out);
        return static_cast<int>(it->second.size());
    };
    return h;
}

// Drafts that turn out wrong after two rows leave the forward where a call of only the right rows
// leaves it: same states for every later token.
TEST(InklingLookahead, RejectedRowsLeaveNoTrace) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    std::vector<int32_t> t = read_ids(std::string(oracle) + "/paris_prompt.json");
    const std::vector<int32_t> g = read_ids(std::string(oracle) + "/paris_greedy.json");
    ASSERT_FALSE(t.empty());
    const int P = static_cast<int>(t.size());
    t.insert(t.end(), g.begin(), g.end());
    const std::vector<int32_t> prompt(t.begin(), t.begin() + P);
    const auto next = [&](int pos, int n) {
        return std::vector<int32_t>(t.begin() + pos + 1, t.begin() + pos + 1 + n);
    };
    // A: at P the drafts are right for two rows, then wrong. B: only the two right rows are drafted.
    std::map<int, std::vector<int32_t>> plan_a, plan_b;
    plan_a[P] = next(P, 7);
    for (size_t i = 2; i < 7; ++i)
        plan_a[P][i] = t[0];
    plan_b[P] = next(P, 2);
    for (int pos = P + 3; pos + 8 < static_cast<int>(t.size()); pos += 8)
        plan_a[pos] = plan_b[pos] = next(pos, 7);
    const std::vector<int> taps = {5, 23, 41, 48};
    {
        Stepper st(fx.w);
        InklingForward a, b, plain;
        ASSERT_TRUE(a.init(fx.w, fx.va, 512, 4096));
        ASSERT_TRUE(b.init(fx.w, fx.va, 512, 4096));
        ASSERT_TRUE(plain.init(fx.w, fx.va, 512, 4096));
        ASSERT_TRUE(a.enable_lookahead(planned(plan_a), taps, fx.w.embed));
        ASSERT_TRUE(b.enable_lookahead(planned(plan_b), taps, fx.w.embed));
        for (InklingForward* f : {&a, &b, &plain})
            ASSERT_FALSE(run_rows(*f, fx.w, prompt, 0, {}).empty());
        Err ab, ap;
        for (int pos = P; pos < static_cast<int>(t.size()); ++pos) {
            const std::vector<float> ra = st.step(a, t[static_cast<size_t>(pos)], pos);
            const std::vector<float> rb = st.step(b, t[static_cast<size_t>(pos)], pos);
            const std::vector<float> rp = st.step(plain, t[static_cast<size_t>(pos)], pos);
            ASSERT_FALSE(ra.empty());
            ASSERT_FALSE(rb.empty());
            ASSERT_FALSE(rp.empty());
            ab.add(ra.data(), rb.data(), ra.size());
            ap.add(ra.data(), rp.data(), ra.size());
        }
        const InklingForward::SpecStats sa = a.spec_stats(), sb = b.spec_stats();
        std::printf(
            "lookahead: rejected-rows run vs right-rows run rel. error %.3g, vs one-row decode %.3g; "
            "A %ld steps %ld drafted %ld confirmed, B %ld / %ld / %ld\n",
            ab.rel(), ap.rel(), sa.steps, sa.drafted, sa.accepted, sb.steps, sb.drafted, sb.accepted);
        EXPECT_LT(ab.rel(), 2e-3);
        // Rows of an eight-row call against one-row calls: other GEMM row counts, the forward's own
        // shape noise (the transformers reference sits 4 % from both).
        EXPECT_LT(ap.rel(), 0.15);
        EXPECT_EQ(sa.accepted, sb.accepted);
        EXPECT_GT(sa.accepted, static_cast<long>(g.size()) / 2);
        EXPECT_EQ(sa.drafted - sa.accepted, 5);
    }
}

// A checkpoint that falls inside a lookahead call holds the conv state of its row, not of the call's
// last row: a prompt resumed there ends where it ends when the call stopped at that row.
TEST(InklingLookahead, CheckpointInsideACallIsTheRowsState) {
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
    std::vector<int32_t> t;
    for (size_t i = 0; i < 268; ++i)
        t.push_back(sky[i % sky.size()]);
    const int P = 249;  // anchor of a call that covers 249..256: position 256 is its last row
    const std::vector<int32_t> prompt(t.begin(), t.begin() + P);
    // look: seven drafts, position 255 is row 6 of 8. edge: six drafts, 255 is the call's last row.
    std::map<int, std::vector<int32_t>> plan, edge;
    plan[P] = std::vector<int32_t>(t.begin() + P + 1, t.begin() + P + 8);
    edge[P] = std::vector<int32_t>(t.begin() + P + 1, t.begin() + P + 7);
    std::vector<int32_t> b(t.begin(), t.begin() + 262);
    for (size_t i = 0; i < 20; ++i)
        b.push_back(paris[i % paris.size()]);
    const std::vector<int> taps = {5, 23, 41, 48};
    {
        Stepper st(fx.w);
        InklingForward look, last;
        std::vector<float> got[2];
        int k = 0;
        for (InklingForward* f : {&look, &last}) {
            ASSERT_TRUE(f->init(fx.w, fx.va, 512, 4096));
            ASSERT_TRUE(f->enable_lookahead(planned(f == &look ? plan : edge), taps, fx.w.embed));
            ASSERT_FALSE(run_rows(*f, fx.w, prompt, 0, {}).empty());
            for (int pos = P; pos < 262; ++pos)
                ASSERT_FALSE(st.step(*f, t[static_cast<size_t>(pos)], pos).empty());
            ASSERT_EQ(f->rewind(b, nullptr), 256);
            got[k++] = run_rows(*f, fx.w, b, 256, {});
            ASSERT_FALSE(got[k - 1].empty());
        }
        EXPECT_EQ(look.spec_stats().accepted, 7);
        Err e;
        e.add(got[0].data(), got[1].data(), got[0].size());
        std::printf(
            "lookahead: rows resumed at a checkpoint inside a call, rel. error %.3g against the "
            "checkpoint at a call's end\n",
            e.rel());
        EXPECT_LT(e.rel(), 2e-3);
        EXPECT_EQ(last.spec_stats().accepted, 6);
    }
}

// Conv-state checkpoints further back than a resume can reach are freed (3.8 MB each, one per 256
// positions): after 9000 positions the forward holds the last 24, a prompt that diverges 2000 positions
// before the end still resumes at its checkpoint with the rows of a fresh run (calls of 256 rows in
// both, so the shapes match and the rows are bit-equal), one that diverges 6000 before the end starts
// over as before.
TEST(InklingRewind, CheckpointsBehindTheResumeWindowAreFreed) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const std::vector<int32_t> p = read_ids(std::string(oracle) + "/sky_prompt.json");
    const std::vector<int32_t> q = read_ids(std::string(oracle) + "/paris_prompt.json");
    ASSERT_FALSE(p.empty());
    ASSERT_FALSE(q.empty());
    const int N = 9000;
    std::vector<int32_t> a;
    for (int i = 0; i < N; ++i)
        a.push_back(p[static_cast<size_t>(i) % p.size()]);
    std::vector<int> cuts;
    for (int c = 256; c < N; c += 256)
        cuts.push_back(c);
    InklingForward kept, fresh;
    ASSERT_TRUE(kept.init(fx.w, fx.va, 512, 16384));
    ASSERT_TRUE(fresh.init(fx.w, fx.va, 512, 16384));
    ASSERT_FALSE(run_rows(kept, fx.w, a, 0, cuts).empty());
    // 35 checkpoints taken; resumable are those within 5632 positions of 9000, kept with 512 of margin.
    const size_t one = kept.checkpoint_bytes() / 24;
    EXPECT_EQ(kept.checkpoint_bytes(), 24 * one);
    EXPECT_GT(one, size_t{3} << 20);
    EXPECT_LT(one, size_t{5} << 20);
    std::printf("checkpoints after %d positions: %.1f MiB (35 taken)\n", N,
                static_cast<double>(kept.checkpoint_bytes()) / 1048576.0);
    // Diverges at 7000: resumes at checkpoint 27 (6912).
    std::vector<int32_t> b(a.begin(), a.begin() + 7000);
    for (int i = 0; i < 200; ++i)
        b.push_back(q[static_cast<size_t>(i) % q.size()]);
    std::vector<int32_t> far(a.begin(), a.begin() + 3000);
    far.push_back(q[0]);
    far.push_back(q[1]);
    EXPECT_EQ(kept.match(far), 0);
    ASSERT_EQ(kept.rewind(b, nullptr), 6912);
    const std::vector<float> got = run_rows(kept, fx.w, b, 6912, {});
    std::vector<int> cuts_b;
    for (int c = 256; c <= 6912; c += 256)
        cuts_b.push_back(c);
    const std::vector<float> want = run_rows(fresh, fx.w, b, 0, cuts_b);
    ASSERT_FALSE(got.empty());
    ASSERT_GT(want.size(), got.size());
    EXPECT_EQ(std::memcmp(got.data(), want.data() + (want.size() - got.size()), got.size() * sizeof(float)),
              0);
}

// Full-attention K/V in pool blocks behind a SHUFFLED table (B-111 S2b) against the forward's own rows
// behind the identity table: a 2061-token prompt in calls of 512, then lookahead steps anchored at
// positions 13 mod 16 whose eight rows cross a block boundary, three drafts confirmed and four rejected
// (their rows rewritten in the next block). Same kernels, other addresses: bit-equal rows.
TEST(InklingPagedKv, ShuffledPoolBlocksAreBitEqual) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const std::vector<int32_t> p = read_ids(std::string(oracle) + "/sky_prompt.json");
    ASSERT_FALSE(p.empty());
    const int P = 2061, N = P + 160, ctx = 4096;
    ASSERT_EQ(P % 16, 13);
    std::vector<int32_t> t;
    while (static_cast<int>(t.size()) < N + 8)
        t.insert(t.end(), p.begin(), p.end());
    std::map<int, std::vector<int32_t>> plan;
    for (int a = P; a + 8 < N; a += 16) {
        std::vector<int32_t> d(t.begin() + a + 1, t.begin() + a + 8);
        d[3] = t[0] == d[3] ? t[1] : t[0];  // rows a+1..a+3 confirmed (a+3 opens a block), a+4.. rejected
        plan[a] = d;
    }
    std::vector<int> cuts;
    for (int c = 512; c < P; c += 512)
        cuts.push_back(c);
    const std::vector<int> taps = {5, 23, 41, 48};
    const InklingKvLayout l = inkling_kv_apps(fx.w.cfg, false);
    const size_t KD = static_cast<size_t>(fx.w.cfg.n_kv_heads) * fx.w.cfg.head_dim;
    std::vector<float> rows[2], steps[2];
    long accepted[2] = {};
    std::vector<void*> owned;
    for (int g = 0; g < 2; ++g) {
        InklingForward f;
        ASSERT_TRUE(f.init(fx.w, fx.va, 512, ctx, nullptr, /*own_kv=*/g == 0));
        if (g == 1) {
            // Pool of table_cap + 7 blocks per layer handed out shuffled (37 prime to the pool: a
            // permutation).
            const int nb = f.table_cap(), pool = nb + 7;
            ASSERT_EQ(std::gcd(37, pool), 1);
            std::vector<int> tbl(static_cast<size_t>(nb));
            for (int i = 0; i < nb; ++i)
                tbl[static_cast<size_t>(i)] = (i * 37 + 5) % pool;
            std::vector<half*> k, v;
            for (int i = 0; i < 2 * l.n_global; ++i) {
                void* d = nullptr;
                ASSERT_EQ(cudaMalloc(&d, static_cast<size_t>(pool) * 16 * KD * sizeof(half)), cudaSuccess);
                ASSERT_EQ(cudaMemset(d, 0x7b, static_cast<size_t>(pool) * 16 * KD * sizeof(half)),
                          cudaSuccess);
                owned.push_back(d);
                (i % 2 ? v : k).push_back(static_cast<half*>(d));
            }
            int* d_tbl = nullptr;
            ASSERT_EQ(cudaMalloc(&d_tbl, tbl.size() * sizeof(int)), cudaSuccess);
            owned.push_back(d_tbl);
            cudaMemcpy(d_tbl, tbl.data(), tbl.size() * sizeof(int), cudaMemcpyHostToDevice);
            ASSERT_TRUE(f.bind_kv(k, v));
            f.set_table(d_tbl, nb, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
        ASSERT_TRUE(f.enable_lookahead(planned(plan), taps, fx.w.embed));
        rows[g] = run_rows(f, fx.w, std::vector<int32_t>(t.begin(), t.begin() + P), 0, cuts);
        ASSERT_FALSE(rows[g].empty());
        Stepper st(fx.w);
        for (int pos = P; pos < N; ++pos) {
            const std::vector<float> r = st.step(f, t[static_cast<size_t>(pos)], pos);
            ASSERT_FALSE(r.empty());
            steps[g].insert(steps[g].end(), r.begin(), r.end());
        }
        accepted[g] = f.spec_stats().accepted;
    }
    for (void* d : owned)
        cudaFree(d);
    EXPECT_EQ(accepted[0], accepted[1]);
    EXPECT_GE(accepted[0], 9);  // three anchors at least (the draft policy rests between probes)
    EXPECT_EQ(std::memcmp(rows[0].data(), rows[1].data(), rows[0].size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(steps[0].data(), steps[1].data(), steps[0].size() * sizeof(float)), 0);
}

// A real stream turns the side streams on (k / v chains, message mix): rows of every call size must
// come out as on the default stream, where everything runs in order. Three cuBLAS projections at
// once shared a workspace and passed every other test.
TEST(InklingRewind, SideStreamsDoNotChangeTheRows) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    std::vector<int32_t> t = read_ids(std::string(oracle) + "/paris_prompt.json");
    const std::vector<int32_t> g = read_ids(std::string(oracle) + "/paris_greedy.json");
    ASSERT_FALSE(t.empty());
    t.insert(t.end(), g.begin(), g.end());
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    // Calls of 21 (cuBLAS), 8, 3 and 1 rows.
    const std::vector<int> cuts = {21, 29, 32, 33, 41, 44, 45, 53, 56, 57, 65, 68};
    {
        InklingForward plain, streamed;
        ASSERT_TRUE(plain.init(fx.w, fx.va, 512, 4096));
        ASSERT_TRUE(streamed.init(fx.w, fx.va, 512, 4096));
        const std::vector<float> want = run_rows(plain, fx.w, t, 0, cuts);
        const std::vector<float> got = run_rows(streamed, fx.w, t, 0, cuts, s);
        ASSERT_FALSE(want.empty());
        ASSERT_FALSE(got.empty());
        Err e;
        e.add(got.data(), want.data(), want.size());
        std::printf("side streams: rows rel. error %.3g against the default stream\n", e.rel());
        // Graph replay against eager calls of the same kernels: equal up to the GEMM picks.
        EXPECT_LT(e.rel(), 2e-3);
    }
    cudaStreamDestroy(s);
}

// One-row calls run two conv sites per layer application inside the block mix: the rows must be the
// bits of the same calls with every site launched on its own.
TEST(InklingRewind, FusedConvSitesAreBitEqual) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    std::vector<int32_t> t = read_ids(std::string(oracle) + "/paris_prompt.json");
    const std::vector<int32_t> g = read_ids(std::string(oracle) + "/paris_greedy.json");
    ASSERT_FALSE(t.empty());
    t.insert(t.end(), g.begin(), g.end());
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    // One call for the prompt's first rows, then one row per call.
    const int head = static_cast<int>(t.size()) - 40;
    std::vector<int> cuts;
    for (int c = head; c < static_cast<int>(t.size()); ++c)
        cuts.push_back(c);
    {
        InklingForward split, fused;
        ASSERT_TRUE(split.init(fx.w, fx.va, 512, 4096));
        ASSERT_TRUE(fused.init(fx.w, fx.va, 512, 4096));
        split.set_fused_sites(false);
        const std::vector<float> want = run_rows(split, fx.w, t, 0, cuts, s);
        const std::vector<float> got = run_rows(fused, fx.w, t, 0, cuts, s);
        ASSERT_FALSE(want.empty());
        ASSERT_EQ(got.size(), want.size());
        size_t differ = 0;
        for (size_t i = 0; i < want.size(); ++i)
            differ += std::memcmp(&got[i], &want[i], sizeof(float)) != 0;
        std::printf("fused conv sites: %zu of %zu values differ over 40 one-row calls\n", differ,
                    want.size());
        EXPECT_EQ(differ, 0u);
    }
    cudaStreamDestroy(s);
}

// Calls of up to 8 rows run the mix and gate kernels of inkling_fixed.cu (counts fixed at compile time,
// loads issued up front). Same rows through the runtime-bounded kernels: 20 one-row and 5 four-row calls.
TEST(InklingRewind, FixedShapeKernelsAreBitEqual) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    std::vector<int32_t> t = read_ids(std::string(oracle) + "/paris_prompt.json");
    const std::vector<int32_t> g = read_ids(std::string(oracle) + "/paris_greedy.json");
    ASSERT_FALSE(t.empty());
    t.insert(t.end(), g.begin(), g.end());
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    const int n = static_cast<int>(t.size()), head = n - 40;
    std::vector<int> cuts;
    for (int c = head; c < head + 20; ++c)
        cuts.push_back(c);
    for (int c = head + 20; c < n; c += 4)
        cuts.push_back(c);
    {
        InklingForward loose, fixed;
        ASSERT_TRUE(loose.init(fx.w, fx.va, 512, 4096));
        ASSERT_TRUE(fixed.init(fx.w, fx.va, 512, 4096));
        inkling::set_fixed_shape_kernels(false);
        const std::vector<float> want = run_rows(loose, fx.w, t, 0, cuts, s);
        inkling::set_fixed_shape_kernels(true);
        const std::vector<float> got = run_rows(fixed, fx.w, t, 0, cuts, s);
        ASSERT_FALSE(want.empty());
        ASSERT_EQ(got.size(), want.size());
        size_t differ = 0;
        for (size_t i = 0; i < want.size(); ++i)
            differ += std::memcmp(&got[i], &want[i], sizeof(float)) != 0;
        std::printf("fixed-shape kernels: %zu of %zu values differ over 20 one-row and 5 four-row calls\n",
                    differ, want.size());
        EXPECT_EQ(differ, 0u);
    }
    cudaStreamDestroy(s);
}

}  // namespace

// Calls of 1..8 rows share the key loads between the rows and the heads of a KV head (attn_part_rows_k;
// two heads per KV head take attn_part_k). Same inputs through the row form (one block per row and head,
// no part buffer): a sliding layer past its ring's wrap, a full-attention layer with the position scale
// on, and a sequence's first rows.
// Calls of many rows sort the expert slots and run grouped GEMMs (NVFP4 and MXFP8 layers) and compute
// the attention as GEMMs: the rows against the same call with one GEMV per slot / the row kernel.
TEST(InklingPrefill, GroupedExpertsMatchTheSlotForm) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    std::vector<int32_t> t = read_ids(std::string(oracle) + "/paris_prompt.json");
    const std::vector<int32_t> g = read_ids(std::string(oracle) + "/paris_greedy.json");
    ASSERT_FALSE(t.empty());
    t.insert(t.end(), g.begin(), g.end());
    InklingForward slot, grouped, row_attn;
    ASSERT_TRUE(row_attn.init(fx.w, fx.va, 512, 4096));
    row_attn.set_gemm_attention(false);
    ASSERT_TRUE(slot.init(fx.w, fx.va, 512, 4096));
    ASSERT_TRUE(grouped.init(fx.w, fx.va, 512, 4096));
    slot.set_grouped_experts(false);
    // The stream after the first application (an NVFP4 layer) and after the last (an MXFP8 one).
    const size_t n0 = t.size() * static_cast<size_t>(fx.w.cfg.hidden);
    std::vector<float> first[3];
    int i = 0;
    for (InklingForward* f : {&slot, &grouped, &row_attn}) {
        std::vector<float>& dst = first[i++];
        dst.resize(n0);
        f->tap = [&dst, n0](int app, const float* h, int) {
            if (app == 0)
                cudaMemcpy(dst.data(), h, n0 * sizeof(float), cudaMemcpyDeviceToHost);
        };
    }
    const std::vector<float> want = run_rows(slot, fx.w, t, 0, {});
    const std::vector<float> got = run_rows(grouped, fx.w, t, 0, {});
    ASSERT_FALSE(run_rows(row_attn, fx.w, t, 0, {}).empty());
    Err ea;
    ea.add(first[1].data(), first[2].data(), n0);
    std::printf("GEMM attention: first application rel. error %.3g against the row kernel\n", ea.rel());
    EXPECT_GT(ea.rel(), 0.0);
    EXPECT_LT(ea.rel(), 1e-3);
    Err e0;
    e0.add(first[1].data(), first[0].data(), n0);
    std::printf("grouped experts: first application rel. error %.3g\n", e0.rel());
    EXPECT_GT(e0.rel(), 0.0);  // another kernel ran
    EXPECT_LT(e0.rel(), 1e-3);
    ASSERT_FALSE(want.empty());
    ASSERT_EQ(got.size(), want.size());
    Err e;
    e.add(got.data(), want.data(), want.size());
    std::printf("grouped experts: %zu rows, rel. error %.3g against one GEMV per slot\n", t.size(), e.rel());
    // 48 applications amplify the first one's difference (both runs sit about 4 % from the reference).
    EXPECT_LT(e.rel(), 0.15);
}
