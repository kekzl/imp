// Inkling2-Loop output quality against the transformers reference (BF16, experts dequantized) over
// wikitext-2 chunks: mean KL divergence, top-1 agreement and perplexity of the scored rows, per call form
// (large calls, one-row decode steps, 8-row windows). Needs IMP_TEST_MODEL_INKLING (checkpoint directory)
// and IMP_TEST_INKLING_KLD (directory with ids.json and chunk_NN_logits.npy from
// tools/analysis/inkling_kld_oracle.py). The head here is the checkpoint's fp16 unembedding.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "compute/gemm.h"
#include "compute/inkling_ops.h"
#include "core/tensor.h"
#include "exec/inkling_forward.h"
#include "inkling_fixture.h"
#include "memory/vram_allocator.h"
#include "model/inkling_weights.h"

using namespace imp;

namespace {

// NumPy v1 fp32 C-order 2-D array.
bool read_npy2(const std::string& path, std::vector<float>& out, int64_t& rows, int64_t& cols) {
    std::ifstream f(path, std::ios::binary);
    char magic[10];
    if (!f.read(magic, 10) || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        return false;
    const uint16_t hlen = static_cast<uint16_t>(static_cast<unsigned char>(magic[8]) |
                                                (static_cast<unsigned char>(magic[9]) << 8));
    std::string hdr(hlen, ' ');
    f.read(hdr.data(), hlen);
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("False") == std::string::npos)
        return false;
    const size_t lp = hdr.find('(', hdr.find("shape")), cm = hdr.find(',', lp), rp = hdr.find(')', lp);
    if (lp == std::string::npos || cm == std::string::npos || rp == std::string::npos)
        return false;
    rows = std::stoll(hdr.substr(lp + 1, cm - lp - 1));
    cols = std::stoll(hdr.substr(cm + 1, rp - cm - 1));
    out.resize(static_cast<size_t>(rows * cols));
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    return static_cast<bool>(f);
}

// "chunks": [[...], [...]] of ids.json.
std::vector<std::vector<int32_t>> read_chunks(const std::string& path) {
    std::ifstream f(path);
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<std::vector<int32_t>> out;
    size_t at = s.find('[', s.find("\"chunks\""));
    if (at == std::string::npos)
        return out;
    for (++at; at < s.size();) {
        const size_t a = s.find_first_of("[]", at);
        if (a == std::string::npos || s[a] == ']')
            break;
        const size_t b = s.find(']', a);
        std::stringstream ss(s.substr(a + 1, b - a - 1));
        std::string tok;
        out.emplace_back();
        while (std::getline(ss, tok, ','))
            out.back().push_back(std::stoi(tok));
        at = b + 1;
    }
    return out;
}

struct Score {
    double kld = 0.0, nll = 0.0, nll_ref = 0.0;
    long rows = 0, top1 = 0, nll_rows = 0;
    double worst = 0.0;
};

// log-softmax of one row in double.
void log_softmax(const float* x, int64_t V, std::vector<double>& lp) {
    double mx = x[0];
    for (int64_t i = 1; i < V; ++i)
        mx = std::max(mx, static_cast<double>(x[i]));
    double sum = 0.0;
    for (int64_t i = 0; i < V; ++i)
        sum += std::exp(static_cast<double>(x[i]) - mx);
    const double lz = mx + std::log(sum);
    for (int64_t i = 0; i < V; ++i)
        lp[static_cast<size_t>(i)] = static_cast<double>(x[i]) - lz;
}

// Rows of got and ref ([n, V]) : KL(ref || got), argmax agreement, NLL of next[r] (-1: not scored).
void score_rows(const float* got, const float* ref, int n, int64_t V, const int32_t* next, Score& sc) {
    std::vector<double> lg(static_cast<size_t>(V)), lr(static_cast<size_t>(V));
    for (int r = 0; r < n; ++r) {
        log_softmax(got + static_cast<size_t>(r) * V, V, lg);
        log_softmax(ref + static_cast<size_t>(r) * V, V, lr);
        double kl = 0.0;
        size_t ag = 0, ar = 0;
        for (size_t i = 0; i < static_cast<size_t>(V); ++i) {
            kl += std::exp(lr[i]) * (lr[i] - lg[i]);
            if (lg[i] > lg[ag])
                ag = i;
            if (lr[i] > lr[ar])
                ar = i;
        }
        sc.kld += kl;
        sc.worst = std::max(sc.worst, kl);
        sc.top1 += ag == ar;
        ++sc.rows;
        if (next[r] >= 0) {
            sc.nll -= lg[static_cast<size_t>(next[r])];
            sc.nll_ref -= lr[static_cast<size_t>(next[r])];
            ++sc.nll_rows;
        }
    }
}

}  // namespace

// The three call forms a served request runs: its prompt in large calls, then one-row decode steps or the
// lookahead's windows of up to 8 rows. Rows keep_from.. of each chunk are scored; the rows before them
// always come from one large call.
TEST(InklingQuality, KldAgainstTheReference) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* dir = std::getenv("IMP_TEST_INKLING_KLD");
    if (!model || !dir)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_KLD not set";
    // The process's one load (inkling_fixture.h): a second 17.61 GiB copy does not fit a 32 GB card
    // beside the one the static fixture holds until exit.
    inkling_test::Fixture& fx = inkling_test::fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    VRAMAllocator& va = fx.va;
    const InklingWeights& w = fx.w;
    const InklingConfig& c = w.cfg;
    const int H = c.hidden;
    const int64_t V = c.vocab;
    std::vector<std::vector<int32_t>> chunks = read_chunks(std::string(dir) + "/ids.json");
    ASSERT_FALSE(chunks.empty());
    if (const char* lim = std::getenv("IMP_TEST_INKLING_KLD_CHUNKS"))
        chunks.resize(std::min(chunks.size(), static_cast<size_t>(std::atoi(lim))));
    const int T = static_cast<int>(chunks[0].size());
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    int32_t* d_ids = nullptr;
    float *d_f = nullptr, *d_logits = nullptr;
    half *d_h = nullptr, *d_xn = nullptr, *d_lh = nullptr;
    const char* names[3] = {"large calls", "one-row steps", "8-row windows"};
    const int step[3] = {0, 1, 8};
    Score sc[3];
    {
        // The forms a default server runs: calls of up to 8 rows read the dense projections as I8B32
        // (IMP_TEST_INKLING_FP16_DENSE=1: the fp16 ones, as a server without room for the copies).
        InklingDense8 d8;
        InklingForward fwd;
        ASSERT_TRUE(fwd.init(w, va, 512, 4096));
        if (!std::getenv("IMP_TEST_INKLING_FP16_DENSE")) {
            ASSERT_TRUE(build_inkling_dense8(w, va, d8));
            fwd.set_dense8(&d8);
        }
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            const std::vector<int32_t>& ids = chunks[ci];
            ASSERT_EQ(static_cast<int>(ids.size()), T);
            char name[64];
            std::snprintf(name, sizeof(name), "/chunk_%02zu_logits.npy", ci);
            std::vector<float> ref;
            int64_t rr = 0, rc = 0;
            ASSERT_TRUE(read_npy2(std::string(dir) + name, ref, rr, rc)) << name;
            ASSERT_EQ(rc, V);
            const int keep = static_cast<int>(rr), from = T - keep;
            ASSERT_GT(from, 0);
            if (!d_ids) {
                const size_t n = static_cast<size_t>(T) * H, nl = static_cast<size_t>(keep) * V;
                ASSERT_EQ(cudaMalloc(&d_ids, static_cast<size_t>(T) * sizeof(int32_t)), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&d_f, n * sizeof(float)), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&d_h, n * sizeof(half)), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&d_xn, static_cast<size_t>(keep) * H * sizeof(half)), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&d_lh, nl * sizeof(half)), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&d_logits, nl * sizeof(float)), cudaSuccess);
            }
            std::vector<int32_t> next(static_cast<size_t>(keep), -1);
            for (int r = 0; r + 1 < keep; ++r)
                next[static_cast<size_t>(r)] = ids[static_cast<size_t>(from) + r + 1];
            std::vector<float> got(static_cast<size_t>(keep) * V);
            for (int form = 0; form < 3; ++form) {
                cudaMemcpy(d_ids, ids.data(), static_cast<size_t>(T) * sizeof(int32_t),
                           cudaMemcpyHostToDevice);
                inkling::embed_rows(w.embed, d_ids, d_f, T, H, nullptr);
                inkling::f32_to_f16(d_f, d_h, static_cast<size_t>(T) * H, nullptr);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                for (int at = 0; at < T;) {
                    const int m = at < from         ? std::min(from - at, 256)
                                  : step[form] == 0 ? std::min(T - at, 256)
                                                    : std::min(T - at, step[form]);
                    ASSERT_TRUE(fwd.run(d_h + static_cast<size_t>(at) * H, m, at, s, ids.data() + at));
                    at += m;
                }
                ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
                // The head of InklingForward::forward over the scored rows.
                inkling::f16_to_f32(d_h + static_cast<size_t>(from) * H, d_f, static_cast<size_t>(keep) * H,
                                    nullptr);
                inkling::rms_norm(d_f, w.norm, nullptr, d_xn, keep, H, c.eps, nullptr);
                const int64_t as[2] = {keep, H}, bs[2] = {V, H}, cs[2] = {keep, V};
                const Tensor A(d_xn, QType::F16, 2, as, true);
                const Tensor B(w.unembed, QType::F16, 2, bs, true);
                Tensor C(d_lh, QType::F16, 2, cs, true);
                gemm(A, B, C, 1.0f, 0.0f, nullptr);
                inkling::softcap_f32(d_lh, d_logits, static_cast<size_t>(keep) * V, c.logit_cap, nullptr);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                cudaMemcpy(got.data(), d_logits, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
                score_rows(got.data(), ref.data(), keep, V, next.data(), sc[form]);
            }
            std::printf("chunk %zu: KLD %.5f / %.5f / %.5f (large calls / one-row steps / 8-row windows)\n",
                        ci, sc[0].kld / sc[0].rows, sc[1].kld / sc[1].rows, sc[2].kld / sc[2].rows);
            std::fflush(stdout);
        }
    }
    for (int form = 0; form < 3; ++form) {
        const Score& x = sc[form];
        const double kld = x.kld / x.rows, top1 = 100.0 * x.top1 / x.rows;
        std::printf(
            "quality %s: %zu chunks, %ld rows, mean KLD %.5f (worst row %.3f), top-1 %.2f %%, PPL %.4f "
            "(reference %.4f)\n",
            names[form], chunks.size(), x.rows, kld, x.worst, top1, std::exp(x.nll / x.nll_rows),
            std::exp(x.nll_ref / x.nll_rows));
        // 20 chunks measure 0.00229 to 0.00237 and 97.66 to 97.77 %; transformers and llama.cpp's BF16
        // model, both references, sit 0.0017 to 0.0019 apart.
        EXPECT_LT(kld, 0.0035) << names[form];
        EXPECT_GT(top1, 97.0) << names[form];
    }
    cudaStreamDestroy(s);
    for (void* p : {static_cast<void*>(d_ids), static_cast<void*>(d_f), static_cast<void*>(d_h),
                    static_cast<void*>(d_xn), static_cast<void*>(d_lh), static_cast<void*>(d_logits)})
        cudaFree(p);
}
