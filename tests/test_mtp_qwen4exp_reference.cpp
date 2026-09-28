// Qwen4Exp (Qwen3.8-Flash-Next) MTP draft step against the numpy reference of the vLLM math
// (tools/analysis/mtp_qwen4exp_reference.py -> tests/fixtures/mtp_qwen4exp_ref.txt): two chained steps,
// positions 0 and 1, from a fixed hc-stream input. Loads only the head, embed rows and lm_head.
//
// Band per logit: 4 x the reference's own FP16-storage noise (its FP16 arm vs FP64) + 2^-7, the
// FP16 ulp of a |logit| < 16. imp stores the same intermediates in FP16 but accumulates in a
// different order; 4x covers that. Argmax must match (reference top1-top2 >= 0.40 on both steps).
#include "compute/mtp_forward.h"
#include "model/model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kD = 2560, kHc = 4, kVocab = 248320, kExperts = 512, kTopK = 10, kEff = 640;
constexpr int kHeads = 24, kKvHeads = 2, kHeadDim = 256, kLowrank = 320;
constexpr int kTokens[2] = {9707, 1234};

struct Mapped {
    const uint8_t* base = nullptr;
    size_t size = 0;
    size_t data_off = 0;
    nlohmann::json hdr;
};

class Checkpoint {
public:
    explicit Checkpoint(const std::string& dir) : dir_(dir) {
        std::ifstream f(dir + "/model.safetensors.index.json");
        where_ = nlohmann::json::parse(f)["weight_map"];
    }
    ~Checkpoint() {
        for (auto& [k, m] : files_)
            munmap(const_cast<uint8_t*>(m.base), m.size);
    }
    // Host view of one tensor, dtype mapped the way the loader maps it.
    bool tensor(const std::string& name, imp::Tensor& out) {
        if (!where_.contains(name))
            return false;
        Mapped& m = file(where_[name].get<std::string>());
        const auto& t = m.hdr[name];
        const std::string dt = t["dtype"];
        const imp::QType q = dt == "BF16" ? imp::QType::BF16 : dt == "F8_E4M3" ? imp::QType::FP8_E4M3 : imp::QType::F32;
        std::vector<int64_t> shape = t["shape"].get<std::vector<int64_t>>();
        const size_t off = t["data_offsets"][0].get<size_t>();
        out = imp::Tensor(const_cast<uint8_t*>(m.base + m.data_off + off), q, static_cast<int>(shape.size()),
                          shape.data(), /*on_device=*/false);
        return true;
    }
    std::vector<std::string> names_with_prefix(const std::string& p) const {
        std::vector<std::string> r;
        for (auto it = where_.begin(); it != where_.end(); ++it)
            if (it.key().rfind(p, 0) == 0)
                r.push_back(it.key());
        return r;
    }

private:
    Mapped& file(const std::string& shard) {
        auto it = files_.find(shard);
        if (it != files_.end())
            return it->second;
        Mapped m;
        const std::string path = dir_ + "/" + shard;
        const int fd = open(path.c_str(), O_RDONLY);
        struct stat st {};
        fstat(fd, &st);
        m.size = static_cast<size_t>(st.st_size);
        m.base = static_cast<const uint8_t*>(mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, fd, 0));
        close(fd);
        uint64_t n = 0;
        std::memcpy(&n, m.base, 8);
        m.hdr = nlohmann::json::parse(m.base + 8, m.base + 8 + n);
        m.data_off = 8 + n;
        return files_.emplace(shard, std::move(m)).first->second;
    }
    std::string dir_;
    nlohmann::json where_;
    std::map<std::string, Mapped> files_;
};

// BF16 [rows, cols] host rows -> FP16 device tensor.
imp::Tensor bf16_rows_to_device(const imp::Tensor& t, int64_t rows, std::vector<void*>& allocs) {
    const int64_t cols = t.shape[1];
    std::vector<__half> h(static_cast<size_t>(rows * cols));
    const uint16_t* src = static_cast<const uint16_t*>(t.data);
    for (size_t i = 0; i < h.size(); ++i) {
        const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        h[i] = __float2half(f);
    }
    void* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(__half)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);
    allocs.push_back(d);
    const int64_t shape[2] = {rows, cols};
    return imp::Tensor(d, imp::QType::F16, 2, shape, /*on_device=*/true);
}

struct RefStep {
    int token = 0, pos = 0, argmax = 0;
    float noise_logit = 0.0f, noise_hidden = 0.0f;
    std::vector<float> sample;
    std::vector<int32_t> ids, top_ids;
    std::vector<float> logits, top_vals;
    std::vector<int32_t> experts;
};

// Text fixture: header "imp-qwen4exp-mtp-ref 1 <steps>", per step one "step ..." line, then
// "<tag> <count> values..." lines (sample, ids, logits, top_ids, top_vals, experts).
bool read_ref(const std::string& path, std::vector<RefStep>& out) {
    std::ifstream f(path);
    std::string magic, tag;
    int ver = 0, n = 0;
    if (!(f >> magic >> ver >> n) || magic != "imp-qwen4exp-mtp-ref" || ver != 1)
        return false;
    auto list = [&](const char* want, auto& v) {
        size_t k = 0;
        if (!(f >> tag >> k) || tag != want)
            return false;
        v.resize(k);
        for (auto& x : v)
            f >> x;
        return static_cast<bool>(f);
    };
    for (int s = 0; s < n; ++s) {
        RefStep r;
        if (!(f >> tag >> r.token >> r.pos >> r.argmax >> r.noise_logit >> r.noise_hidden) || tag != "step")
            return false;
        if (!list("sample", r.sample) || !list("ids", r.ids) || !list("logits", r.logits) ||
            !list("top_ids", r.top_ids) || !list("top_vals", r.top_vals) || !list("experts", r.experts))
            return false;
        out.push_back(std::move(r));
    }
    return true;
}

std::vector<float> d2h_f16(const void* d, size_t n) {
    std::vector<__half> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i)
        f[i] = __half2float(h[i]);
    return f;
}

}  // namespace

TEST(MtpQwen4ExpReference, TwoDraftStepsMatchVllmMath) {
    const char* env = std::getenv("IMP_TEST_MODEL_MTP_QWEN4EXP");
    const std::string dir = (env && *env) ? env : "/models/Qwen3.8-Flash-Next-NVFP4";
    const std::string ref_path = std::string(IMP_TEST_FIXTURES_DIR) + "/mtp_qwen4exp_ref.txt";
    if (!fs::exists(dir + "/model-fp8-mtp-ple.safetensors"))
        GTEST_SKIP() << "Qwen3.8-Flash-Next checkpoint not present at " << dir;
    std::vector<RefStep> ref;
    ASSERT_TRUE(read_ref(ref_path, ref)) << "fixture " << ref_path;
    ASSERT_EQ(ref.size(), 2u);

    Checkpoint ck(dir);
    std::unordered_map<std::string, imp::Tensor> tm;
    size_t bytes = 0;
    for (const std::string& n : ck.names_with_prefix("mtp.")) {
        imp::Tensor t;
        ASSERT_TRUE(ck.tensor(n, t));
        bytes += t.nbytes();
        tm.emplace(n, t);
    }
    imp::MtpHead head = imp::dispatch_mtp_head(tm, dir, bytes);
    ASSERT_TRUE(head.loaded);
    ASSERT_EQ(head.layout, imp::MtpLayout::Qwen4Exp);
    ASSERT_EQ(head.hc_count, kHc);

    std::vector<void*> allocs, head_allocs;
    ASSERT_TRUE(imp::upload_mtp_head(head, imp::QType::F16, /*arch_norm_offset=*/1.0f, nullptr, head_allocs));
    imp::Tensor emb_bf16, lm_bf16;
    ASSERT_TRUE(ck.tensor("model.language_model.embed_tokens.weight", emb_bf16));
    ASSERT_TRUE(ck.tensor("lm_head.weight", lm_bf16));
    const imp::Tensor emb = bf16_rows_to_device(emb_bf16, *std::max_element(kTokens, kTokens + 2) + 1, allocs);
    const imp::Tensor lm = bf16_rows_to_device(lm_bf16, kVocab, allocs);

    imp::MtpDraftWorkspace ws{};
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws, kD, kVocab, kExperts, kTopK, kEff, /*shared_d_ff=*/kEff, kHeads,
                                            kKvHeads, kHeadDim, /*max_seq_len=*/16, /*n_kv_slots=*/1, kHc,
                                            kLowrank));
    ws.rope_theta = 1.0e7f;
    ws.rope_dim = 64;
    ws.mrope_sec0 = 11, ws.mrope_sec1 = 11, ws.mrope_sec2 = 10;
    ws.rms_norm_eps = 1e-6f;
    ws.arch_norm_offset = 0.0f;  // the +1 is baked in at upload

    // h_prev: k / 1024, k = ((i * 2654435761) mod 2^32 >> 20) - 2048, exact in FP16 (as the script).
    std::vector<__half> h(kHc * kD);
    for (uint32_t i = 0; i < h.size(); ++i)
        h[i] = __float2half(static_cast<float>(static_cast<int>((i * 2654435761u) >> 20) - 2048) / 1024.0f);
    void* d_h = nullptr;
    ASSERT_EQ(cudaMalloc(&d_h, h.size() * sizeof(__half)), cudaSuccess);
    allocs.push_back(d_h);
    ASSERT_EQ(cudaMemcpy(d_h, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);

    const void* h_prev = d_h;
    for (int s = 0; s < 2; ++s) {
        const RefStep& r = ref[static_cast<size_t>(s)];
        ASSERT_EQ(r.token, kTokens[s]);
        ASSERT_EQ(ws.mtp_pos, r.pos);
        int tok = -1;
        ASSERT_TRUE(imp::mtp_draft_step(r.token, h_prev, head, emb, lm, ws, kD, kVocab, &tok, nullptr));
        const std::vector<float> lg = d2h_f16(ws.d_logits, kVocab);
        const std::vector<float> sample = d2h_f16(ws.d_h_final, kD);
        std::vector<int32_t> experts(kTopK);
        cudaMemcpy(experts.data(), ws.routing_buf.expert_indices, kTopK * sizeof(int32_t), cudaMemcpyDeviceToHost);

        float dh = 0.0f, dl = 0.0f;
        for (int i = 0; i < kD; ++i)
            dh = std::max(dh, std::fabs(sample[i] - r.sample[i]));
        for (size_t i = 0; i < r.ids.size(); ++i)
            dl = std::max(dl, std::fabs(lg[r.ids[i]] - r.logits[i]));
        for (size_t i = 0; i < r.top_ids.size(); ++i)
            dl = std::max(dl, std::fabs(lg[r.top_ids[i]] - r.top_vals[i]));
        const float band_l = 4.0f * r.noise_logit + 1.0f / 128.0f;
        const float band_h = 4.0f * r.noise_hidden + 1.0f / 128.0f;
        std::printf("[qwen4exp-ref] step %d pos %d: argmax imp %d ref %d, max|dlogit| %.4f (band %.4f, %zu ids), "
                    "max|dsample| %.4f (band %.4f), experts %s\n",
                    s, r.pos, tok, r.argmax, dl, band_l, r.ids.size() + r.top_ids.size(), dh, band_h,
                    std::equal(experts.begin(), experts.end(), r.experts.begin()) ? "same" : "DIFFER");
        EXPECT_EQ(tok, r.argmax);
        EXPECT_LE(dl, band_l);
        EXPECT_LE(dh, band_h);
        EXPECT_TRUE(std::is_permutation(experts.begin(), experts.end(), r.experts.begin()));
        h_prev = ws.d_hc_x;  // chain: the next step reads this step's multi_hidden
    }
    imp::mtp_workspace_free(ws);
    for (void* p : allocs)
        cudaFree(p);
    for (void* p : head_allocs)
        cudaFreeAsync(p, nullptr);  // stream-ordered pool, as Model frees its weights
    cudaDeviceSynchronize();
}
