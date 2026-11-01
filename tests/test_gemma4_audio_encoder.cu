// Gemma-4 audio tower on the GPU vs HF transformers (Gemma4AudioModel + Gemma4MultimodalEmbedder,
// fp32 CPU) on a 2.3 s synthetic clip. Needs the checkpoint (IMP_TEST_MODEL_GEMMA4_AUDIO, a
// single-file model.safetensors such as gemma-4-E4B-it) and the reference dir
// (IMP_TEST_GEMMA4_AUDIO_REF: features.f32, features_pcm16.f32, features_mask.u8, sscp.f32, layer0.f32,
// tower_out.f32, embeds.f32, signal.wav from tools/analysis/gemma4_audio_ref.py). Skips otherwise.

#include "audio/audio_features.h"
#include "audio/audio_pipeline.h"
#include "audio/gemma4_audio_encoder.h"
#include "audio/gemma4_audio_model.h"
#include "model/json_util.h"
#include "model/safetensors_raw.h"
#include "scoped_engine_arena.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

const char* model_dir() { return std::getenv("IMP_TEST_MODEL_GEMMA4_AUDIO"); }
const char* ref_dir() { return std::getenv("IMP_TEST_GEMMA4_AUDIO_REF"); }

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::vector<float> read_f32(const std::string& path) {
    const auto b = read_bytes(path);
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

// Worst row cosine and global relative L2 error over the first `rows` rows of [*, width].
struct Diff {
    double min_cos = 1.0;
    double rel_l2 = 0.0;
};
Diff compare(const std::vector<float>& got, const std::vector<float>& ref, size_t rows, size_t width) {
    Diff d;
    double num = 0.0, den = 0.0;
    for (size_t r = 0; r < rows; ++r) {
        double dot = 0.0, ng = 0.0, nr = 0.0;
        for (size_t c = 0; c < width; ++c) {
            const double g = got[(r * width) + c];
            const double e = ref[(r * width) + c];
            dot += g * e;
            ng += g * g;
            nr += e * e;
            num += (g - e) * (g - e);
        }
        den += nr;
        d.min_cos = std::min(d.min_cos, dot / (std::sqrt(ng * nr) + 1e-30));
    }
    d.rel_l2 = std::sqrt(num / (den + 1e-30));
    return d;
}

class Gemma4AudioEncoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_dir() || !ref_dir())
            GTEST_SKIP() << "IMP_TEST_MODEL_GEMMA4_AUDIO / IMP_TEST_GEMMA4_AUDIO_REF not set";
        const std::string dir = model_dir();
        ASSERT_EQ(st_.open(dir + "/model.safetensors"), "");
        for (const RawTensor& t : st_.tensors()) {
            if (!is_gemma4_audio_tensor_name(t.name))
                continue;
            ASSERT_EQ(t.dtype, "BF16") << t.name;
            int64_t shape[4] = {};
            for (size_t i = 0; i < t.shape.size() && i < 4; ++i)
                shape[i] = t.shape[i];
            tensors_.emplace(t.name, Tensor(const_cast<void*>(t.data), QType::BF16,
                                            static_cast<int>(t.shape.size()), shape, false));
        }
        tower_ = load_gemma4_audio_from_checkpoint(dir, tensors_);
        ASSERT_NE(tower_, nullptr);
        const size_t demand = AudioPipeline::demand_bytes(*tower_);
        arena_ = std::make_unique<ScopedEngineArena>(2 * demand);  // the test builds a second encoder
        ASSERT_TRUE(arena_->opened());
        ASSERT_TRUE(pipeline_.init(*tower_));
        ASSERT_EQ(pipeline_.taken_bytes(), demand) << "demand_bytes and init drifted apart";
    }

    AudioFeatures hf_features() const {
        const std::string r = ref_dir();
        AudioFeatures f;
        f.mel = read_f32(r + "/features.f32");
        f.mask = read_bytes(r + "/features_mask.u8");
        f.n_mels = 128;
        f.n_frames = static_cast<int>(f.mask.size());
        f.n_tokens = gemma4_audio_soft_tokens(f.mask, 750);
        return f;
    }

    RawSafeTensors st_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::unique_ptr<Gemma4AudioModel> tower_;
    std::unique_ptr<ScopedEngineArena> arena_;
    AudioPipeline pipeline_;
};

TEST_F(Gemma4AudioEncoderTest, WavFrontEndMatchesHfFeatures) {
    const std::string r = ref_dir();
    auto f = pipeline_.preprocess(read_bytes(r + "/signal.wav"));
    ASSERT_TRUE(f.has_value()) << f.error();
    const AudioFeatures ref = hf_features();
    ASSERT_EQ(f->n_frames, ref.n_frames);
    EXPECT_EQ(f->mask, ref.mask);
    EXPECT_EQ(f->n_tokens, ref.n_tokens);
    // Reference = HF features of the WAV's PCM16 samples: PCM16 noise lifts near-silent bins off the
    // log floor by up to 0.659 vs the float32 features, so those cannot be the bar.
    const std::vector<float> wav_ref = read_f32(r + "/features_pcm16.f32");
    ASSERT_EQ(wav_ref.size(), f->mel.size());
    double worst = 0.0;
    for (size_t i = 0; i < f->mel.size(); ++i)
        worst = std::max(worst, static_cast<double>(std::fabs(f->mel[i] - wav_ref[i])));
    std::printf("front end vs HF (PCM16 samples): worst |d log-mel| %.3g\n", worst);
    EXPECT_LT(worst, 1e-4);  // measured 4.77e-07 (2026-11-01)
}

TEST_F(Gemma4AudioEncoderTest, MatchesHfTowerAndEmbedder) {
    const std::string r = ref_dir();
    const AudioFeatures f = hf_features();
    const int lm_d = pipeline_.lm_d();
    const auto ref_emb = read_f32(r + "/embeds.f32");
    ASSERT_EQ(ref_emb.size(), static_cast<size_t>(f.n_tokens) * lm_d) << "token count disagrees with HF";

    half* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(f.n_tokens) * lm_d * sizeof(half)), cudaSuccess);
    Gemma4AudioEncoder enc;
    ASSERT_TRUE(enc.init(*tower_, Gemma4AudioEncoder::max_frames_for(Gemma4AudioFrontend{})));
    Gemma4AudioTaps taps;
    ASSERT_TRUE(enc.encode(f, d_out, nullptr, &taps));
    std::vector<half> h(static_cast<size_t>(f.n_tokens) * lm_d);
    ASSERT_EQ(cudaMemcpy(h.data(), d_out, h.size() * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
    (void)cudaFree(d_out);
    std::vector<float> got(h.size());
    for (size_t i = 0; i < h.size(); ++i)
        got[i] = __half2float(h[i]);

    const size_t T = static_cast<size_t>(f.n_tokens);
    const auto& c = tower_->config;
    const Diff sub = compare(taps.subsample, read_f32(r + "/sscp.f32"), T, c.hidden_size);
    const Diff l0 = compare(taps.layer0, read_f32(r + "/layer0.f32"), T, c.hidden_size);
    const Diff tow = compare(taps.tower, read_f32(r + "/tower_out.f32"), T, c.output_proj_dims);
    const Diff emb = compare(got, ref_emb, T, lm_d);
    std::printf(
        "subsample cos %.6f rel %.4g | layer0 cos %.6f rel %.4g | tower cos %.6f rel %.4g | "
        "embeds cos %.6f rel %.4g\n",
        sub.min_cos, sub.rel_l2, l0.min_cos, l0.rel_l2, tow.min_cos, tow.rel_l2, emb.min_cos, emb.rel_l2);
    EXPECT_GT(sub.min_cos, 0.9999);
    EXPECT_GT(l0.min_cos, 0.999);
    EXPECT_GT(tow.min_cos, 0.995);
    EXPECT_GT(emb.min_cos, 0.995);
    EXPECT_LT(emb.rel_l2, 0.05);
}

}  // namespace
}  // namespace imp
