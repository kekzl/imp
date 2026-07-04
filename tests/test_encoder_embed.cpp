// =============================================================================
// test_encoder_embed.cpp — encoder-only embedder e2e (#836, nomic-bert)
// =============================================================================
// Loads the real nomic-embed-text-v1.5 Q8_0 GGUF through the full engine path
// (encoder branch: upload + workspace, no KV/warmup/executor) and checks:
//   1. WordPiece tokenization matches the HF reference ids for a fixed string.
//   2. encoder_embed returns a unit-norm vector of d_model floats.
//   3. Semantic structure: cos(paraphrase pair) >> cos(unrelated pair), with
//      the HF-oracle-verified reference values as loose anchors
//      (imp 0.903 / 0.395; oracle cos(imp, hf) >= 0.999 on 2026-07-04).
// GTEST_SKIPs when the model file is absent.
// =============================================================================

#include "model/gguf_loader.h"
#include "model/model.h"
#include "runtime/engine.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

namespace {
constexpr const char kNomicGguf[] = "/models/nomic-embed-text-v1.5.Q8_0.gguf";

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
}  // namespace

TEST(EncoderEmbedTest, NomicBertEmbedsMatchReferenceStructure) {
    if (!fs::exists(kNomicGguf))
        GTEST_SKIP() << "nomic GGUF not present at " << kNomicGguf;

    std::shared_ptr<imp::Model> model = imp::load_gguf(kNomicGguf);
    ASSERT_NE(model, nullptr);
    model->build_profile();  // Engine::init does this in production
    ASSERT_TRUE(model->profile().is_encoder);

    // 1. WordPiece vs HF reference ids (bert-base-uncased vocab):
    //    "The lighthouse keeper swept the stairs." (no [CLS]/[SEP]).
    auto toks = model->tokenizer()->encode("The lighthouse keeper swept the stairs.");
    const std::vector<int32_t> ref = {1996, 10171, 10684, 7260, 1996, 5108, 1012};
    ASSERT_EQ(toks.size(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i) EXPECT_EQ(toks[i], ref[i]) << "token " << i;

    auto* tok = model->tokenizer();
    imp::EngineConfig ecfg{};
    imp::Engine engine;
    ASSERT_TRUE(engine.init(model, ecfg));
    ASSERT_TRUE(engine.is_encoder_model());

    auto embed = [&](const std::string& text) {
        std::vector<int32_t> t = tok->encode(text);
        std::vector<int32_t> framed;
        framed.push_back(101);  // [CLS]
        framed.insert(framed.end(), t.begin(), t.end());
        framed.push_back(102);  // [SEP]
        std::vector<float> out;
        EXPECT_TRUE(engine.encoder_embed(framed.data(), static_cast<int>(framed.size()), out));
        return out;
    };

    auto a = embed("The lighthouse keeper swept the stairs.");
    auto b = embed("A keeper of a lighthouse cleaned the steps.");
    auto c = embed("Quantum chromodynamics describes the strong interaction.");

    // 2. Unit norm, right dimensionality.
    ASSERT_EQ(a.size(), 768u);
    EXPECT_NEAR(std::sqrt(cosine(a, a)), 1.0f, 1e-3f);

    // 3. Semantic structure (HF-oracle anchors 0.903 / 0.395, loose bounds).
    const float sim_para = cosine(a, b);
    const float sim_unrel = cosine(a, c);
    EXPECT_GT(sim_para, 0.85f);
    EXPECT_LT(sim_unrel, 0.55f);
    EXPECT_GT(sim_para - sim_unrel, 0.3f);
}
