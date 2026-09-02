// Tests that read a real model file from disk: a checkpoint's safetensors
// index, an HF-cache spiece.model. GPU lane (test-e2e), where /models and the
// HF cache are mounted; the unit lane has no models and no skips.

#include "model/mtp_head.h"
#include "model/sentencepiece_loader.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace imp {
namespace {

// The synthetic cases above pin the rule; this one pins it against a real
// checkpoint, because a hand-written weight_map is a statement about what I
// think checkpoints look like. Reads the index only, no GPU, no weights.
// Skipped where the model is absent, which includes CI.
TEST(MtpProbe, FindsTheHeadInARealCheckpoint) {
    const char* env = std::getenv("IMP_MTP_MODEL");
    std::string path = env ? env : "/models/Qwen3.8-27B-NVFP4";
    if (!fs::exists(path))
        GTEST_SKIP() << "checkpoint not present: " << path;
    EXPECT_TRUE(imp::probe_mtp_head(path)) << "known to carry mtp.fc.weight (15 tensors)";
}

// Optional integration test: parse a real spiece.model from the HF cache.
// Skipped when the file is not present (the test runner box may differ).
TEST(SentencePieceLoader, RealHfCacheSpieceModelLoadsCleanly) {
    namespace fs = std::filesystem;
    const char* home = std::getenv("HOME");
    std::vector<std::string> candidates = {
        // Container path (when -v $HOME/.cache/huggingface:/hf_cache is bound)
        "/hf_cache/hub/models--facebook--musicgen-small/"
        "snapshots/4c8334b02c6ec4e8664a91979669a501ec497792/spiece.model",
    };
    if (home) {
        candidates.push_back(std::string(home) +
            "/.cache/huggingface/hub/models--facebook--musicgen-small/"
            "snapshots/4c8334b02c6ec4e8664a91979669a501ec497792/spiece.model");
    }

    std::string found;
    for (const auto& p : candidates) {
        if (fs::exists(p)) {
            found = p;
            break;
        }
    }
    if (found.empty()) {
        GTEST_SKIP() << "no real spiece.model fixture present";
    }

    SentencePieceModel m = load_sentencepiece_model_file(found);
    EXPECT_FALSE(m.empty());
    EXPECT_GT(m.pieces.size(), 100u) << "expected a non-trivial vocabulary";
    // T5 vocab has <pad>, </s>, <unk> as ids 0, 1, 2.
    EXPECT_EQ(m.pieces[0], "<pad>");
    EXPECT_EQ(m.pieces[1], "</s>");
    EXPECT_EQ(m.pieces[2], "<unk>");
    EXPECT_EQ(m.pieces.size(), m.scores.size());
    EXPECT_EQ(m.pieces.size(), m.types.size());
}

}  // namespace
}  // namespace imp
