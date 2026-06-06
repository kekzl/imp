// =============================================================================
// LoRA hot-swap E2E (issue #522) — C-API level, real model.
//
// Strategy: synthetic PEFT adapters crafted in-test (full control, no
// network):
//   1. zero-B adapter      → delta is exactly 0 → greedy output must be
//      BIT-IDENTICAL to base. This catches every wiring bug (wrong x, wrong
//      buffer, wrong layer) because any misapplied delta breaks identity.
//   2. nonzero-B adapter   → greedy output must DIFFER (the delta reaches
//      the logits) while the engine stays healthy (no abort, non-empty).
//   3. swap back to base   → bit-identical to the original base output
//      (graph re-capture path; a stale capture would reproduce the adapter).
//
// Default model: Llama-3.2-3B (q_proj/v_proj adapters, r=8) — the classic
// PEFT target set. Skips when the model file is absent.
// =============================================================================

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "imp/imp.h"

namespace {

const char* model_path() {
    if (const char* p = getenv("IMP_TEST_MODEL_LLAMA"))
        return p;
    return "/models/Llama-3.2-3B-Instruct-Q8_0.gguf";
}

// Minimal safetensors writer: header JSON + F32 tensor blobs.
struct StWriter {
    std::string header_entries;
    std::vector<float> data;

    void add(const std::string& name, int64_t d0, int64_t d1, bool ramp) {
        size_t start = data.size() * sizeof(float);
        for (int64_t i = 0; i < d0 * d1; i++) {
            float v = ramp ? 0.02f * static_cast<float>(static_cast<int>((i * 7 + 3) % 11) - 5) : 0.0f;
            data.push_back(v);
        }
        size_t end = data.size() * sizeof(float);
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[%lld,%lld],\"data_offsets\":[%zu,%zu]}",
                 header_entries.empty() ? "" : ",", name.c_str(), (long long)d0, (long long)d1, start,
                 end);
        header_entries += buf;
    }

    void write(const std::string& path) {
        std::string hdr = "{" + header_entries + "}";
        while (hdr.size() % 8 != 0)  // safetensors pads headers with spaces
            hdr += ' ';
        uint64_t hlen = hdr.size();
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&hlen), 8);
        f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size() * sizeof(float)));
    }
};

// Build a PEFT adapter dir targeting q_proj/v_proj of layers 0..2.
// zero_B=true → mathematically a no-op adapter.
std::string make_adapter(int d_model, int kv_rows, bool zero_B, const char* dirname) {
    const int r = 8;
    std::string dir = std::string(testing::TempDir()) + dirname;
    std::filesystem::create_directories(dir);

    std::ofstream cfg(dir + "/adapter_config.json");
    cfg << "{\"r\": 8, \"lora_alpha\": 16, \"use_rslora\": false, "
           "\"target_modules\": [\"q_proj\", \"v_proj\"]}";
    cfg.close();

    StWriter w;
    for (int layer = 0; layer < 3; layer++) {
        char key[160];
        snprintf(key, sizeof(key), "base_model.model.model.layers.%d.self_attn.q_proj.lora_A.weight",
                 layer);
        w.add(key, r, d_model, /*ramp=*/true);
        snprintf(key, sizeof(key), "base_model.model.model.layers.%d.self_attn.q_proj.lora_B.weight",
                 layer);
        w.add(key, d_model, r, /*ramp=*/!zero_B);
        snprintf(key, sizeof(key), "base_model.model.model.layers.%d.self_attn.v_proj.lora_A.weight",
                 layer);
        w.add(key, r, d_model, /*ramp=*/true);
        snprintf(key, sizeof(key), "base_model.model.model.layers.%d.self_attn.v_proj.lora_B.weight",
                 layer);
        w.add(key, kv_rows, r, /*ramp=*/!zero_B);
    }
    w.write(dir + "/adapter_model.safetensors");
    return dir;
}

// Teacher-forced PPL is the project's bit-stable A/B instrument under
// deterministic mode (#542) — greedy TEXT comparison is invalid here because
// dense greedy logit ties flip across runs (docs/determinism.md), which a
// pre-LoRA control run confirmed on this exact model/prompt.
double ppl(ImpModel model, ImpContext ctx, const char* text) {
    std::vector<int32_t> toks(512);
    int n = 0;
    EXPECT_EQ(imp_tokenize(model, text, toks.data(), &n, 512), IMP_SUCCESS);
    double out = 0.0;
    EXPECT_EQ(imp_perplexity(ctx, toks.data(), n, &out), IMP_SUCCESS);
    return out;
}

// Greedy generation as a HEALTH probe only (non-empty, no abort) — never
// compared for equality.
std::string greedy(ImpContext ctx, const char* prompt) {
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 24;
    params.temperature = 0.0f;
    params.seed = 42;
    params.apply_chat_template = 0;
    char out[4096];
    size_t out_len = 0;
    if (imp_generate(ctx, prompt, &params, out, sizeof(out), &out_len) != IMP_SUCCESS)
        return "";
    return std::string(out, out_len);
}

TEST(LoraHotSwap, ZeroAdapterIdentity_EffectAdapterDiffers_SwapBack) {
    if (!std::filesystem::exists(model_path()))
        GTEST_SKIP() << "model not present: " << model_path();

    // The identity assertions need bit-stable greedy across runs — dense
    // greedy logit ties otherwise flip mid-sequence (documented determinism
    // boundary, docs/determinism.md). Same env-seed mechanism as
    // DetEvalE2ETest: must be set BEFORE model/engine creation.
    setenv("IMP_DETERMINISTIC", "1", 1);

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(model_path(), IMP_FORMAT_GGUF, &model), IMP_SUCCESS);
    ImpConfig config = imp_config_default();
    config.max_seq_len = 512;
    config.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &config, &ctx), IMP_SUCCESS);

    const int d_model = imp_model_d_model(model);
    const int kv_rows = 1024;  // Llama-3.2-3B: 8 KV heads × 128

    const char* corpus =
        "The three primary colors are red, blue, and yellow. Mixing two primary colors "
        "produces a secondary color: red and blue make purple, blue and yellow make green, "
        "and red and yellow make orange.";

    const double base = ppl(model, ctx, corpus);
    ASSERT_GT(base, 1.0);
    // Control: teacher-forced NLL is bit-stable in deterministic mode — the
    // identity assertions below ride on this.
    const double base2 = ppl(model, ctx, corpus);
    ASSERT_EQ(base, base2) << "baseline PPL not bit-stable — deterministic mode not active?";

    // --- 1. zero-B adapter: delta is exactly 0 → bit-identical PPL ---
    std::string zero_dir = make_adapter(d_model, kv_rows, /*zero_B=*/true, "imp_lora_zero");
    int32_t zero_id = 0;
    ASSERT_EQ(imp_lora_load(ctx, zero_dir.c_str(), &zero_id), IMP_SUCCESS);
    ASSERT_GE(zero_id, 1);
    ASSERT_EQ(imp_lora_set(ctx, zero_id), IMP_SUCCESS);
    const double with_zero = ppl(model, ctx, corpus);
    EXPECT_EQ(base, with_zero) << "zero-B adapter must be a mathematical no-op (wiring bug otherwise)";

    // --- 2. effect adapter: PPL must move materially, generation healthy ---
    std::string eff_dir = make_adapter(d_model, kv_rows, /*zero_B=*/false, "imp_lora_eff");
    int32_t eff_id = 0;
    ASSERT_EQ(imp_lora_load(ctx, eff_dir.c_str(), &eff_id), IMP_SUCCESS);
    ASSERT_EQ(imp_lora_set(ctx, eff_id), IMP_SUCCESS);
    const double with_eff = ppl(model, ctx, corpus);
    EXPECT_GT(std::abs(with_eff - base) / base, 0.01)
        << "a strong nonzero adapter must move teacher-forced PPL (base=" << base
        << ", eff=" << with_eff << ")";
    std::string gen = greedy(ctx, "The three primary colors are");
    EXPECT_GE(gen.size(), 1u) << "generation with active adapter must not abort";

    // --- 3. back to base: bit-identical to the original PPL ---
    ASSERT_EQ(imp_lora_set(ctx, 0), IMP_SUCCESS);
    const double back = ppl(model, ctx, corpus);
    EXPECT_EQ(base, back) << "deactivating must restore the base model exactly (stale graph capture?)";

    imp_context_free(ctx);
    imp_model_free(model);
}

}  // namespace
