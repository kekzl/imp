// GGUF vs SafeTensors loader parity on what attention reads per layer (#2530): RoPE theta, freq
// scale, inv-freq tables, softmax scale, embed/logit scales. One 64-d stub per arch, written as
// GGUF metadata and as config.json; load_gguf and load_safetensors must agree.
#include <gtest/gtest.h>

#include "gguf_stub.h"
#include "model/gguf_loader.h"
#include "model/model.h"
#include "model/model_arch.h"
#include "model/model_config.h"
#include "model/safetensors_loader.h"

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using imp::ModelArch;

struct LayerRope {
    double theta = 0, freq_scale = 0, softmax_scale = 0;
    int head_dim = 0, rot_dim = 0;
    std::vector<double> inv_short, inv_long;
};

struct RopeView {
    ModelArch arch = ModelArch::GENERIC;
    std::vector<LayerRope> layers;
    double embed_scale = 0, logits_scaling = 0, yarn_ext = 0, yarn_attn = 0;
};

// Per-layer resolution as executor_attention.cpp (theta, Gemma-4 rope_freqs) and
// executor_workspace.cpp (short/long factor tables) do it. rope_neox is left out: GGUF
// pre-permutes Q/K for interleaved RoPE, SafeTensors runs NeoX on the same weights.
RopeView view_of(const imp::Model& m) {
    const imp::ModelConfig& c = m.config();
    const bool g4 = c.arch == ModelArch::GEMMA4;
    RopeView v;
    v.arch = c.arch;
    v.embed_scale = c.embed_scale;
    v.logits_scaling = c.logits_scaling;
    v.yarn_ext = c.yarn_ext_factor;
    v.yarn_attn = c.yarn_attn_factor;
    for (int i = 0; i < c.n_layers; ++i) {
        LayerRope r;
        const bool swa = i < static_cast<int>(c.swa_layers.size()) && c.swa_layers[i];
        r.head_dim = i < static_cast<int>(c.head_dim_per_layer.size()) ? c.head_dim_per_layer[i] : c.head_dim;
        r.theta = c.rope_theta;
        r.freq_scale = c.rope_freq_scale;
        if (g4 && swa) {
            r.theta = c.rope_theta_swa > 0.0f ? c.rope_theta_swa : c.rope_local_theta;
            r.freq_scale = 1.0;
        } else if (!g4 && c.sliding_window_pattern > 0 &&
                   i % c.sliding_window_pattern != c.sliding_window_pattern - 1) {
            if (c.rope_local_theta > 0.0f)
                r.theta = c.rope_local_theta;
            r.freq_scale = 1.0;
        }
        r.rot_dim = g4 ? r.head_dim : (c.rope_dim > 0 && c.rope_dim <= r.head_dim ? c.rope_dim : r.head_dim);
        const imp::Tensor& t = m.layer(i).rope_freqs;
        if (g4 && !swa && t.data != nullptr && t.qtype == imp::QType::F32) {
            const float* f = static_cast<const float*>(t.data);
            r.inv_short.assign(f, f + t.shape[0]);
            r.inv_long = r.inv_short;
        } else {
            const int pairs = r.rot_dim / 2;
            for (int p = 0; p < pairs; ++p) {
                const double base = std::pow(r.theta, -2.0 * p / r.rot_dim);
                const bool sf = static_cast<int>(c.rope_short_factor.size()) == pairs;
                const bool lf = static_cast<int>(c.rope_long_factor.size()) == pairs;
                r.inv_short.push_back(base / (sf ? c.rope_short_factor[p] : 1.0f));
                r.inv_long.push_back(base / (lf ? c.rope_long_factor[p] : 1.0f));
            }
        }
        r.softmax_scale = imp::attention_softmax_scale(c, g4, r.head_dim);
        v.layers.push_back(r);
    }
    return v;
}

// 1e-30 GGUF divisors (unrotated pairs) give ~1e-36, the HF table writes 0: absolute floor 1e-20.
bool close(double a, double b) {
    return std::fabs(a - b) <= 1e-5 * std::max(std::fabs(a), std::fabs(b)) + 1e-20;
}

void expect_parity(const RopeView& g, const RopeView& s) {
    ASSERT_EQ(g.arch, s.arch);
    ASSERT_EQ(g.layers.size(), s.layers.size());
    EXPECT_TRUE(close(g.embed_scale, s.embed_scale)) << g.embed_scale << " vs " << s.embed_scale;
    EXPECT_TRUE(close(g.logits_scaling, s.logits_scaling)) << g.logits_scaling << " vs " << s.logits_scaling;
    EXPECT_TRUE(close(g.yarn_ext, s.yarn_ext)) << g.yarn_ext << " vs " << s.yarn_ext;
    EXPECT_TRUE(close(g.yarn_attn, s.yarn_attn)) << g.yarn_attn << " vs " << s.yarn_attn;
    for (size_t i = 0; i < g.layers.size(); ++i) {
        const LayerRope& a = g.layers[i];
        const LayerRope& b = s.layers[i];
        SCOPED_TRACE("layer " + std::to_string(i));
        EXPECT_TRUE(close(a.theta, b.theta)) << a.theta << " vs " << b.theta;
        EXPECT_TRUE(close(a.freq_scale, b.freq_scale)) << a.freq_scale << " vs " << b.freq_scale;
        EXPECT_TRUE(close(a.softmax_scale, b.softmax_scale)) << a.softmax_scale << " vs " << b.softmax_scale;
        EXPECT_EQ(a.head_dim, b.head_dim);
        EXPECT_EQ(a.rot_dim, b.rot_dim);
        ASSERT_EQ(a.inv_short.size(), b.inv_short.size());
        ASSERT_EQ(a.inv_long.size(), b.inv_long.size());
        for (size_t p = 0; p < a.inv_short.size(); ++p) {
            EXPECT_TRUE(close(a.inv_short[p], b.inv_short[p]))
                << "short pair " << p << ": gguf " << a.inv_short[p] << " vs safetensors " << b.inv_short[p];
            EXPECT_TRUE(close(a.inv_long[p], b.inv_long[p]))
                << "long pair " << p << ": gguf " << a.inv_long[p] << " vs safetensors " << b.inv_long[p];
        }
    }
}

struct LoadedPair {
    std::string gguf_path, hf_dir;
    std::unique_ptr<imp::Model> gguf, st;
    LoadedPair(const imp::test::GgufStubSpec& spec, const std::string& config_json) {
        gguf_path = imp::test::generate_gguf_stub(spec);
        hf_dir = imp::test::generate_hf_stub(config_json, spec.n_layers);
        if (!gguf_path.empty())
            gguf = imp::load_gguf(gguf_path);
        if (!hf_dir.empty())
            st = imp::load_safetensors(hf_dir);
    }
    ~LoadedPair() {
        gguf.reset();
        st.reset();
        if (!gguf_path.empty())
            imp::test::remove_gguf_stub(gguf_path);
        if (!hf_dir.empty())
            imp::test::remove_gguf_stub(hf_dir);
    }
};

// Stub geometry shared by both files (tests/gguf_stub.cpp): d_model 64, 2 heads, head_dim 32.
std::string hf_config(const std::string& arch_class, int n_layers, const std::string& extra) {
    std::ostringstream o;
    o << R"({"architectures": [")" << arch_class << R"("], "hidden_size": 64, "num_attention_heads": 2, )"
      << R"("num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 128, "num_hidden_layers": )"
      << n_layers << R"(, "vocab_size": 256, "max_position_embeddings": 512, "rms_norm_eps": 1e-5)"
      << (extra.empty() ? "" : ", ") << extra << "}";
    return o.str();
}

// *out = the GGUF view, for per-test checks that the stub reached the path under test.
void run_parity(const imp::test::GgufStubSpec& spec, const std::string& config_json, ModelArch want,
                RopeView* out) {
    LoadedPair p(spec, config_json);
    ASSERT_FALSE(p.gguf_path.empty());
    ASSERT_FALSE(p.hf_dir.empty());
    ASSERT_NE(p.gguf, nullptr) << "load_gguf failed";
    ASSERT_NE(p.st, nullptr) << "load_safetensors failed";
    ASSERT_EQ(p.gguf->config().arch, want);
    *out = view_of(*p.gguf);
    expect_parity(*out, view_of(*p.st));
}

int rotated_pairs(const LayerRope& r) {
    int n = 0;
    for (double f : r.inv_short)
        n += f > 1e-20 ? 1 : 0;
    return n;
}

}  // namespace

// Gemma-4 global layers: partial_rotary_factor 0.25 (config.json) vs rope_freqs.weight with
// 4 x 1.0 and 12 x 1e30 divisors (GGUF): 4 of 16 pairs rotate. SWA layer: theta 1e4 both.
TEST(LoaderRopeParity, Gemma4ProportionalRope) {
    imp::test::GgufStubSpec spec;
    spec.arch = "gemma4";
    spec.n_layers = 2;
    spec.rope_freqs.assign(16, 1e30f);
    for (int p = 0; p < 4; ++p)
        spec.rope_freqs[p] = 1.0f;
    spec.u32 = {{"attention.key_length", 32},
                {"attention.key_length_swa", 32},
                {"attention.sliding_window", 1024}};
    spec.f32 = {{"rope.freq_base", 1e6f}, {"rope.freq_base_swa", 1e4f}};
    spec.i32_arrays = {{"attention.sliding_window_pattern", {1, 0}}};
    const std::string cfg =
        hf_config("Gemma4ForCausalLM", 2,
                  R"("global_head_dim": 32, "sliding_window": 1024, )"
                  R"("layer_types": ["sliding_attention", "full_attention"], )"
                  R"("rope_parameters": {"full_attention": {"rope_type": "proportional", )"
                  R"("rope_theta": 1000000.0, "partial_rotary_factor": 0.25}, )"
                  R"("sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}})");
    RopeView g;
    ASSERT_NO_FATAL_FAILURE(run_parity(spec, cfg, ModelArch::GEMMA4, &g));
    ASSERT_EQ(g.layers.size(), 2u);
    EXPECT_EQ(rotated_pairs(g.layers[0]), 16);  // SWA: full rotation at theta 1e4
    EXPECT_EQ(rotated_pairs(g.layers[1]), 4);   // global: 4 of 16 pairs
    EXPECT_DOUBLE_EQ(g.layers[0].theta, 1e4);
}

// Llama 3 scaling: rope_freqs.weight divisors (GGUF) vs rope_scaling llama3 (config.json).
// theta 5e5, rd 32: pairs span unscaled, smoothed and fully scaled regions.
TEST(LoaderRopeParity, Llama3RopeScaling) {
    constexpr double kTheta = 500000.0, kFactor = 32.0, kLow = 1.0, kHigh = 4.0, kOrig = 8192.0;
    imp::test::GgufStubSpec spec;
    spec.arch = "llama";
    spec.f32 = {{"rope.freq_base", static_cast<float>(kTheta)}};
    for (int i = 0; i < 16; ++i) {  // HF _compute_llama3_parameters, per pair
        const double wavelen = 2.0 * M_PI * std::pow(kTheta, 2.0 * i / 32.0);
        double f = kFactor;
        if (wavelen < kOrig / kHigh)
            f = 1.0;
        else if (wavelen <= kOrig / kLow) {
            const double smooth = (kOrig / wavelen - kLow) / (kHigh - kLow);
            f = kFactor / (1.0 - smooth + smooth * kFactor);
        }
        spec.rope_freqs.push_back(static_cast<float>(f));
    }
    const std::string cfg = hf_config("LlamaForCausalLM", 1,
                                      R"("rope_theta": 500000.0, "rope_scaling": {"rope_type": "llama3", )"
                                      R"("factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, )"
                                      R"("original_max_position_embeddings": 8192})");
    RopeView g;
    ASSERT_NO_FATAL_FAILURE(run_parity(spec, cfg, ModelArch::LLAMA, &g));
    // Last pair fully scaled: base / 32, not the plain table.
    EXPECT_NEAR(g.layers[0].inv_short.back() * kFactor, std::pow(kTheta, -30.0 / 32.0), 1e-9);
}

// Granite multipliers: attention.scale/embedding_scale/residual_scale/logit_scale (GGUF) vs
// attention_multiplier/embedding_multiplier/residual_multiplier/logits_scaling (config.json).
// logits_scaling stays 1.0: both loaders refuse any other value.
TEST(LoaderRopeParity, GraniteMultipliers) {
    imp::test::GgufStubSpec spec;
    spec.arch = "granite";
    spec.f32 = {{"rope.freq_base", 10000.0f},
                {"attention.scale", 0.0078125f},
                {"embedding_scale", 12.0f},
                {"residual_scale", 0.22f},
                {"logit_scale", 1.0f}};
    const std::string cfg = hf_config("GraniteForCausalLM", 1,
                                      R"("rope_theta": 10000.0, "attention_multiplier": 0.0078125, )"
                                      R"("embedding_multiplier": 12.0, "residual_multiplier": 0.22, )"
                                      R"("logits_scaling": 1.0)");
    RopeView g;
    ASSERT_NO_FATAL_FAILURE(run_parity(spec, cfg, ModelArch::GRANITE, &g));
    EXPECT_FLOAT_EQ(g.layers[0].softmax_scale, 0.0078125f);
    EXPECT_FLOAT_EQ(g.embed_scale, 12.0f / 0.22f);
}

// Qwen3: plain RoPE at theta 1e6, head_dim from attention.key_length vs head_dim.
TEST(LoaderRopeParity, Qwen3PlainRope) {
    imp::test::GgufStubSpec spec;
    spec.arch = "qwen3";
    spec.u32 = {{"attention.key_length", 32}};
    spec.f32 = {{"rope.freq_base", 1e6f}};
    const std::string cfg = hf_config("Qwen3ForCausalLM", 1, R"("rope_theta": 1000000.0)");
    RopeView g;
    ASSERT_NO_FATAL_FAILURE(run_parity(spec, cfg, ModelArch::QWEN3, &g));
    EXPECT_DOUBLE_EQ(g.layers[0].theta, 1e6);
    EXPECT_FLOAT_EQ(g.layers[0].softmax_scale, 1.0f / std::sqrt(32.0f));
}
