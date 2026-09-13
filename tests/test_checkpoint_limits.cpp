// Bounds on what a checkpoint may declare about itself (#1611-1613): layer count, expert
// count, shard filename, JSON/template nesting all come from an untrusted file. Failure mode
// without a refusal is not a wrong answer but an 18.9 TiB allocation or SIGSEGV before the
// first weight is read.
// Caps: sizeof(TransformerLayer)=9680B so INT_MAX layers = 18.9 TiB; JsonParser survives
// 30000 nesting levels on an 8MiB stack, SIGSEGVs before 40000; SafeTensors header may
// declare 128MiB.

#include "core/logging.h"
#include "model/jinja.h"
#include "model/json_util.h"
#include "model/model.h"
#include "model/model_limits.h"
#include "model/safetensors_loader.h"
#include "model/weight_map.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

// Any test reading a log line must pin the level itself: test_process_diag.cpp mutates the
// process-global level and never restores it, so seeing an ERROR line here depended on
// alphabetical suite order (green alone, red in the full binary).
struct LogLevelPin {
    LogLevel saved = log_get_level();
    LogLevelPin() { log_set_level(LogLevel::ERROR); }
    ~LogLevelPin() { log_set_level(saved); }
};

Tensor fake_weight(void* backing) {
    Tensor t;
    t.data = backing;
    t.qtype = QType::F16;
    t.ndim = 2;
    t.shape[0] = 8;
    t.shape[1] = 8;
    return t;
}

// ---------------------------------------------------------------------------
// #1611 - an index parsed out of a tensor name drives a resize
// ---------------------------------------------------------------------------

TEST(CheckpointLimits, HugeLayerIndexInATensorNameIsDropped) {
    std::vector<uint16_t> backing(1024, 0);
    Model model;
    model.config_.arch = ModelArch::LLAMA;
    model.config_.n_layers = 2;
    model.config_.d_model = 8;
    model.layers_.resize(2);

    std::unordered_map<std::string, Tensor> tensors;
    tensors["model.layers.0.self_attn.q_proj.weight"] = fake_weight(backing.data());
    tensors["model.layers.2147483000.self_attn.q_proj.weight"] = fake_weight(backing.data());

    WeightMap wm(ModelArch::LLAMA);
    wm.apply_weights(model, tensors);

    // The hostile name must not have grown the layer vector. Before the fix
    // this was resize(2147483001), i.e. 18.9 TiB.
    EXPECT_LE(model.layers_.size(), static_cast<size_t>(kMaxModelLayers));
}

TEST(CheckpointLimits, HugeExpertIndexInATensorNameIsDropped) {
    std::vector<uint16_t> backing(1024, 0);
    Model model;
    model.config_.arch = ModelArch::MIXTRAL;
    model.config_.n_layers = 1;
    model.config_.n_experts = 2;
    model.config_.d_model = 8;
    model.layers_.resize(1);

    std::unordered_map<std::string, Tensor> tensors;
    tensors["model.layers.0.block_sparse_moe.experts.2000000000.w1.weight"] = fake_weight(backing.data());

    WeightMap wm(ModelArch::MIXTRAL);
    wm.apply_weights(model, tensors);

    ASSERT_EQ(model.layers_.size(), 1u);
    EXPECT_LE(model.layers_[0].expert_w_gate.size(), static_cast<size_t>(kMaxModelExperts));
}

TEST(CheckpointLimits, AnIndexThatFitsIsStillAccepted) {
    std::vector<uint16_t> backing(1024, 0);
    Model model;
    model.config_.arch = ModelArch::LLAMA;
    model.config_.n_layers = 4;
    model.config_.d_model = 8;
    model.layers_.resize(4);

    std::unordered_map<std::string, Tensor> tensors;
    tensors["model.layers.3.self_attn.q_proj.weight"] = fake_weight(backing.data());

    WeightMap wm(ModelArch::LLAMA);
    wm.apply_weights(model, tensors);

    ASSERT_GE(model.layers_.size(), 4u);
    EXPECT_NE(model.layers_[3].wq.data, nullptr);
}

// std::atoi("4294967296") returns 0 on this toolchain, and the old guard only checked `< 0`,
// so tensor name model.layers.4294967296.* wrote into layer 0 over the real weight. The two
// saturating spellings ("2147483648"->INT_MIN, "999...9"->-1) were rejected by that guard
// only by accident.
TEST(CheckpointLimits, AnAliasingLayerIndexDoesNotOverwriteLayerZero) {
    std::vector<uint16_t> backing(1024, 0);
    Model model;
    model.config_.arch = ModelArch::LLAMA;
    model.config_.n_layers = 1;
    model.config_.d_model = 8;
    model.layers_.resize(1);

    // Only the hostile name, not paired with a real layer-0 tensor: pairing would make the
    // assertion depend on unordered_map iteration order (measured: passes with the defect in
    // place about half the time).
    std::unordered_map<std::string, Tensor> tensors;
    tensors["model.layers.4294967296.self_attn.q_proj.weight"] = fake_weight(backing.data());

    WeightMap wm(ModelArch::LLAMA);
    wm.apply_weights(model, tensors);

    ASSERT_EQ(model.layers_.size(), 1u);
    EXPECT_EQ(model.layers_[0].wq.data, nullptr);
}

// `std::atoi` returned an unspecified int here and stopped at the first
// non-digit instead of rejecting it.
TEST(CheckpointLimits, IndexParsingRejectsOverflowAndJunk) {
    EXPECT_EQ(parse_index("0"), 0);
    EXPECT_EQ(parse_index("42"), 42);
    EXPECT_EQ(parse_index(""), -1);
    EXPECT_EQ(parse_index("1x"), -1);
    EXPECT_EQ(parse_index("-1"), -1);
    EXPECT_EQ(parse_index("2147483648"), -1);            // INT_MAX + 1
    EXPECT_EQ(parse_index("99999999999999999999"), -1);  // past uint64 as well
    EXPECT_EQ(parse_index("4294967296"), -1);            // atoi returned 0 for this
}

// ---------------------------------------------------------------------------
// #1611 - a count the checkpoint declares about itself
// ---------------------------------------------------------------------------

TEST(CheckpointLimits, DeclaredDimensionsAreRefusedNotClamped) {
    ModelConfig cfg;
    std::string err;

    cfg.n_layers = 80;
    cfg.n_experts = 128;
    EXPECT_TRUE(validate_declared_dimensions(cfg, &err)) << err;

    cfg.n_layers = 2147483000;  // "num_hidden_layers" in a hostile config.json
    EXPECT_FALSE(validate_declared_dimensions(cfg, &err));
    EXPECT_NE(err.find("layer count"), std::string::npos) << err;

    cfg.n_layers = 80;
    cfg.n_experts = 2000000000;
    EXPECT_FALSE(validate_declared_dimensions(cfg, &err));
    EXPECT_NE(err.find("expert count"), std::string::npos) << err;

    cfg.n_experts = -1;
    EXPECT_FALSE(validate_declared_dimensions(cfg, &err));
}

// ---------------------------------------------------------------------------
// #1612 - shard names out of index.json are paths
// ---------------------------------------------------------------------------

TEST(CheckpointLimits, ShardNameMustBeABareFilename) {
    EXPECT_TRUE(safetensors_shard_name_is_safe("model-00001-of-00008.safetensors"));

    EXPECT_FALSE(safetensors_shard_name_is_safe("../../../../etc/hostname"));
    EXPECT_FALSE(safetensors_shard_name_is_safe("/etc/shadow"));
    EXPECT_FALSE(safetensors_shard_name_is_safe("sub/dir/model.safetensors"));
    EXPECT_FALSE(safetensors_shard_name_is_safe("..\\..\\windows\\system32"));
    EXPECT_FALSE(safetensors_shard_name_is_safe(".."));
    EXPECT_FALSE(safetensors_shard_name_is_safe(""));
}

// EXPECT_EQ(model, nullptr) alone is a blind assertion: this load returns nullptr either way
// since a shard with no usable tensors fails later regardless. The test reads the refusal
// reason from stderr, which only names the containment check if the loader consults it
// before opening the traversed file.
TEST(CheckpointLimits, ATraversingIndexFailsForTheRightReason) {
    LogLevelPin log_pin;
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "imp_shard_traversal_test";
    fs::remove_all(root);
    fs::create_directories(root / "model");

    {
        // 8-byte little-endian header length, then "{}" - a valid, empty file,
        // so open() and mmap() would both succeed on it.
        std::ofstream target(root / "outside.safetensors", std::ios::binary);
        const uint64_t hdr = 2;
        target.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        target << "{}";
    }
    {
        std::ofstream index(root / "model" / "model.safetensors.index.json");
        index << R"({"weight_map": {"model.embed_tokens.weight": "../outside.safetensors"}})";
    }

    testing::internal::CaptureStderr();
    auto model = load_safetensors((root / "model").string());
    const std::string log = testing::internal::GetCapturedStderr();

    EXPECT_EQ(model, nullptr);
    EXPECT_NE(log.find("escapes the model directory"), std::string::npos) << log;

    fs::remove_all(root);
}

// An absolute name is the same defect: `model_dir + "/" + "/etc/shadow"` is
// "//etc/shadow", which resolves normally.
TEST(CheckpointLimits, AnAbsoluteShardNameFailsTheSameWay) {
    LogLevelPin log_pin;
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "imp_shard_absolute_test";
    fs::remove_all(root);
    fs::create_directories(root);

    {
        std::ofstream index(root / "model.safetensors.index.json");
        index << R"({"weight_map": {"model.embed_tokens.weight": "/etc/hostname"}})";
    }

    testing::internal::CaptureStderr();
    auto model = load_safetensors(root.string());
    const std::string log = testing::internal::GetCapturedStderr();

    EXPECT_EQ(model, nullptr);
    EXPECT_NE(log.find("escapes the model directory"), std::string::npos) << log;

    fs::remove_all(root);
}

// The declared-count check at its call site, rather than through the helper:
// a config.json is enough on its own, no hostile tensor name required.
TEST(CheckpointLimits, AConfigDeclaringTwoBillionLayersIsRefused) {
    LogLevelPin log_pin;
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "imp_huge_config_test";
    fs::remove_all(root);
    fs::create_directories(root);

    {
        // One real tensor: an empty file is refused two steps earlier, before
        // the count is used for anything, so it would not exercise the guard.
        const std::string header =
            R"({"model.embed_tokens.weight":{"dtype":"F16","shape":[4,2],"data_offsets":[0,16]}})";
        std::ofstream st(root / "model.safetensors", std::ios::binary);
        const uint64_t hdr = header.size();
        st.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        st << header;
        const std::vector<char> data(16, 0);
        st.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    {
        std::ofstream cfg(root / "config.json");
        cfg << R"({"model_type": "llama", "num_hidden_layers": 2147483000,
                   "hidden_size": 8, "num_attention_heads": 1, "vocab_size": 8})";
    }

    testing::internal::CaptureStderr();
    auto model = load_safetensors(root.string());
    const std::string log = testing::internal::GetCapturedStderr();

    EXPECT_EQ(model, nullptr);
    EXPECT_NE(log.find("exceeds the limit"), std::string::npos) << log;

    fs::remove_all(root);
}

// ---------------------------------------------------------------------------
// #1613 - recursion over attacker-sized input
// ---------------------------------------------------------------------------

TEST(CheckpointLimits, DeepJsonIsRejectedRatherThanOverflowingTheStack) {
    // Well inside the cap: still parses.
    {
        const int d = 100;
        std::string doc(d, '[');
        doc += std::string(d, ']');
        JsonParser p(doc);
        JValue v = p.parse();
        EXPECT_TRUE(p.ok());
        EXPECT_EQ(v.type, JType::ARRAY);
    }
    // Past the cap: an error, and no crash. 50 000 is the depth that was
    // measured to SIGSEGV before this change.
    {
        const int d = 50000;
        std::string doc(d, '[');
        doc += std::string(d, ']');
        JsonParser p(doc);
        (void)p.parse();
        EXPECT_FALSE(p.ok());
    }
    // Objects recurse through the same function.
    {
        const int d = 50000;
        std::string doc;
        for (int i = 0; i < d; i++)
            doc += R"({"a":)";
        doc += "1";
        doc += std::string(d, '}');
        JsonParser p(doc);
        (void)p.parse();
        EXPECT_FALSE(p.ok());
    }
}

TEST(CheckpointLimits, DeepChatTemplateIsRejectedRatherThanOverflowingTheStack) {
    {
        jinja::Template ok;
        EXPECT_TRUE(ok.parse("{{ ((((( x ))))) }}"));
    }
    {
        const int d = 50000;
        std::string src = "{{ " + std::string(d, '(') + "x" + std::string(d, ')') + " }}";
        jinja::Template deep;
        EXPECT_FALSE(deep.parse(src));
    }
    {
        // Statement nesting recurses through parse_node instead.
        const int d = 5000;
        std::string src;
        for (int i = 0; i < d; i++)
            src += "{% if x %}";
        src += "y";
        for (int i = 0; i < d; i++)
            src += "{% endif %}";
        jinja::Template deep;
        EXPECT_FALSE(deep.parse(src));
    }
}

// ---------------------------------------------------------------------------
// AUDIT_arch_2026 F1-9 - the evaluation side of the same template
// ---------------------------------------------------------------------------

TEST(CheckpointLimits, RecursiveMacroIsRefusedRatherThanOverflowingTheStack) {
    LogLevelPin log_pin;
    jinja::Template t;
    ASSERT_TRUE(t.parse("{% macro f() %}{{ f() }}{% endmacro %}{{ f() }}"));
    jinja::Context ctx;
    EXPECT_EQ(t.render(ctx), "");
    EXPECT_NE(t.error().find("call depth"), std::string::npos) << t.error();

    // A macro chain of ordinary depth still renders.
    jinja::Template ok;
    ASSERT_TRUE(ok.parse(
        "{% macro g(x) %}<{{ x }}>{% endmacro %}{% macro f(x) %}{{ g(x) }}{% endmacro %}{{ f(1) }}"));
    EXPECT_EQ(ok.render(ctx), "<1>") << ok.error();
}

TEST(CheckpointLimits, HugeLoopsAreRefusedRatherThanExhaustingMemory) {
    LogLevelPin log_pin;
    jinja::Context ctx;
    {
        jinja::Template t;
        ASSERT_TRUE(t.parse("{% for i in range(10000000000) %}x{% endfor %}"));
        EXPECT_EQ(t.render(ctx), "");
        EXPECT_NE(t.error().find("range length"), std::string::npos) << t.error();
    }
    {
        // The budget is per render, so nesting cannot dodge the range cap.
        jinja::Template t;
        ASSERT_TRUE(t.parse("{% for i in range(2000) %}{% for j in range(2000) %}x{% endfor %}{% endfor %}"));
        EXPECT_EQ(t.render(ctx), "");
        EXPECT_NE(t.error().find("loop iterations"), std::string::npos) << t.error();
    }
    {
        jinja::Template t;
        ASSERT_TRUE(t.parse(
            "{% for i in range(3) %}{{ i }}{% endfor %}{% for i in range(5, 1, -2) %}{{ i }}{% endfor %}"));
        EXPECT_EQ(t.render(ctx), "01253") << t.error();
        EXPECT_TRUE(t.error().empty());
    }
}

// ---------------------------------------------------------------------------
// AUDIT_arch_2026 F1-10 - the shared JSON parser and the config.json dims
// ---------------------------------------------------------------------------

TEST(CheckpointLimits, MalformedJsonNumbersAreParseErrorsNotExceptions) {
    // ("1." is accepted: from_chars takes the C grammar, the same as before.)
    for (const char* doc : {R"({"a": -})", R"({"a": 1e999})", "[-]", "[1e]"}) {
        JsonParser p(doc);
        EXPECT_NO_THROW((void)p.parse()) << doc;
        EXPECT_FALSE(p.ok()) << doc;
    }
    JsonParser fine(R"({"a": -1.5e3, "b": 42})");
    JValue v = fine.parse();
    ASSERT_TRUE(fine.ok());
    EXPECT_DOUBLE_EQ(jobj_find(v, "a")->num_val, -1500.0);
    EXPECT_EQ(jobj_find(v, "b")->as_int(), 42);
}

TEST(CheckpointLimits, JsonNodeBudgetRefusesATreeThatWouldNotFit) {
    // 200 nodes under a 100-node budget: refused. The production budget is
    // 8M; the constructor seam keeps the test at microseconds.
    std::string doc = "[";
    for (int i = 0; i < 199; i++)
        doc += "0,";
    doc += "0]";
    {
        JsonParser p(doc, 100);
        (void)p.parse();
        EXPECT_FALSE(p.ok());
    }
    {
        JsonParser p(doc, 1000);
        JValue v = p.parse();
        EXPECT_TRUE(p.ok());
        EXPECT_EQ(v.arr.size(), 200u);
    }
}

TEST(CheckpointLimits, DeclaredPerDimensionCountsAreRefusedNotSized) {
    ModelConfig cfg;
    std::string err;
    cfg.n_layers = 126;
    cfg.n_heads = 128;
    cfg.n_kv_heads = 8;
    cfg.head_dim = 512;
    cfg.d_model = 16384;
    cfg.d_ff = 53248;
    cfg.vocab_size = 262144;
    EXPECT_TRUE(validate_declared_dimensions(cfg, &err)) << err;

    cfg.head_dim = 2147483647;  // two 8.6 GiB rope-factor tables before the fix
    EXPECT_FALSE(validate_declared_dimensions(cfg, &err));
    EXPECT_NE(err.find("head_dim"), std::string::npos) << err;

    cfg.head_dim = 128;
    cfg.vocab_size = -5;
    EXPECT_FALSE(validate_declared_dimensions(cfg, &err));
    EXPECT_NE(err.find("vocab size"), std::string::npos) << err;

    cfg.vocab_size = 32000;
    cfg.d_ff = 1 << 21;
    EXPECT_FALSE(validate_declared_dimensions(cfg, &err));
    EXPECT_NE(err.find("intermediate size"), std::string::npos) << err;
}

TEST(CheckpointLimits, AConfigDeclaringAGiantHeadDimIsRefused) {
    LogLevelPin log_pin;
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "imp_huge_head_dim_test";
    fs::remove_all(root);
    fs::create_directories(root);

    {
        const std::string header =
            R"({"model.embed_tokens.weight":{"dtype":"F16","shape":[4,2],"data_offsets":[0,16]}})";
        std::ofstream st(root / "model.safetensors", std::ios::binary);
        const uint64_t hdr = header.size();
        st.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        st << header;
        const std::vector<char> data(16, 0);
        st.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    {
        // The Llama-3 rope_scaling branch sizes two vectors from head_dim / 2.
        std::ofstream cfg(root / "config.json");
        cfg << R"({"model_type": "llama", "num_hidden_layers": 1, "hidden_size": 8,
                   "num_attention_heads": 1, "vocab_size": 8, "head_dim": 2147483647,
                   "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                    "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}})";
    }

    testing::internal::CaptureStderr();
    auto model = load_safetensors(root.string());
    const std::string log = testing::internal::GetCapturedStderr();

    EXPECT_EQ(model, nullptr);
    EXPECT_NE(log.find("head_dim"), std::string::npos) << log;
    EXPECT_NE(log.find("exceeds the limit"), std::string::npos) << log;

    fs::remove_all(root);
}

}  // namespace
}  // namespace imp
