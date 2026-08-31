#include <gtest/gtest.h>
#include "runtime/config.h"
#include "model/model_arch.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace imp {
namespace {

// Helper to write a temporary config file and clean it up.
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& body) {
        path = "/tmp/imp_test_config_" + std::to_string(::getpid()) + ".conf";
        std::ofstream ofs(path);
        ofs << body;
    }
    ~TempFile() { std::remove(path.c_str()); }
};

// Apply overrides and fail if any of them bound to nothing — the contract the
// tool mains enforce. Going through this helper means a typo in a test is a red
// test rather than an assertion that quietly stops testing anything.
void set_(RuntimeConfig& cfg, const std::vector<std::string>& kvs) {
    const std::vector<std::string> rejected = cfg.apply_overrides(kvs);
    ASSERT_TRUE(rejected.empty()) << "unbound override: " << rejected.front();
}

TEST(RuntimeConfigTest, DefaultsAreSane) {
    RuntimeConfig cfg;
    EXPECT_FALSE(cfg.runtime.deterministic_gemm);
    EXPECT_EQ(cfg.runtime.cuda_graphs, "auto");
    EXPECT_TRUE(cfg.runtime.warmup);  // default ON: greedy request-order independence (docs/determinism.md)
    EXPECT_EQ(cfg.kv_cache.dtype, "auto");
    EXPECT_EQ(cfg.kv_cache.swa_sizing, "auto");
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Auto);
    EXPECT_EQ(cfg.moe.expert_overhead_pct, 10);
    EXPECT_FALSE(cfg.gdn.fp32_scan);
    EXPECT_EQ(cfg.diagnostics.exit_layer, -1);
}

// kv_cache.swa_sizing tri-state: "auto"/"on"/"off" plus legacy bool literals
// (the key was a bool until 2026-07-24 — existing imp.conf files keep parsing).
TEST(RuntimeConfigTest, SwaSizingTriState) {
    RuntimeConfig cfg;
    set_(cfg, {"kv_cache.swa_sizing=on"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::On);
    set_(cfg, {"kv_cache.swa_sizing=off"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Off);
    set_(cfg, {"kv_cache.swa_sizing=auto"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Auto);
    // Legacy bool spellings map to On/Off.
    set_(cfg, {"kv_cache.swa_sizing=true"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::On);
    set_(cfg, {"kv_cache.swa_sizing=false"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Off);
    // Unknown value falls back to Off (never silently forces sizing on).
    set_(cfg, {"kv_cache.swa_sizing=banana"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Off);
    // SWA snapshot budget: off by default, plain int binder.
    EXPECT_EQ(RuntimeConfig{}.kv_cache.swa_snapshot_mb, 0);
    set_(cfg, {"kv_cache.swa_snapshot_mb=512"});
    EXPECT_EQ(cfg.kv_cache.swa_snapshot_mb, 512);
}

// The long-context FP8-KV quality gate (kv_cache.dtype=auto): only arch families
// empirically verified safe honor the model author's kv_cache_quant_algo=FP8 hint
// by default. Keep this allowlist conservative — see model.cpp for the evidence.
TEST(RuntimeConfigTest, KvFp8HintDefaultSafeAllowlist) {
    EXPECT_TRUE(kv_fp8_hint_default_safe(ModelArch::QWEN3));
    EXPECT_TRUE(kv_fp8_hint_default_safe(ModelArch::QWEN3_MOE));
    // LLAMA: verified via Phi-4-reasoning-plus-NVFP4 (+0.25% PPL, PR #749).
    EXPECT_TRUE(kv_fp8_hint_default_safe(ModelArch::LLAMA));
    // NEMOTRON_H_MOE: verified via Nemotron-3-Nano-30B — FP8 vs FP16 mean PPL
    // ~0.00% over a 26.5k-token corpus, 5 runs each (the 2026-06-23 single A/B
    // was run-to-run noise on this MoE+Mamba2 hybrid).
    EXPECT_TRUE(kv_fp8_hint_default_safe(ModelArch::NEMOTRON_H_MOE));
    // Not yet verified on this box → must stay FP16 by default. GEMMA4 baseline
    // PPL broken on the gate corpus; QWEN36_MOE quality fine but no hint declared.
    EXPECT_FALSE(kv_fp8_hint_default_safe(ModelArch::GEMMA4));
    EXPECT_FALSE(kv_fp8_hint_default_safe(ModelArch::QWEN35));
    EXPECT_FALSE(kv_fp8_hint_default_safe(ModelArch::QWEN36_MOE));
    EXPECT_FALSE(kv_fp8_hint_default_safe(ModelArch::GENERIC));
}

// The NVFP4-KV capacity gate (kv_cache.dtype=auto). Unlike the two FP8 lists,
// this one is not a ~neutral-quality bar: it is a deliberate trade of ~0.3 % PPL
// for 2.7x the context, because on a GDN hybrid the KV cache is what bounds
// max_seq_len (only the attention layers hold one). Measured 2026-08-24 on
// Qwen3.8-27B-NVFP4 (+0.29..0.35 %) and Qwen3.5-4B mxfp4 (+0.15..0.18 %),
// alternating arms; see model.cpp.
TEST(RuntimeConfigTest, KvNvfp4DefaultSafeAllowlist) {
    EXPECT_TRUE(kv_nvfp4_default_safe(ModelArch::QWEN35));
    // The MoE siblings are deliberately OFF: FP8 KV already costs QWEN36_MOE
    // +1.47 % PPL because NVFP4 attention weights compound with a quantised KV,
    // and NVFP4 KV is the more aggressive quantiser. Unmeasured there.
    EXPECT_FALSE(kv_nvfp4_default_safe(ModelArch::QWEN36_MOE));
    EXPECT_FALSE(kv_nvfp4_default_safe(ModelArch::QWEN35_MOE));
    // Families the FP8 lists already serve must not be flipped by this gate.
    EXPECT_FALSE(kv_nvfp4_default_safe(ModelArch::QWEN3));
    EXPECT_FALSE(kv_nvfp4_default_safe(ModelArch::QWEN3_MOE));
    EXPECT_FALSE(kv_nvfp4_default_safe(ModelArch::LLAMA));
    EXPECT_FALSE(kv_nvfp4_default_safe(ModelArch::NEMOTRON_H_MOE));
    // GPT_OSS carries learned attention sinks; a quantised KV drops the sink
    // term, and the resolver's sink fallback is the backstop — this gate must
    // not walk into it in the first place.
    EXPECT_FALSE(kv_nvfp4_default_safe(ModelArch::GPT_OSS));
    EXPECT_FALSE(kv_nvfp4_default_safe(ModelArch::GEMMA4));
    EXPECT_FALSE(kv_nvfp4_default_safe(ModelArch::GENERIC));
}

// What an EXPLICIT dtype pin costs against the auto default. The pin that
// motivates this is `IMP_KV_FP8=1`: correct when `auto` meant FP16, and since
// the NVFP4 default it doubles the bytes per token on QWEN35 (max_model_len
// 90 528 instead of 126 432) while logging nothing, because --kv-fp8 sets the
// enum directly and skips the resolver branch that would have said so.
TEST(RuntimeConfigTest, KvPinContextCostFactor) {
    // On the NVFP4-default family, a wider pin is a context forfeit.
    EXPECT_EQ(2, kv_pin_context_cost_factor(ModelArch::QWEN35, QType::FP8_E4M3));
    EXPECT_EQ(2, kv_pin_context_cost_factor(ModelArch::QWEN35, QType::INT8));
    EXPECT_EQ(4, kv_pin_context_cost_factor(ModelArch::QWEN35, QType::F16));
    // Same width or narrower costs nothing - pinning the default is not a warning.
    EXPECT_EQ(0, kv_pin_context_cost_factor(ModelArch::QWEN35, QType::NVFP4));
    EXPECT_EQ(0, kv_pin_context_cost_factor(ModelArch::QWEN35, QType::MXFP4_KV));
    EXPECT_EQ(0, kv_pin_context_cost_factor(ModelArch::QWEN35, QType::INT4));
    // Everywhere else `auto` already lands on FP8 or FP16, so the same pin is
    // not a downgrade and must stay silent. This is the half that keeps the
    // warning from firing on every deployment that ever passed --kv-fp8.
    EXPECT_EQ(0, kv_pin_context_cost_factor(ModelArch::QWEN3, QType::FP8_E4M3));
    EXPECT_EQ(0, kv_pin_context_cost_factor(ModelArch::QWEN35_MOE, QType::FP8_E4M3));
    EXPECT_EQ(0, kv_pin_context_cost_factor(ModelArch::LLAMA, QType::F16));
    EXPECT_EQ(0, kv_pin_context_cost_factor(ModelArch::GENERIC, QType::F16));
    // Tied to the allowlist rather than to a second copy of it: whatever
    // kv_nvfp4_default_safe says is what can lose context.
    for (auto arch : {ModelArch::QWEN35, ModelArch::QWEN3, ModelArch::LLAMA, ModelArch::GEMMA4,
                      ModelArch::GPT_OSS, ModelArch::NEMOTRON_H_MOE}) {
        EXPECT_EQ(kv_nvfp4_default_safe(arch), kv_pin_context_cost_factor(arch, QType::FP8_E4M3) > 0)
            << "arch " << model_arch_name(arch);
    }
}

// The other half of the pin warning: did the operator choose this dtype at all.
// Split out of the resolver so both halves of the decision are decidable without
// a GPU - what is left at the call site is the log line.
TEST(RuntimeConfigTest, KvDtypeIsExplicitPin) {
    // CLI surface: --kv-fp8 and friends set the enum before the resolver runs,
    // which is exactly why they log nothing on their own.
    EXPECT_TRUE(kv_dtype_is_explicit_pin(QType::FP8_E4M3, "auto"));
    EXPECT_TRUE(kv_dtype_is_explicit_pin(QType::NVFP4, "auto"));
    // Config surface: every value the resolver accepts as a choice.
    EXPECT_TRUE(kv_dtype_is_explicit_pin(QType::F16, "fp8"));
    EXPECT_TRUE(kv_dtype_is_explicit_pin(QType::F16, "int8"));
    EXPECT_TRUE(kv_dtype_is_explicit_pin(QType::F16, "fp16"));
    EXPECT_TRUE(kv_dtype_is_explicit_pin(QType::F16, "nvfp4"));
    // Not a choice: the default, and a typo (which the resolver already warns
    // about separately - one bad value must not also claim to be a pin).
    EXPECT_FALSE(kv_dtype_is_explicit_pin(QType::F16, "auto"));
    EXPECT_FALSE(kv_dtype_is_explicit_pin(QType::F16, ""));
    EXPECT_FALSE(kv_dtype_is_explicit_pin(QType::F16, "mxfp4_kv"));
    EXPECT_FALSE(kv_dtype_is_explicit_pin(QType::F16, "FP8"));
}

// The NO-HINT FP8-KV gate (kv_cache.dtype=auto on checkpoints without a
// kv_cache_quant_algo hint — i.e. every GGUF). Stricter bar than the hint list:
// the family must gate ~PPL-neutral at 16k context. See model.cpp for the
// 2026-07-12 evidence and the measured exclusions (QWEN36_MOE: +1.47% real on
// the NVFP4 variant; LLAMA: gate-corpus baseline broken).
TEST(RuntimeConfigTest, KvFp8NoHintDefaultSafeAllowlist) {
    EXPECT_TRUE(kv_fp8_no_hint_default_safe(ModelArch::QWEN3));
    EXPECT_TRUE(kv_fp8_no_hint_default_safe(ModelArch::QWEN3_MOE));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::LLAMA));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::QWEN36_MOE));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::NEMOTRON_H_MOE));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::GEMMA4));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::GENERIC));
}

TEST(RuntimeConfigTest, ParsesBasicSections) {
    TempFile f(R"(
[runtime]
deterministic_gemm = true
cuda_graphs = "never"

[kv_cache]
dtype = "fp8"

[moe]
expert_overhead_pct = 30
)");

    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.load_from_file(f.path));
    EXPECT_TRUE(cfg.runtime.deterministic_gemm);
    EXPECT_EQ(cfg.runtime.cuda_graphs, "never");
    EXPECT_EQ(cfg.kv_cache.dtype, "fp8");
    EXPECT_EQ(cfg.moe.expert_overhead_pct, 30);
}

TEST(RuntimeConfigTest, ParsesRopeAndVramSections) {
    // Defaults: override off, planner at the historical envelope.
    RuntimeConfig defaults;
    EXPECT_TRUE(defaults.rope.scaling.empty());
    EXPECT_FLOAT_EQ(defaults.rope.factor, 1.0f);
    EXPECT_EQ(defaults.rope.orig_ctx, 0);
    EXPECT_FLOAT_EQ(defaults.rope.attn_factor, 1.0f);
    EXPECT_FLOAT_EQ(defaults.vram.kv_fraction, 0.8f);
    EXPECT_EQ(defaults.vram.reserve_floor_pct, 10);
    // #1653: 4x4 MiB, not the 4x128 that shipped until the sweep. A default
    // that drifts back silently costs 0.68 s of every process start.
    EXPECT_EQ(defaults.vram.upload_ring_depth, 4);
    EXPECT_EQ(defaults.vram.upload_ring_chunk_mib, 4);

    TempFile f(R"(
[rope]
scaling = "yarn"
factor = 4.0
orig_ctx = 32768
attn_factor = 1.5
beta_fast = 24
beta_slow = 2

[vram]
kv_fraction = 0.5
reserve_floor_pct = 5
)");

    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.load_from_file(f.path));
    EXPECT_EQ(cfg.rope.scaling, "yarn");
    EXPECT_FLOAT_EQ(cfg.rope.factor, 4.0f);
    EXPECT_EQ(cfg.rope.orig_ctx, 32768);
    EXPECT_FLOAT_EQ(cfg.rope.attn_factor, 1.5f);
    EXPECT_FLOAT_EQ(cfg.rope.beta_fast, 24.0f);
    EXPECT_FLOAT_EQ(cfg.rope.beta_slow, 2.0f);
    EXPECT_FLOAT_EQ(cfg.vram.kv_fraction, 0.5f);
    EXPECT_EQ(cfg.vram.reserve_floor_pct, 5);

    // --set style overrides reach the same fields.
    set_(cfg, {"vram.upload_ring_depth=2", "vram.upload_ring_chunk_mib=64"});
    EXPECT_EQ(cfg.vram.upload_ring_depth, 2);
    EXPECT_EQ(cfg.vram.upload_ring_chunk_mib, 64);

    set_(cfg, {"rope.scaling=linear", "rope.factor=2", "vram.kv_fraction=0.9"});
    EXPECT_EQ(cfg.rope.scaling, "linear");
    EXPECT_FLOAT_EQ(cfg.rope.factor, 2.0f);
    EXPECT_FLOAT_EQ(cfg.vram.kv_fraction, 0.9f);
}

TEST(RuntimeConfigTest, IgnoresCommentsAndBlankLines) {
    TempFile f(R"(
# top comment
[runtime]
# inline comment

deterministic_gemm = true   # trailing comment
warmup              = false
)");

    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.load_from_file(f.path));
    EXPECT_TRUE(cfg.runtime.deterministic_gemm);
    EXPECT_FALSE(cfg.runtime.warmup);
}

TEST(RuntimeConfigTest, ApplyOverrides) {
    RuntimeConfig cfg;
    set_(cfg, {
        "kv_cache.dtype=fp8",
        "runtime.cuda_graphs=never",
        "moe.expert_overhead_pct=30",
        "gdn.fp32_scan=true",
    });
    EXPECT_EQ(cfg.kv_cache.dtype, "fp8");
    EXPECT_EQ(cfg.runtime.cuda_graphs, "never");
    EXPECT_EQ(cfg.moe.expert_overhead_pct, 30);
    EXPECT_TRUE(cfg.gdn.fp32_scan);
}

TEST(RuntimeConfigTest, OverrideWinsOverFile) {
    TempFile f(R"(
[kv_cache]
dtype = "fp16"
)");

    RuntimeConfig cfg;
    cfg.load_from_file(f.path);
    EXPECT_EQ(cfg.kv_cache.dtype, "fp16");
    set_(cfg, {"kv_cache.dtype=fp8"});
    EXPECT_EQ(cfg.kv_cache.dtype, "fp8");
}

TEST(RuntimeConfigTest, BoolParsingIsLenient) {
    RuntimeConfig cfg;
    set_(cfg, {"runtime.warmup=false"});
    EXPECT_FALSE(cfg.runtime.warmup);
    set_(cfg, {"runtime.warmup=on"});
    EXPECT_TRUE(cfg.runtime.warmup);
    set_(cfg, {"runtime.warmup=0"});
    EXPECT_FALSE(cfg.runtime.warmup);
    set_(cfg, {"runtime.warmup=1"});
    EXPECT_TRUE(cfg.runtime.warmup);
}

TEST(RuntimeConfigTest, UnknownOverrideIsReportedNotSwallowed) {
    // An override that binds to nothing comes back named, so `--set` can refuse
    // instead of running a measurement the flag never configured. This was not
    // hypothetical: `--set gemm.deterministic=true` (the key is
    // runtime.deterministic_gemm) sat in the AWQ reproduction harness and did
    // nothing at all.
    RuntimeConfig cfg;
    const std::vector<std::string> rejected =
        cfg.apply_overrides({"runtime.does_not_exist=42", "runtime.warmup=false", "gemm.deterministic=true"});
    ASSERT_EQ(rejected.size(), 2u);
    EXPECT_NE(rejected[0].find("runtime.does_not_exist=42"), std::string::npos);
    EXPECT_NE(rejected[1].find("gemm.deterministic=true"), std::string::npos);
    // The good key in the same batch still applied, and no field moved for the
    // bad ones.
    EXPECT_FALSE(cfg.runtime.warmup);
    EXPECT_FALSE(cfg.runtime.deterministic_gemm);
}

TEST(RuntimeConfigTest, MalformedOverrideIsReported) {
    RuntimeConfig cfg;
    const std::vector<std::string> rejected = cfg.apply_overrides({"runtime.warmup"});
    ASSERT_EQ(rejected.size(), 1u);
    EXPECT_NE(rejected[0].find("expected key=value"), std::string::npos);
}

TEST(RuntimeConfigTest, UnknownKeyInAFileStaysAWarning) {
    // The other half of the contract: a config file may outlive the build that
    // understood every key in it, so the file path must not become fatal.
    TempFile tf("[runtime]\ndoes_not_exist = 42\nwarmup = false\n");
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.load_from_file(tf.path));
    EXPECT_FALSE(cfg.runtime.warmup);
}

TEST(RuntimeConfigTest, MissingFileFallsBackToDefaults) {
    RuntimeConfig cfg;
    EXPECT_FALSE(cfg.load_from_file("/nonexistent/path/imp.conf"));
    // Defaults still in place.
    EXPECT_EQ(cfg.kv_cache.dtype, "auto");
}

}  // namespace
}  // namespace imp

// ---- #1627: an unreadable VALUE is rejected, not silently dropped ----
//
// `--set` refused an unknown KEY and accepted anything as a value: the three
// parsers returned the current value for input they could not read, with no
// warning, so `--set server.prefix_cache=disabled` kept the default and said
// nothing. 157 of the 185 bound keys go through those three.

TEST(ConfigBadValue, BoolKeyRejectsANonBoolean) {
    imp::RuntimeConfig cfg;
    const bool before = cfg.speculative.capture;
    auto rejected = cfg.apply_overrides({"speculative.capture=disabled"});
    ASSERT_EQ(rejected.size(), 1u) << "an unreadable value must be reported";
    EXPECT_NE(rejected[0].find("value not readable"), std::string::npos) << rejected[0];
    EXPECT_EQ(cfg.speculative.capture, before) << "and the default must be kept";
}

TEST(ConfigBadValue, IntKeyRejectsWordsAndTrailingGarbage) {
    imp::RuntimeConfig cfg;
    EXPECT_EQ(cfg.apply_overrides({"speculative.k=one"}).size(), 1u);
    // stoi stops at the first non-digit, so this parsed as 16 and reported
    // nothing. The whole string has to be consumed.
    EXPECT_EQ(cfg.apply_overrides({"speculative.k=16k"}).size(), 1u);
    EXPECT_EQ(cfg.apply_overrides({"speculative.k=1,2"}).size(), 1u);
}

TEST(ConfigBadValue, FloatKeyRejectsANonNumber) {
    imp::RuntimeConfig cfg;
    EXPECT_EQ(cfg.apply_overrides({"speculative.mtp_econ_min_emit=high"}).size(), 1u);
}

TEST(ConfigBadValue, EveryAcceptedBooleanSpellingStillWorks) {
    // Negative control. imp.conf.example uses table values on all 107 boolean
    // lines, so a stricter parser must not start rejecting them.
    for (const char* v : {"true", "True", "1", "yes", "on"}) {
        imp::RuntimeConfig cfg;
        EXPECT_TRUE(cfg.apply_overrides({std::string("speculative.capture=") + v}).empty()) << v;
        EXPECT_TRUE(cfg.speculative.capture) << v;
    }
    for (const char* v : {"false", "False", "0", "no", "off"}) {
        imp::RuntimeConfig cfg;
        EXPECT_TRUE(cfg.apply_overrides({std::string("speculative.capture=") + v}).empty()) << v;
        EXPECT_FALSE(cfg.speculative.capture) << v;
    }
}

TEST(ConfigBadValue, TriStateKeyKeepsItsThreeSpellings) {
    for (const char* v : {"auto", "on", "off"}) {
        imp::RuntimeConfig cfg;
        EXPECT_TRUE(cfg.apply_overrides({std::string("gemm.cublas_fp16_acc=") + v}).empty()) << v;
        EXPECT_EQ(cfg.gemm.cublas_fp16_acc, v) << v;
    }
    imp::RuntimeConfig cfg;
    EXPECT_EQ(cfg.apply_overrides({"gemm.cublas_fp16_acc=sometimes"}).size(), 1u);
}

// ---- #1638: a decode-path switch nothing could set ----

TEST(ConfigBinding, SpeculativeBatchRrIsBound) {
    imp::RuntimeConfig cfg;
    ASSERT_TRUE(cfg.speculative.batch_rr) << "default is on";
    EXPECT_TRUE(cfg.apply_overrides({"speculative.batch_rr=false"}).empty())
        << "read at engine_scheduler.cpp:1422 and :2882, bound to no key until #1638";
    EXPECT_FALSE(cfg.speculative.batch_rr);
}

// #1645: the knob that sets the prefill TTFT/ITL trade had no imp.conf key,
// while the knob that merely caps it did.
TEST(ConfigBinding, PrefillChunkSizeIsBound) {
    imp::RuntimeConfig cfg;
    EXPECT_EQ(cfg.runtime.prefill_chunk_size, -1) << "default is the per-arch resolver";
    EXPECT_TRUE(cfg.apply_overrides({"runtime.prefill_chunk_size=512"}).empty());
    EXPECT_EQ(cfg.runtime.prefill_chunk_size, 512);
    EXPECT_TRUE(cfg.apply_overrides({"runtime.prefill_chunk_size=0"}).empty());
    EXPECT_EQ(cfg.runtime.prefill_chunk_size, 0);
    // And it is still a number, not anything (#1627).
    EXPECT_EQ(cfg.apply_overrides({"runtime.prefill_chunk_size=large"}).size(), 1u);
}
