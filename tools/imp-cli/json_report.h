#pragma once

// The --json documents imp-cli emits (#1583). Separate from main.cpp (which sits at the
// 800-line hard-review threshold). Key names here are a contract scripts/gen_perf_baseline.sh,
// scripts/verify.sh and scripts/bench_gate.sh read; each writer uses imp_tools::json_emit(),
// a no-op unless json_stdout_reserve() ran.

#include <string>

namespace imp_cli {

struct BenchReport {
    std::string model;
    double prefill_tps = 0;
    double decode_tps = 0;
    int pp_tokens = 0;
    double pp_ms = 0;
    int tg_tokens = 0;
    double tg_ms = 0;
    int reps = 0;
    long long peak_vram_mib = 0;
};

struct GenerateReport {
    std::string model;
    std::string text;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double prefill_tps = 0;
    double decode_tps = 0;
    double prefill_ms = 0;
    double decode_ms = 0;
    double total_ms = 0;
};

void emit_bench(const BenchReport& r);
void emit_generate(const GenerateReport& r);
void emit_perplexity(const std::string& model, double ppl, long long tokens, const std::string& corpus,
                     const std::string& calibration);

}  // namespace imp_cli
