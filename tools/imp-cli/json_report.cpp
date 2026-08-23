#include "json_report.h"

#include "common/json_out.h"

namespace imp_cli {

void emit_bench(const BenchReport& r) {
    // prefill_tps / decode_tps sit at the top level, flat: those are the two
    // numbers scripts/gen_perf_baseline.sh used to regex out of the stderr
    // table, and a nested shape would have kept it regexing.
    imp_tools::JsonOut j;
    j.str("mode", "bench")
        .str("model", r.model)
        .num("prefill_tps", r.prefill_tps, 2)
        .num("decode_tps", r.decode_tps, 2)
        .intg("pp_tokens", r.pp_tokens)
        .num("pp_ms", r.pp_ms, 2)
        .intg("tg_tokens", r.tg_tokens)
        .num("tg_ms", r.tg_ms, 2)
        .intg("reps", r.reps)
        .intg("peak_vram_mib", r.peak_vram_mib);
    imp_tools::json_emit(j.str());
}

void emit_generate(const GenerateReport& r) {
    imp_tools::JsonOut j;
    j.str("mode", "generate")
        .str("model", r.model)
        .str("text", r.text)
        .intg("prompt_tokens", r.prompt_tokens)
        .intg("completion_tokens", r.completion_tokens)
        .num("prefill_tps", r.prefill_tps, 2)
        .num("decode_tps", r.decode_tps, 2)
        .num("prefill_ms", r.prefill_ms, 2)
        .num("decode_ms", r.decode_ms, 2)
        .num("total_ms", r.total_ms, 2);
    imp_tools::json_emit(j.str());
}

void emit_perplexity(const std::string& model, double ppl, long long tokens, const std::string& corpus,
                     const std::string& calibration) {
    imp_tools::JsonOut j;
    j.str("mode", "perplexity")
        .str("model", model)
        .num("perplexity", ppl)
        .intg("tokens", tokens)
        .str("corpus", corpus);
    if (!calibration.empty())
        j.str("calibration", calibration);
    imp_tools::json_emit(j.str());
}

}  // namespace imp_cli
