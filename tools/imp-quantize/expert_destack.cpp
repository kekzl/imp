#include "expert_destack.h"

#include "core/fp_bits.h"
#include "model/json_util.h"

#include <cstring>

namespace imp::quantize {
namespace {

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

const char* part_name(ExpertPart p) {
    switch (p) {
        case ExpertPart::Gate:
            return "gate_proj";
        case ExpertPart::Up:
            return "up_proj";
        case ExpertPart::Down:
            return "down_proj";
    }
    return "down_proj";
}

}  // namespace

// One row per model family that ships stacked experts. Shapes verified against the BF16
// sources: gpt-oss-20b gate_up [32, 2880, 5760] = [ne, hidden, 2 * d_ff] (transposed,
// interleaved), Gemma-4-26B-A4B gate_up [128, 1408, 2816] = [ne, 2 * d_ff, hidden].
std::optional<StackedExpertLayout> stacked_expert_layout(const std::string& model_type) {
    if (model_type == "gpt_oss")
        return StackedExpertLayout{/*transposed=*/true, GateUpOrder::Interleaved};
    if (model_type == "gemma4" || model_type == "gemma4_text")
        return StackedExpertLayout{/*transposed=*/false, GateUpOrder::Concatenated};
    return std::nullopt;
}

std::string model_type_from_config(const std::string& config_json_path) {
    JValue cfg;
    if (!parse_json_file(config_json_path, cfg) || cfg.type != JType::OBJECT)
        return "";
    std::string out;
    if (jobj_get_string(cfg, "model_type", out) && !out.empty())
        return out;
    if (const JValue* text = jobj_find(cfg, "text_config"); text && text->type == JType::OBJECT) {
        if (jobj_get_string(*text, "model_type", out))
            return out;
    }
    return "";
}

std::expected<std::vector<DestackedMatrix>, std::string> destack_plan(const RawTensor& stack,
                                                                      const StackedExpertLayout& layout) {
    static const std::string kGateUp = "experts.gate_up_proj";
    static const std::string kDown = "experts.down_proj";
    const bool gate_up = ends_with(stack.name, kGateUp);
    if (!gate_up && !ends_with(stack.name, kDown))
        return std::unexpected(
            "not an expert stack this tool splits (expects ...experts.gate_up_proj or "
            "...experts.down_proj)");
    if (stack.dtype != "BF16" && stack.dtype != "F16")
        return std::unexpected("dtype " + stack.dtype + " (only BF16/F16 stacks are split)");
    if (stack.shape.size() != 3)
        return std::unexpected(std::to_string(stack.shape.size()) + "-D, expected [ne, rows, cols]");
    const int64_t ne = stack.shape[0];
    // R = the fused/output row count, K = the reduction dim, whichever axis holds them.
    const int64_t R = layout.transposed ? stack.shape[2] : stack.shape[1];
    const int64_t K = layout.transposed ? stack.shape[1] : stack.shape[2];
    if (ne <= 0 || R <= 0 || K <= 0)
        return std::unexpected("empty stack");
    if (K % 16 != 0)
        return std::unexpected("K=" + std::to_string(K) + " is not a multiple of the 16-value micro-block");
    if (gate_up && (R % 2) != 0)
        return std::unexpected("fused gate_up row count " + std::to_string(R) + " is odd");

    // `<prefix>.experts` + `.<e>.<proj>.weight`: the HF per-expert spelling under the same parent.
    const std::string prefix = stack.name.substr(0, stack.name.size() -
                                                        (gate_up ? kGateUp.size() : kDown.size())) +
                               "experts.";
    std::vector<DestackedMatrix> out;
    out.reserve(static_cast<size_t>(ne) * (gate_up ? 2 : 1));
    for (int64_t e = 0; e < ne; ++e) {
        if (gate_up) {
            for (ExpertPart p : {ExpertPart::Gate, ExpertPart::Up})
                out.push_back({prefix + std::to_string(e) + "." + part_name(p) + ".weight", e, p, R / 2, K});
        } else {
            out.push_back({prefix + std::to_string(e) + ".down_proj.weight", e, ExpertPart::Down, R, K});
        }
    }
    return out;
}

std::vector<uint16_t> destack_read(const RawTensor& stack, const StackedExpertLayout& layout,
                                   const DestackedMatrix& m) {
    const int64_t R = layout.transposed ? stack.shape[2] : stack.shape[1];
    const int64_t K = layout.transposed ? stack.shape[1] : stack.shape[2];
    const auto* src = static_cast<const uint16_t*>(stack.data) + m.expert * R * K;
    const bool f16 = stack.dtype == "F16";
    std::vector<uint16_t> out(static_cast<size_t>(m.N) * static_cast<size_t>(m.K));
    for (int64_t n = 0; n < m.N; ++n) {
        // Which fused row holds output channel n of this part.
        int64_t r = n;
        if (m.part != ExpertPart::Down) {
            const bool up = m.part == ExpertPart::Up;
            r = layout.gate_up == GateUpOrder::Interleaved ? 2 * n + (up ? 1 : 0) : n + (up ? m.N : 0);
        }
        uint16_t* dst = out.data() + n * m.K;
        for (int64_t k = 0; k < m.K; ++k) {
            const uint16_t v = layout.transposed ? src[k * R + r] : src[r * K + k];
            dst[k] = f16 ? v : float_to_half(bf16_to_float(v));
        }
    }
    return out;
}

}  // namespace imp::quantize
