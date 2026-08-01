#include "tensor_policy.h"

namespace imp::quantize {
namespace {

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool contains(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }

bool is_float_dtype(const std::string& dtype) { return dtype == "BF16" || dtype == "F16"; }

}  // namespace

// A weight qualifies when it is a 2-D linear matrix the runtime reads through
// the NVFP4 GEMM path. Everything else is copied through untouched: norms and
// biases are 1-D, embeddings stay full precision (quantizing them costs quality
// for no bandwidth win on the decode hot path), and K must be a multiple of 16
// because that is the micro-block size.
bool should_quantize(const RawTensor& t, bool quantize_lm_head, std::string& why_not) {
    if (!is_float_dtype(t.dtype)) {
        why_not = "dtype " + t.dtype + " (already quantized or unsupported)";
        return false;
    }
    // Rank is checked BEFORE the name, so a 3-D stack is diagnosed as a stack
    // whatever it is called. The other order made this branch unreachable for
    // every real stacked checkpoint, since none of them name the tensor
    // `.weight` — see find_stacked_expert_tensors.
    if (t.shape.size() == 3) {
        why_not = "3-D stacked tensor — needs the per-expert 2-D layout, not supported yet";
        return false;
    }
    if (!ends_with(t.name, ".weight")) {
        why_not = "not a .weight tensor";
        return false;
    }
    if (t.shape.size() != 2) {
        why_not = std::to_string(t.shape.size()) + "-D";
        return false;
    }
    if (contains(t.name, "embed_tokens") || contains(t.name, "embed_positions")) {
        why_not = "embedding";
        return false;
    }
    if (contains(t.name, "norm")) {
        why_not = "norm";
        return false;
    }
    if (contains(t.name, "lm_head") && !quantize_lm_head) {
        why_not = "lm_head (use --lm-head to include)";
        return false;
    }
    // Two weight roles that are 2-D and K-aligned — so every shape check waves
    // them through — but that must NOT be quantized. Both were found by
    // bisection on DeepSeek-V2-Lite (MLA + 64-expert MoE), where quantizing
    // everything produced a checkpoint that loaded and then emitted
    // cross-script repetition garbage while the BF16 source answered normally.
    // Excluding them costs almost nothing: a handful of small matrices per
    // layer against 4992 expert tensors that quantize fine.
    //
    // 1. MLA latent projections. `kv_a_proj_with_mqa` packs latent+RoPE into one
    //    output and `kv_b_proj` up-projects the latent into per-head nope/v
    //    halves; the runtime slices and reshapes both. Leaving only these two in
    //    full precision made the same checkpoint coherent again.
    if (contains(t.name, "kv_a_proj") || contains(t.name, "kv_b_proj")) {
        why_not = "MLA latent projection (runtime slices it — must stay full precision)";
        return false;
    }
    // 2. MoE router. It decides WHICH experts run, and FP4 across 16
    //    shared-scale values is enough to change the top-k pick. Measured
    //    independently: with the MLA pair already excluded, quantizing the
    //    router alone still produced garbage. Published exports (Modelopt,
    //    llm-compressor) keep routers full precision for the same reason.
    //    `.gate.weight` is the router; expert projections are `gate_proj.weight`
    //    and are unaffected by this suffix test.
    if (ends_with(t.name, ".gate.weight") || ends_with(t.name, ".router.weight")) {
        why_not = "MoE router (FP4 changes expert selection)";
        return false;
    }
    if (t.shape[1] % 16 != 0) {
        why_not = "K=" + std::to_string(t.shape[1]) + " is not a multiple of 16";
        return false;
    }
    return true;
}

std::vector<const RawTensor*> find_stacked_expert_tensors(const std::vector<RawTensor>& tensors) {
    std::vector<const RawTensor*> out;
    for (const auto& t : tensors) {
        // Rank 3 alone is not enough: conv1d kernels, patch embeddings and
        // position tables are 3-D too, and none of them is a reason to refuse a
        // checkpoint. "expert" in the name is what makes it the MoE bulk.
        if (t.shape.size() == 3 && is_float_dtype(t.dtype) && contains(t.name, "expert"))
            out.push_back(&t);
    }
    return out;
}

}  // namespace imp::quantize
