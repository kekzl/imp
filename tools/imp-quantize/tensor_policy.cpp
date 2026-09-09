#include "tensor_policy.h"

#include "model/nvfp4_module_policy.h"

#include <map>

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
    // The role exclusions live in src/model/nvfp4_module_policy.h, which the
    // LOADER also calls to decide whether a module the checkpoint left plain is
    // a Linear it should have found packed. One list, so the writer's silence
    // and the reader's refusal cannot disagree about which roles are exempt.
    if (imp::nvfp4_policy::role_excluded(t.name, t.shape[1], quantize_lm_head, why_not))
        return false;
    return true;
}

Fp8SourceAction fp8_source_action(const RawTensor& weight, bool gated, bool quantize_lm_head,
                                  std::string& why_not) {
    if (gated) {
        why_not = "fused Q+gate projection (--keep-attn-gate)";
        return Fp8SourceAction::WidenToFullPrecision;
    }
    // Ask about the widened form. The dtype gate in should_quantize would
    // otherwise refuse every FP8 tensor as "already quantized" before it ever
    // looked at the role, and the roles are the whole point of this call.
    RawTensor widened = weight;
    widened.dtype = "BF16";
    return should_quantize(widened, quantize_lm_head, why_not) ? Fp8SourceAction::Quantize
                                                               : Fp8SourceAction::WidenToFullPrecision;
}

size_t nvfp4_output_bytes(int64_t N, int64_t K) {
    if (N <= 0 || K <= 0)
        return 0;
    const size_t n = static_cast<size_t>(N), k = static_cast<size_t>(K);
    return n * k / 2       // packed FP4 nibbles, two per byte
           + n * k / 16    // one FP8 E4M3 micro-scale per 16 values
           + sizeof(float);  // the per-tensor F32 scale
}

std::vector<const RawTensor*> find_fused_gate_q_projections(const std::vector<RawTensor>& tensors) {
    static const std::string kQ = ".q_proj.weight";
    static const std::string kO = ".o_proj.weight";
    // Compared per LAYER, keyed on the shared prefix. Pairing a q_proj against
    // some other layer's o_proj would report the wrong set the moment a hybrid
    // gives its layers different head counts.
    std::map<std::string, int64_t> o_proj_cols;
    for (const auto& t : tensors) {
        if (t.shape.size() == 2 && ends_with(t.name, kO))
            o_proj_cols[t.name.substr(0, t.name.size() - kO.size())] = t.shape[1];
    }
    std::vector<const RawTensor*> out;
    for (const auto& t : tensors) {
        if (t.shape.size() != 2 || !ends_with(t.name, kQ))
            continue;
        const auto it = o_proj_cols.find(t.name.substr(0, t.name.size() - kQ.size()));
        // No o_proj for this layer means no reference; a bare "shape[0] is even"
        // test would flag every ordinary projection, so say nothing instead.
        if (it != o_proj_cols.end() && t.shape[0] == 2 * it->second)
            out.push_back(&t);
    }
    return out;
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
