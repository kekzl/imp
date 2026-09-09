#include "awq_sites.h"

#include <algorithm>
#include <cctype>

namespace imp::awq {

namespace {

bool has(const std::set<std::string>& names, const std::string& n) { return names.count(n) != 0; }

bool wants(const std::string& groups, char g) { return groups.find(g) != std::string::npos; }

// Members that exist, in the order given. Callers decide whether a partial set
// is usable: for a shared pre-norm it never is.
std::vector<std::string> present(const std::set<std::string>& names, const std::string& base,
                                 const std::vector<const char*>& suffixes) {
    std::vector<std::string> out;
    for (const char* s : suffixes) {
        const std::string n = base + s;
        if (has(names, n))
            out.push_back(n);
    }
    return out;
}

std::vector<std::string> strs(const std::vector<const char*>& in) {
    return std::vector<std::string>(in.begin(), in.end());
}

}  // namespace

std::optional<NormConvention> arch_norm_convention(const std::string& model_type) {
    // Plain: out = norm(x) * g. The four this tool has always accepted.
    static const char* kPlain[] = {"qwen2", "qwen3", "llama", "mistral"};
    // Unit offset: out = norm(x) * (1 + g). imp bakes the +1 in at load for
    // these (src/model/weight_upload.cu arch_norm_offset), which is why the
    // runtime norm_weight_offset stays 0 and the fold has to carry it instead.
    // Spellings from src/model/hf_config_loader.cpp: a Qwen3.8 checkpoint
    // declares qwen3_5 at the top level and qwen3_5_text under text_config.
    static const char* kUnitOffset[] = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
                                        "qwen3_next"};
    for (const char* a : kPlain)
        if (model_type == a)
            return NormConvention::Plain;
    for (const char* a : kUnitOffset)
        if (model_type == a)
            return NormConvention::UnitOffset;
    // Gemma-class lands here and stays refused, but for the honest reason: its
    // block layout is not modelled, not because an offset cannot be folded.
    return std::nullopt;
}

std::string resolve_layer_prefix(const std::set<std::string>& names) {
    static const char* kKnown[] = {"model.layers.", "model.language_model.layers."};
    for (const char* p : kKnown) {
        const std::string pre = p;
        for (const auto& n : names)
            if (n.size() > pre.size() && n.compare(0, pre.size(), pre) == 0 &&
                std::isdigit(static_cast<unsigned char>(n[pre.size()])))
                return pre;
    }
    // Anything else that numbers its blocks the same way. The draft head is
    // excluded by name: mtp.* is never quantized (tensor_policy.cpp), so a
    // checkpoint whose only "layers." are the draft's has no site here.
    static const std::string kMark = "layers.";
    for (const auto& n : names) {
        if (n.rfind("mtp.", 0) == 0 || n.find(".visual.") != std::string::npos)
            continue;
        const size_t at = n.find(kMark);
        if (at == std::string::npos)
            continue;
        const size_t after = at + kMark.size();
        if (after < n.size() && std::isdigit(static_cast<unsigned char>(n[after])))
            return n.substr(0, after);
    }
    return "";
}

int64_t tie_producer_len(TieMode tie, int64_t K, const Geometry& geo) {
    if (K <= 0)
        return 0;
    switch (tie) {
        case TieMode::Identity:
            return K;
        case TieMode::GqaValue:
            if (geo.head_dim <= 0 || geo.n_rep <= 0 || K % (geo.head_dim * geo.n_rep) != 0)
                return 0;
            return K / geo.n_rep;
        case TieMode::PerHeadDim:
            if (geo.head_dim <= 0 || K % geo.head_dim != 0)
                return 0;
            return geo.head_dim;
    }
    return 0;
}

std::vector<int64_t> tie_map(TieMode tie, int64_t K, const Geometry& geo) {
    if (tie_producer_len(tie, K, geo) <= 0)
        return {};
    std::vector<int64_t> out(static_cast<size_t>(K));
    for (int64_t i = 0; i < K; i++) {
        switch (tie) {
            case TieMode::Identity:
                out[static_cast<size_t>(i)] = i;
                break;
            case TieMode::GqaValue: {
                // Several query heads read one KV head, so every o_proj input
                // channel mapping to the same v channel must get the same scale.
                const int64_t h = i / geo.head_dim, d = i % geo.head_dim;
                out[static_cast<size_t>(i)] = (h / geo.n_rep) * geo.head_dim + d;
                break;
            }
            case TieMode::PerHeadDim:
                // One norm row shared by every value head.
                out[static_cast<size_t>(i)] = i % geo.head_dim;
                break;
        }
    }
    return out;
}

std::vector<FoldSite> layer_fold_sites(const std::set<std::string>& names, const std::string& base,
                                       NormConvention conv, const std::string& groups) {
    const NormOffset block_norm = conv == NormConvention::UnitOffset ? NormOffset::Unit : NormOffset::Plain;
    const std::string attn = base + "self_attn.";
    const std::string gdn = base + "linear_attn.";
    const std::string mlp = base + "mlp.";
    const std::string in_norm = base + "input_layernorm.weight";
    const std::string post_norm = base + "post_attention_layernorm.weight";

    std::vector<FoldSite> out;

    // ---- C: o_proj into v_proj's output rows. No norm, so no offset. ----
    if (wants(groups, 'C') && has(names, attn + "o_proj.weight") && has(names, attn + "v_proj.weight")) {
        FoldSite s;
        s.group = 'C';
        s.members = {attn + "o_proj.weight"};
        s.producer = attn + "v_proj.weight";
        s.producer_bias = attn + "v_proj.bias";
        s.kind = FoldKind::MatrixRows;
        s.tie = TieMode::GqaValue;
        s.calib_keys = {"WO"};
        out.push_back(std::move(s));
    }

    // ---- D: down_proj into up_proj's output rows. ----
    if (wants(groups, 'D') && has(names, mlp + "down_proj.weight") && has(names, mlp + "up_proj.weight")) {
        FoldSite s;
        s.group = 'D';
        s.members = {mlp + "down_proj.weight"};
        s.producer = mlp + "up_proj.weight";
        s.producer_bias = mlp + "up_proj.bias";
        s.kind = FoldKind::MatrixRows;
        s.tie = TieMode::Identity;
        s.calib_keys = {"W_DOWN"};
        out.push_back(std::move(s));
    }

    // ---- E: out_proj into the GDN gated norm. PLAIN, and tied across heads. ----
    // linear_attn.norm is [head_dim] and shared by every value head
    // (src/compute/gdn_gated_norm.cu), so the divisor has to be constant over
    // the heads. The z gate multiplies elementwise AFTER the norm, so dividing
    // the norm weight divides the product and the gate is untouched.
    if (wants(groups, 'E') && has(names, gdn + "out_proj.weight") && has(names, gdn + "norm.weight")) {
        FoldSite s;
        s.group = 'E';
        s.members = {gdn + "out_proj.weight"};
        s.producer = gdn + "norm.weight";
        s.kind = FoldKind::NormVector;
        s.offset = NormOffset::Plain;
        s.tie = TieMode::PerHeadDim;
        s.calib_keys = {"SSM_OUT"};
        s.scan_prefix = gdn;
        // The four in-projections read the block's pre-norm hidden state, not
        // this norm's output, so they are not consumers of it.
        s.scan_exempt = strs(
            {"in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight"});
        out.push_back(std::move(s));
    }

    // ---- A: q/k/v into input_layernorm. ----
    if (wants(groups, 'A') && has(names, in_norm)) {
        auto members = present(names, attn, {"q_proj.weight", "k_proj.weight", "v_proj.weight"});
        if (!members.empty()) {
            FoldSite s;
            s.group = 'A';
            s.members = std::move(members);
            s.producer = in_norm;
            s.kind = FoldKind::NormVector;
            s.offset = block_norm;
            s.calib_keys = strs({"WQ", "WK", "WV"});
            s.scan_prefix = attn;
            // o_proj reads the attention output, not this norm.
            s.scan_exempt = strs({"o_proj.weight"});
            out.push_back(std::move(s));
        }
    }

    // ---- G: the four GDN in-projections into the same input_layernorm. ----
    // All four or none: folding the norm divides its output for every one of
    // them, and a member left out is a consumer whose columns were never
    // multiplied back up.
    if (wants(groups, 'G') && has(names, in_norm)) {
        const std::vector<const char*> want = {"in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight",
                                               "in_proj_b.weight"};
        auto members = present(names, gdn, want);
        if (members.size() == want.size()) {
            FoldSite s;
            s.group = 'G';
            s.members = std::move(members);
            s.producer = in_norm;
            s.kind = FoldKind::NormVector;
            s.offset = block_norm;
            s.calib_keys = strs(
                {"SSM_IN", "GDN_GATE", "GDN_ALPHA", "GDN_BETA", "GDN_INPUT_PACKED", "GDN_ALPHA_BETA_PACKED"});
            s.scan_prefix = gdn;
            // out_proj reads the scan output through linear_attn.norm (group E),
            // not this norm.
            s.scan_exempt = strs({"out_proj.weight"});
            out.push_back(std::move(s));
        }
    }

    // ---- B: gate/up into post_attention_layernorm. ----
    if (wants(groups, 'B') && has(names, post_norm)) {
        auto members = present(names, mlp, {"gate_proj.weight", "up_proj.weight"});
        if (!members.empty()) {
            FoldSite s;
            s.group = 'B';
            s.members = std::move(members);
            s.producer = post_norm;
            s.kind = FoldKind::NormVector;
            s.offset = block_norm;
            s.calib_keys = strs({"W_GATE", "W_UP"});
            s.scan_prefix = mlp;
            // down_proj reads the SwiGLU product, not this norm.
            s.scan_exempt = strs({"down_proj.weight"});
            out.push_back(std::move(s));
        }
    }

    return out;
}

}  // namespace imp::awq
