#include "awq_sites.h"

#include <algorithm>
#include <cctype>

namespace imp::awq {

namespace {

bool has(const std::set<std::string>& names, const std::string& n) { return names.contains(n); }

bool wants(const std::string& groups, char g) { return groups.contains(g); }

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

// The layer's per-expert name prefix ("<base>experts." Gemma-4 destacked, "<base>mlp.experts."
// Qwen-MoE) and the expert indices that carry an up_proj, ascending.
std::pair<std::string, std::vector<int64_t>> layer_experts(const std::set<std::string>& names,
                                                           const std::string& base) {
    for (const char* p : {"experts.", "mlp.experts.", "mixer.experts."}) {
        const std::string pre = base + p;
        std::set<int64_t> ids;
        for (auto it = names.lower_bound(pre); it != names.end() && it->starts_with(pre); ++it) {
            const size_t dot = it->find('.', pre.size());
            if (dot == std::string::npos || dot == pre.size() ||
                it->compare(dot, 15, ".up_proj.weight") != 0 || it->size() != dot + 15)
                continue;
            const std::string num = it->substr(pre.size(), dot - pre.size());
            if (std::all_of(num.begin(), num.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
                ids.insert(std::stoll(num));
        }
        if (!ids.empty())
            return {pre, std::vector<int64_t>(ids.begin(), ids.end())};
    }
    return {};
}

// Nemotron-H (backbone.layers.N.{norm, mixer.*}): one mixer per block, read through the block's
// pre-norm. Mamba2: in_proj (G), out_proj into the gated group norm (E, per channel). Attention:
// q/k/v (A), o_proj into v rows (C). MoE: non-gated relu^2 shared expert, so the S row fold takes the
// square root of the divisor; the pre-norm also feeds the router, so no X.
std::vector<FoldSite> nemotron_h_fold_sites(const std::set<std::string>& names, const std::string& base,
                                            const std::string& groups) {
    const std::string mix = base + "mixer.";
    const std::string pre = base + "norm.weight";
    const std::vector<const char*> mamba_rest = {"conv1d.weight", "conv1d.bias", "norm.weight",
                                                 "A_log",         "D",           "dt_bias"};
    std::vector<FoldSite> out;
    if (wants(groups, 'C') && has(names, mix + "o_proj.weight") && has(names, mix + "v_proj.weight")) {
        FoldSite s;
        s.group = 'C';
        s.members = {mix + "o_proj.weight"};
        s.producer = mix + "v_proj.weight";
        s.kind = FoldKind::MatrixRows;
        s.tie = TieMode::GqaValue;
        s.calib_keys = {"WO"};
        out.push_back(std::move(s));
    }
    // No Y: --calibrate records no down statistic for non-gated experts (executor_forward_moe.cu
    // calibrate_moe_experts_), every expert site would skip.
    const std::string sh = mix + "shared_experts.";
    if (wants(groups, 'S') && has(names, sh + "down_proj.weight") && has(names, sh + "up_proj.weight") &&
        !has(names, sh + "gate_proj.weight")) {
        FoldSite s;
        s.group = 'S';
        s.members = {sh + "down_proj.weight"};
        s.producer = sh + "up_proj.weight";
        s.kind = FoldKind::MatrixRows;
        s.producer_sqrt = true;
        s.calib_keys = {"W_DOWN"};
        out.push_back(std::move(s));
    }
    if (wants(groups, 'E') && has(names, mix + "out_proj.weight") && has(names, mix + "norm.weight")) {
        FoldSite s;
        s.group = 'E';
        s.members = {mix + "out_proj.weight"};
        s.producer = mix + "norm.weight";
        s.kind = FoldKind::NormVector;
        s.offset = NormOffset::Plain;
        s.calib_keys = {"SSM_OUT"};
        s.scan_prefix = mix;
        s.scan_exempt = strs({"in_proj.weight", "conv1d.weight", "conv1d.bias", "A_log", "D", "dt_bias"});
        out.push_back(std::move(s));
    }
    if (wants(groups, 'A') && has(names, pre)) {
        auto members = present(names, mix, {"q_proj.weight", "k_proj.weight", "v_proj.weight"});
        if (members.size() == 3) {
            FoldSite s;
            s.group = 'A';
            s.members = std::move(members);
            s.producer = pre;
            s.kind = FoldKind::NormVector;
            s.offset = NormOffset::Plain;
            s.calib_keys = strs({"WQ", "WK", "WV"});
            s.scan_prefix = mix;
            s.scan_exempt = strs({"o_proj.weight"});
            out.push_back(std::move(s));
        }
    }
    if (wants(groups, 'G') && has(names, pre) && has(names, mix + "in_proj.weight")) {
        FoldSite s;
        s.group = 'G';
        s.members = {mix + "in_proj.weight"};
        s.producer = pre;
        s.kind = FoldKind::NormVector;
        s.offset = NormOffset::Plain;
        s.calib_keys = {"SSM_IN"};
        s.scan_prefix = mix;
        std::vector<std::string> ex = strs(mamba_rest);
        ex.emplace_back("out_proj.weight");  // reads the gated norm (group E)
        s.scan_exempt = std::move(ex);
        out.push_back(std::move(s));
    }
    return out;
}

}  // namespace

std::optional<NormConvention> arch_norm_convention(const std::string& model_type) {
    // Plain: out = norm(x) * g. The four this tool has always accepted.
    // gemma4: imp applies the norm weight as stored (norm_weight_offset 0, no load-time +1).
    // deepseek_v2: plain RMSNorm, MLA attention (kv_a_layernorm on the latent), no load-time offset.
    // glm4_moe_lite (GLM-4.7-Flash): the same MLA plus q_lora (q_a_proj, q_a_layernorm, q_b_proj).
    // nemotron_h: plain RMSNorm block pre-norms and plain gated group norm in the Mamba2 mixer.
    static const char* kPlain[] = {"qwen2",       "qwen3",       "llama",         "mistral",   "gemma4",
                                   "gemma4_text", "deepseek_v2", "glm4_moe_lite", "nemotron_h"};
    // Unit offset: out = norm(x) * (1 + g). imp bakes the +1 at load for these
    // (src/model/weight_upload.cpp arch_norm_offset), so runtime norm_weight_offset stays 0 and the
    // fold must carry it. Spellings from src/model/hf_config_loader.cpp: a Qwen3.8 checkpoint
    // declares qwen3_5 top-level and qwen3_5_text under text_config.
    static const char* kUnitOffset[] = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
                                        "qwen3_next"};
    for (const char* a : kPlain)
        if (model_type == a)
            return NormConvention::Plain;
    for (const char* a : kUnitOffset)
        if (model_type == a)
            return NormConvention::UnitOffset;
    // Gemma 2/3 land here and stay refused: their block layout is not modelled.
    return std::nullopt;
}

bool arch_normalizes_v(const std::string& model_type) {
    return model_type == "gemma4" || model_type == "gemma4_text";
}

std::string resolve_layer_prefix(const std::set<std::string>& names) {
    static const char* kKnown[] = {"model.layers.", "model.language_model.layers."};
    for (const char* p : kKnown) {
        const std::string pre = p;
        for (const auto& n : names)
            if (n.size() > pre.size() && n.starts_with(pre) &&
                std::isdigit(static_cast<unsigned char>(n[pre.size()])))
                return pre;
    }
    // Anything else that numbers its blocks the same way. The draft head is
    // excluded by name: mtp.* is never quantized (tensor_policy.cpp), so a
    // checkpoint whose only "layers." are the draft's has no site here.
    static const std::string kMark = "layers.";
    for (const auto& n : names) {
        if (n.starts_with("mtp.") || n.contains(".visual."))
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
                out[static_cast<size_t>(i)] = ((h / geo.n_rep) * geo.head_dim) + d;
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
                                       NormConvention conv, const std::string& groups, bool v_normed) {
    // Nemotron-H blocks carry one pre-norm (backbone.layers.N.norm) and a single mixer.
    if (has(names, base + "norm.weight") && !has(names, base + "input_layernorm.weight"))
        return nemotron_h_fold_sites(names, base, groups);
    const NormOffset block_norm = conv == NormConvention::UnitOffset ? NormOffset::Unit : NormOffset::Plain;
    const std::string attn = base + "self_attn.";
    const std::string gdn = base + "linear_attn.";
    const std::string mlp = base + "mlp.";
    const std::string in_norm = base + "input_layernorm.weight";
    const std::string post_norm = base + "post_attention_layernorm.weight";

    std::vector<FoldSite> out;

    // ---- C: o_proj into v_proj's output rows. No norm, so no offset. ----
    if (wants(groups, 'C') && !v_normed && has(names, attn + "o_proj.weight") &&
        has(names, attn + "v_proj.weight")) {
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

    // ---- Y: per expert, down_proj into that expert's up_proj rows (D, once per expert). ----
    const auto [experts, expert_ids] = layer_experts(names, base);
    if (wants(groups, 'Y')) {
        for (const int64_t e : expert_ids) {
            const std::string ex = experts + std::to_string(e) + ".";
            if (!has(names, ex + "down_proj.weight") || !has(names, ex + "up_proj.weight"))
                continue;
            FoldSite s;
            s.group = 'Y';
            s.members = {ex + "down_proj.weight"};
            s.producer = ex + "up_proj.weight";
            s.kind = FoldKind::MatrixRows;
            s.tie = TieMode::Identity;
            s.calib_keys = {"EXPERT_DOWN." + std::to_string(e)};
            out.push_back(std::move(s));
        }
    }

    // E: out_proj into the GDN gated norm, PLAIN and tied across heads. linear_attn.norm is
    // [head_dim] and shared by every value head (src/compute/gdn_gated_norm.cu), so the divisor
    // must be constant over heads; the z gate multiplies elementwise AFTER the norm, so dividing
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

    // ---- A: q/k/v (MLA: q_proj or q_a_proj, + kv_a_proj_with_mqa) into input_layernorm. ----
    if (wants(groups, 'A') && has(names, in_norm)) {
        auto members = present(names, attn,
                               {"q_proj.weight", "k_proj.weight", "v_proj.weight",
                                "kv_a_proj_with_mqa.weight", "q_a_proj.weight"});
        if (!members.empty()) {
            FoldSite s;
            s.group = 'A';
            s.members = std::move(members);
            s.producer = in_norm;
            s.kind = FoldKind::NormVector;
            s.offset = block_norm;
            // q_a_proj records under KV_A_PROJ (one TensorKind for both MLA down-projections).
            s.calib_keys = strs({"WQ", "WK", "WV", "KV_A_PROJ"});
            s.scan_prefix = attn;
            // o_proj reads the attention output; MLA kv_b_proj reads kv_a_layernorm (group K),
            // q_b_proj reads q_a_layernorm (group Q).
            s.scan_exempt = strs({"o_proj.weight", "kv_b_proj.weight", "kv_a_layernorm.weight",
                                  "q_b_proj.weight", "q_a_layernorm.weight"});
            out.push_back(std::move(s));
        }
    }

    // ---- K: MLA kv_b_proj into kv_a_layernorm, the RMSNorm on the 512-dim latent. ----
    if (wants(groups, 'K') && has(names, attn + "kv_a_layernorm.weight") &&
        has(names, attn + "kv_b_proj.weight")) {
        FoldSite s;
        s.group = 'K';
        s.members = {attn + "kv_b_proj.weight"};
        s.producer = attn + "kv_a_layernorm.weight";
        s.kind = FoldKind::NormVector;
        s.offset = block_norm;
        s.calib_keys = {"KV_B_PROJ"};
        s.scan_prefix = attn;
        // Every other attention projection reads input_layernorm, q_a_layernorm or the attention output.
        s.scan_exempt = strs({"q_proj.weight", "kv_a_proj_with_mqa.weight", "o_proj.weight",
                              "q_a_proj.weight", "q_b_proj.weight", "q_a_layernorm.weight"});
        out.push_back(std::move(s));
    }

    // ---- Q: MLA q_lora q_b_proj into q_a_layernorm, the RMSNorm on the q latent (GLM, V3). ----
    if (wants(groups, 'Q') && has(names, attn + "q_a_layernorm.weight") &&
        has(names, attn + "q_b_proj.weight")) {
        FoldSite s;
        s.group = 'Q';
        s.members = {attn + "q_b_proj.weight"};
        s.producer = attn + "q_a_layernorm.weight";
        s.kind = FoldKind::NormVector;
        s.offset = block_norm;
        s.calib_keys = {"WQ"};  // q_b_proj is the WQ GEMM; its input is the q latent
        s.scan_prefix = attn;
        s.scan_exempt = strs({"q_a_proj.weight", "kv_a_proj_with_mqa.weight", "kv_b_proj.weight",
                              "kv_a_layernorm.weight", "o_proj.weight"});
        out.push_back(std::move(s));
    }

    // ---- S: the shared expert's down_proj into its up_proj rows (D for mlp.shared_experts). ----
    const std::string shared = mlp + "shared_experts.";
    if (wants(groups, 'S') && has(names, shared + "down_proj.weight") &&
        has(names, shared + "up_proj.weight")) {
        FoldSite s;
        s.group = 'S';
        s.members = {shared + "down_proj.weight"};
        s.producer = shared + "up_proj.weight";
        s.kind = FoldKind::MatrixRows;
        s.tie = TieMode::Identity;
        s.calib_keys = {"W_DOWN"};  // a MoE layer's only W_DOWN GEMM is the shared expert's
        out.push_back(std::move(s));
    }

    // G: the four GDN in-projections into the same input_layernorm, all-or-none: folding the norm
    // divides its output for every one, so a member left out is a consumer whose columns were
    // never multiplied back up.
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

    // ---- B: gate/up into the MLP pre-norm. A sandwich-norm block (Gemma-4) applies
    // post_attention_layernorm to the attention OUTPUT; its MLP reads pre_feedforward_layernorm. ----
    const std::string mlp_norm = has(names, base + "pre_feedforward_layernorm.weight")
                                     ? base + "pre_feedforward_layernorm.weight"
                                     : post_norm;
    if (wants(groups, 'B') && has(names, mlp_norm)) {
        auto members = present(names, mlp, {"gate_proj.weight", "up_proj.weight"});
        if (!members.empty()) {
            FoldSite s;
            s.group = 'B';
            s.members = std::move(members);
            s.producer = mlp_norm;
            s.kind = FoldKind::NormVector;
            s.offset = block_norm;
            s.calib_keys = strs({"W_GATE", "W_UP"});
            s.scan_prefix = mlp;
            // down_proj reads the SwiGLU product, not this norm.
            s.scan_exempt = strs({"down_proj.weight"});
            out.push_back(std::move(s));
        }
    }

    // ---- X: every expert's gate/up into pre_feedforward_layernorm_2, one shared scale. Only that
    // norm: post_attention_layernorm also feeds the router and the shared expert. ----
    const std::string expert_norm = base + "pre_feedforward_layernorm_2.weight";
    if (wants(groups, 'X') && has(names, expert_norm) && !expert_ids.empty()) {
        FoldSite s;
        s.group = 'X';
        for (const int64_t e : expert_ids) {
            const std::string ex = experts + std::to_string(e) + ".";
            for (const char* p : {"gate_proj.weight", "up_proj.weight"})
                if (has(names, ex + p))
                    s.members.push_back(ex + p);
        }
        s.producer = expert_norm;
        s.kind = FoldKind::NormVector;
        s.offset = block_norm;
        s.calib_keys = {"EXPERT_UP"};
        s.scan_prefix = experts;
        s.scan_exempt = strs({"down_proj.weight"});
        out.push_back(std::move(s));
    }

    return out;
}

}  // namespace imp::awq
