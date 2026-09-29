// dispatch_mtp_head: raw mtp.* tensor map -> MtpHead fields, three checkpoint layouts.
#include "model/mtp_head.h"

#include "core/logging.h"

#include <string>

namespace imp {

// Dispatches a raw mtp.* tensor map to MtpHead fields. Qwen layout has two MLP variants: MoE
// (Qwen3.6-35B): router + packed experts + gated shared expert. Dense (Qwen3.6-27B embedded
// head): plain SwiGLU gate/up/down mapped onto the shared_expert fields (router/experts stay
// empty, forward runs it as residual + shared path, no sigmoid gate).
MtpHead dispatch_mtp_head(const std::unordered_map<std::string, Tensor>& tm, const std::string& src,
                          size_t bytes) {
    MtpHead head;
    head.info.path = src;
    head.info.file_bytes = bytes;
    head.info.n_tensors = static_cast<int>(tm.size());
    auto take = [&](const std::string& key, Tensor& dst) -> bool {
        auto it = tm.find(key);
        if (it == tm.end())
            return false;
        dst = it->second;
        head.info.n_mapped++;
        return true;
    };
    // Qwen4Exp layout (docs/plans/2026-09-28-qwen4exp-mtp.md): split fc, hyper-connection
    // modules instead of pre-norms and final norm, QSA indexer, 512 per-expert FP8 experts.
    if (tm.count(kMtpHeadKeyFcEmbedding)) {
        head.layout = MtpLayout::Qwen4Exp;
        bool ok = true;
        ok &= take("mtp.pre_fc_norm_embedding.weight", head.pre_fc_norm_embedding);
        ok &= take("mtp.pre_fc_norm_hidden.weight", head.pre_fc_norm_hidden);
        ok &= take(kMtpHeadKeyFcEmbedding, head.fc_embedding);
        ok &= take("mtp.fc_hidden.weight", head.fc_hidden);
        // pre_fc_norm_hidden spans the whole hc stream: [hc_count * hidden] (4 x 2560).
        if (ok && head.fc_hidden.ndim == 2 && head.fc_hidden.shape[1] > 0 &&
            head.pre_fc_norm_hidden.shape[0] % head.fc_hidden.shape[1] == 0)
            head.hc_count = static_cast<int>(head.pre_fc_norm_hidden.shape[0] / head.fc_hidden.shape[1]);
        ok &= head.hc_count > 1;
        auto take_hc = [&](const std::string& p, MtpHyperConnection& hc, bool inject) -> bool {
            bool r = take(p + ".hc_norm.weight", hc.norm);
            r &= take(p + ".input_mix_weight_down.weight", hc.mix_down);
            r &= take(p + ".input_mix_weight_up.weight", hc.mix_up);
            if (inject)
                r &= take(p + ".block_inject_weight.weight", hc.block_inject);
            return r;
        };
        ok &= take_hc("mtp.hyper_connection_mixer", head.final_mixer, /*inject=*/false);
        ok &= take_hc("mtp.layers.0.attn_hyper_connection", head.attn_hc, /*inject=*/true);
        ok &= take_hc("mtp.layers.0.mlp_hyper_connection", head.mlp_hc, /*inject=*/true);
        ok &= take("mtp.layers.0.self_attn.q_proj.weight", head.q_proj);
        ok &= take("mtp.layers.0.self_attn.k_proj.weight", head.k_proj);
        ok &= take("mtp.layers.0.self_attn.v_proj.weight", head.v_proj);
        ok &= take("mtp.layers.0.self_attn.o_proj.weight", head.o_proj);
        ok &= take("mtp.layers.0.self_attn.q_norm.weight", head.q_norm);
        ok &= take("mtp.layers.0.self_attn.k_norm.weight", head.k_norm);
        ok &= take("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", head.indexer_qk_proj);
        ok &= take("mtp.layers.0.self_attn.indexer.q_layernorm.weight", head.indexer_q_norm);
        ok &= take("mtp.layers.0.self_attn.indexer.k_layernorm.weight", head.indexer_k_norm);
        ok &= take("mtp.layers.0.mlp.gate.weight", head.router);
        ok &= take("mtp.layers.0.mlp.shared_expert.gate_proj.weight", head.shared_expert_gate_proj);
        ok &= take("mtp.layers.0.mlp.shared_expert.up_proj.weight", head.shared_expert_up_proj);
        ok &= take("mtp.layers.0.mlp.shared_expert.down_proj.weight", head.shared_expert_down_proj);
        ok &= take("mtp.layers.0.mlp.shared_expert_gate.weight", head.shared_expert_gate);
        // Gated attention (q_proj = 2 x heads x head_dim), partial RoPE, SwiGLU experts.
        head.attn_output_gate = true;
        head.attn_rope = true;
        head.experts_non_gated = false;

        // Expert count from the router rows; a numbering gap is a hard failure.
        const int n_exp =
            head.router.data != nullptr && head.router.ndim >= 1 ? static_cast<int>(head.router.shape[0]) : 0;
        if (n_exp <= 0)
            ok = false;
        head.experts_fp8.resize(static_cast<size_t>(n_exp > 0 ? n_exp : 0));
        for (int e = 0; e < n_exp && ok; ++e) {
            const std::string p = "mtp.layers.0.mlp.experts." + std::to_string(e) + ".";
            MtpFp8Expert& x = head.experts_fp8[static_cast<size_t>(e)];
            ok &= take(p + "gate_proj.weight", x.gate_proj);
            ok &= take(p + "up_proj.weight", x.up_proj);
            ok &= take(p + "down_proj.weight", x.down_proj);
            ok &= take(p + "gate_proj.weight_scale_inv", x.gate_scale_inv);
            ok &= take(p + "up_proj.weight_scale_inv", x.up_scale_inv);
            ok &= take(p + "down_proj.weight_scale_inv", x.down_scale_inv);
        }

        head.loaded = ok;
        if (ok) {
            IMP_LOG_INFO("MTP head loaded: %s (%d of %d tensors mapped, qwen4_exp layout, %d FP8 experts, "
                         "%.2f GiB, hc_count %d)",
                         src.c_str(), head.info.n_mapped, head.info.n_tensors, n_exp,
                         static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), head.hc_count);
        } else {
            IMP_LOG_WARN("MTP head at %s uses the qwen4_exp layout but is incomplete (%d of %d tensors mapped, "
                         "experts=%d); spec-decode disabled",
                         src.c_str(), head.info.n_mapped, head.info.n_tensors, n_exp);
        }
        return head;
    }
    // Nemotron-3.5 layout: a miniature Nemotron, not Qwen - attention in layers.0, MoE in
    // layers.1, mixer where Qwen says self_attn/mlp, per-expert 2-D weights instead of one
    // packed 3-D stack. Detected on a name only this layout has, so the Qwen path below stays
    // byte-for-byte unaffected.
    if (tm.count(kMtpHeadKeyEhProj)) {
        head.layout = MtpLayout::Nemotron;
        bool nok = true;
        nok &= take("mtp.layers.0.enorm.weight", head.pre_fc_norm_embedding);
        nok &= take("mtp.layers.0.hnorm.weight", head.pre_fc_norm_hidden);
        nok &= take(kMtpHeadKeyEhProj, head.fc);
        nok &= take("mtp.layers.0.norm.weight", head.input_layernorm);
        nok &= take("mtp.layers.0.mixer.q_proj.weight", head.q_proj);
        nok &= take("mtp.layers.0.mixer.k_proj.weight", head.k_proj);
        nok &= take("mtp.layers.0.mixer.v_proj.weight", head.v_proj);
        nok &= take("mtp.layers.0.mixer.o_proj.weight", head.o_proj);
        // No q_norm / k_norm in this layout — left null on purpose.
        nok &= take("mtp.layers.1.norm.weight", head.post_attention_layernorm);
        nok &= take("mtp.layers.1.final_layernorm.weight", head.final_norm);
        nok &= take("mtp.layers.1.mixer.gate.weight", head.router);
        // Router bias is optional: absent means plain top-k.
        take("mtp.layers.1.mixer.gate.e_score_correction_bias", head.router_score_bias);
        nok &= take("mtp.layers.1.mixer.shared_experts.up_proj.weight", head.shared_expert_up_proj);
        nok &= take("mtp.layers.1.mixer.shared_experts.down_proj.weight", head.shared_expert_down_proj);
        // shared_expert_gate_proj / shared_expert_gate stay null: these
        // experts are non-gated (squared ReLU), like the main Nemotron FFN.
        head.experts_non_gated = true;
        // No attn_output_gate (q_proj is n_heads*head_dim, not twice that)
        // and NoPE attention, both as in the main Nemotron-H layers.
        head.attn_output_gate = false;
        head.attn_rope = false;

        // Per-expert weights. Count is taken from the router's row count so
        // a checkpoint with a different expert count still loads, and a gap
        // in the numbering is a hard failure rather than a silent short read.
        const int n_exp = head.router.data && head.router.ndim >= 1
                              ? static_cast<int>(head.router.shape[0])
                              : 0;
        if (n_exp <= 0) {
            nok = false;
        } else {
            head.experts_up.resize(static_cast<size_t>(n_exp));
            head.experts_down.resize(static_cast<size_t>(n_exp));
            for (int e = 0; e < n_exp && nok; ++e) {
                const std::string p = "mtp.layers.1.mixer.experts." + std::to_string(e) + ".";
                nok &= take((p + "up_proj.weight").c_str(), head.experts_up[static_cast<size_t>(e)]);
                nok &= take((p + "down_proj.weight").c_str(), head.experts_down[static_cast<size_t>(e)]);
            }
        }

        head.loaded = nok;
        if (nok) {
            IMP_LOG_INFO(
                "MTP head loaded: %s (%d tensors, Nemotron layout, %d non-gated "
                "experts%s, BF16)",
                src.c_str(), head.info.n_tensors, n_exp,
                head.router_score_bias.data ? " + router bias" : "");
        } else {
            IMP_LOG_WARN(
                "MTP head at %s uses the Nemotron layout but is incomplete "
                "(router=%d experts=%d); spec-decode disabled",
                src.c_str(), head.router.data ? 1 : 0, n_exp);
        }
        return head;
    }

    head.layout = MtpLayout::Qwen;
    bool ok = true;
    ok &= take("mtp.pre_fc_norm_embedding.weight", head.pre_fc_norm_embedding);
    ok &= take("mtp.pre_fc_norm_hidden.weight", head.pre_fc_norm_hidden);
    ok &= take(kMtpHeadKeyFc, head.fc);
    ok &= take("mtp.layers.0.input_layernorm.weight", head.input_layernorm);
    ok &= take("mtp.layers.0.post_attention_layernorm.weight", head.post_attention_layernorm);
    ok &= take("mtp.layers.0.self_attn.q_proj.weight", head.q_proj);
    ok &= take("mtp.layers.0.self_attn.k_proj.weight", head.k_proj);
    ok &= take("mtp.layers.0.self_attn.v_proj.weight", head.v_proj);
    ok &= take("mtp.layers.0.self_attn.o_proj.weight", head.o_proj);
    ok &= take("mtp.layers.0.self_attn.q_norm.weight", head.q_norm);
    ok &= take("mtp.layers.0.self_attn.k_norm.weight", head.k_norm);
    ok &= take("mtp.norm.weight", head.final_norm);

    const bool moe = take("mtp.layers.0.mlp.gate.weight", head.router) &&
                     take("mtp.layers.0.mlp.experts.gate_up_proj", head.experts_gate_up_packed) &&
                     take("mtp.layers.0.mlp.experts.down_proj", head.experts_down_packed) &&
                     take("mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                          head.shared_expert_gate_proj) &&
                     take("mtp.layers.0.mlp.shared_expert.up_proj.weight", head.shared_expert_up_proj) &&
                     take("mtp.layers.0.mlp.shared_expert.down_proj.weight",
                          head.shared_expert_down_proj) &&
                     take("mtp.layers.0.mlp.shared_expert_gate.weight", head.shared_expert_gate);
    const bool dense = !moe && take("mtp.layers.0.mlp.gate_proj.weight", head.shared_expert_gate_proj) &&
                       take("mtp.layers.0.mlp.up_proj.weight", head.shared_expert_up_proj) &&
                       take("mtp.layers.0.mlp.down_proj.weight", head.shared_expert_down_proj);
    ok &= (moe || dense);

    head.loaded = ok;
    if (ok) {
        IMP_LOG_INFO("MTP head loaded: %s (%d tensors, %s MLP, BF16)", src.c_str(), head.info.n_tensors,
                     moe ? "MoE" : "dense");
    } else {
        IMP_LOG_WARN(
            "MTP head detected at %s but some expected tensors were missing; "
            "spec-decode disabled for this model",
            src.c_str());
    }
    return head;
}

}  // namespace imp
