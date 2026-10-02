#include "core/logging.h"
#include "core/process_diag.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <string>

namespace imp {

namespace {

// HF SafeTensors stores GDN heads in grouped order (heads 0..n_v_per_k-1 = group 0); the
// scan kernel's default g=h%n_groups assumes the GGUF tiled layout. Set grouped_layout=1
// for HF loads. Cross-converted checkpoints may ship tiled; override gdn.layout_override="tiled".
void apply_gdn_layout(ModelConfig& cfg) {
    cfg.gdn_grouped_head_layout = true;
    const std::string& v = process_diag_gdn_layout_override();
    if (v == "tiled" || v == "TILED") {
        cfg.gdn_grouped_head_layout = false;
        IMP_LOG_INFO("GDN head layout: forced to TILED via gdn.layout_override=tiled");
    } else if (v == "grouped" || v == "GROUPED") {
        cfg.gdn_grouped_head_layout = true;
        IMP_LOG_INFO("GDN head layout: forced to GROUPED via gdn.layout_override=grouped");
    } else if (!v.empty()) {
        IMP_LOG_WARN("gdn.layout_override='%s' not recognized (expected 'tiled' or 'grouped')", v.c_str());
    }
}

}  // namespace

// Qwen3.5/3.6 GDN: HF exposes linear_* fields, mapped onto ssm_* slots read by
// executor_ssm_gdn.cu. Missing this leaves ssm_inner_size=0, conv_channels=0,
// ssm_proj_buf_ allocates 0 bytes: IMA on first GDN GEMM.
//   linear_value_head_dim x linear_num_value_heads -> ssm_inner_size
//   linear_key_head_dim                            -> ssm_state_size
//   linear_num_key_heads                           -> ssm_group_count
//   linear_num_value_heads                         -> ssm_dt_rank (n_heads)
//   linear_conv_kernel_dim                         -> ssm_conv_kernel
bool parse_qwen_gdn_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    apply_gdn_layout(cfg);

    int lin_v_heads = 0, lin_v_hdim = 0;
    int lin_k_heads = 0, lin_k_hdim = 0;
    int lin_conv = 0;
    jobj_opt_int(eff, "linear_num_value_heads", lin_v_heads);
    jobj_opt_int(eff, "linear_value_head_dim", lin_v_hdim);
    jobj_opt_int(eff, "linear_num_key_heads", lin_k_heads);
    jobj_opt_int(eff, "linear_key_head_dim", lin_k_hdim);
    jobj_opt_int(eff, "linear_conv_kernel_dim", lin_conv);
    // Qwen4Exp gated residual widths (absent on Qwen3.5/3.6: stay 0).
    jobj_opt_int(eff, "hc_count", cfg.hc_count);
    jobj_opt_int(eff, "hc_lowrank", cfg.hc_lowrank);
    jobj_opt_int(eff, "indexer_budget", cfg.qsa_budget);
    jobj_opt_int(eff, "indexer_compress_ratio", cfg.qsa_ratio);
    if (cfg.hc_count > 0)
        jobj_opt_int(eff, "eos_token_id", cfg.ple_eos_token_id);
    {
        std::string gate_act;
        if (jobj_get_string(eff, "output_gate_type", gate_act) && gate_act == "sigmoid")
            cfg.gdn_gate_sigmoid = true;
    }

    if (lin_v_heads > 0 && lin_v_hdim > 0) {
        cfg.ssm_inner_size = lin_v_heads * lin_v_hdim;
        cfg.ssm_dt_rank = lin_v_heads;
    }
    if (lin_k_hdim > 0)
        cfg.ssm_state_size = lin_k_hdim;
    if (lin_k_heads > 0)
        cfg.ssm_group_count = lin_k_heads;
    if (lin_conv > 0)
        cfg.ssm_conv_kernel = lin_conv;

    // layer_types[] decides per-layer GDN-vs-attention. The GGUF Qwen3.6 loader infers this
    // from tensor presence; the HF side surfaces it explicitly so executor_workspace/ssm-state
    // sizing picks the right layer count.
    const JValue* lt = jobj_find(eff, "layer_types");
    if (lt && lt->type == JType::ARRAY) {
        cfg.n_kv_heads_per_layer.clear();
        cfg.n_kv_heads_per_layer.reserve(lt->arr.size());
        for (const auto& v : lt->arr) {
            // 0 = no attention this layer (GDN), else cfg.n_kv_heads.
            bool is_linear = (v.str_val == "linear_attention");
            cfg.n_kv_heads_per_layer.push_back(is_linear ? 0 : cfg.n_kv_heads);
        }
    }

    IMP_LOG_INFO("  GDN config: inner=%d state=%d groups=%d n_heads=%d conv_kernel=%d", cfg.ssm_inner_size,
                 cfg.ssm_state_size, cfg.ssm_group_count, cfg.ssm_dt_rank, cfg.ssm_conv_kernel);

    // Qwen3.5 / 3.6 MoE shared-expert intermediate size. Used by the MTP
    // forward to size the shared-expert FFN scratch and by diagnostic logs
    // for the main model. Key name differs from DeepSeek-style configs.
    int qwen_shared_d_ff = 0;
    jobj_opt_int(eff, "shared_expert_intermediate_size", qwen_shared_d_ff);
    if (qwen_shared_d_ff == 0)
        jobj_opt_int(eff, "moe_shared_expert_intermediate_size", qwen_shared_d_ff);
    if (qwen_shared_d_ff > 0)
        cfg.expert_shared_d_ff = qwen_shared_d_ff;
    return true;
}

}  // namespace imp
