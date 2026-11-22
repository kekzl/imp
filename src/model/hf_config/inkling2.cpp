#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

namespace imp {

// Inkling2-Loop (thinkingmachines/Inkling-2-Mini): the generic keys cover layers, width and heads.
// The lm_head is the first unpadded_vocab_size rows of the unembedding; everything arch-specific
// (loop, distance bias, convs, experts) is read by model/inkling_weights.cu from the same file.
bool parse_inkling2_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    int vocab = 0;
    if (!jobj_get_int(eff, "unpadded_vocab_size", vocab) || vocab <= 0) {
        IMP_LOG_ERROR("config.json: Inkling2-Loop needs unpadded_vocab_size");
        return false;
    }
    cfg.vocab_size = vocab;
    int ctx = 0;
    jobj_opt_int(eff, "model_max_length", ctx);
    if (ctx > 0)
        cfg.max_seq_len = ctx;
    jobj_opt_float(eff, "rms_norm_eps", cfg.rms_norm_eps);
    jobj_opt_float(eff, "final_logit_softcapping", cfg.final_logit_softcap);
    cfg.tie_word_embeddings = 0;
    return true;
}

}  // namespace imp
