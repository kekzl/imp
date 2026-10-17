// Classifier-free guidance (roadmap row 117). The owner waits outside the scheduler until its
// shadow (the negative prompt) has prefilled and saved its last-row logits; the owner's first
// token then samples from l_a + scale * (l_a - l_b). From there both decode in the same step:
// rows of a pair run together or not at all, the owner's row is combined before sampling and the
// shadow appends the owner's token. A shadow never stops on its own (ignore_eos); it is released
// when its owner finishes or is cancelled.

#include "runtime/engine.h"
#include "runtime/cfg_guidance.h"
#include "runtime/request.h"
#include "compute/cfg_combine.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "model/model.h"

#include <algorithm>
#include <unordered_map>

namespace imp {

namespace {

bool cfg_done(const Request& r) {
    return r.status == RequestStatus::FINISHED || r.status == RequestStatus::CANCELLED;
}

// The partner of a pair member: the owner's shadow or the shadow's owner (null when gone).
std::shared_ptr<Request> cfg_partner(const Request& r) {
    if (!r.cfg)
        return nullptr;
    return r.cfg->shadow ? r.cfg->shadow : r.cfg->owner.lock();
}

}  // namespace

bool Engine::cfg_supported() const {
    // A pair decodes as two rows of one step: hybrids need the batched GDN decode for that.
    const bool gdn_batch_ok = runtime_config_.runtime.gdn_batched_decode && d_ssm_seq_slots_ != nullptr;
    return model_ && (!ssm_state_ || gdn_batch_ok);
}

void Engine::cfg_admit_(std::shared_ptr<Request> req) {
    if (req->cfg && req->cfg->shadow && !cfg_supported()) {
        IMP_LOG_WARN("cfg: req %d runs unguided, the model cannot decode the pair in one step", req->id);
        req->cfg.reset();
    }
    if (!req->cfg || !req->cfg->shadow) {  // plain request, or a shadow
        scheduler_->add_request(std::move(req));
        return;
    }
    std::shared_ptr<Request> shadow = req->cfg->shadow;
    shadow->cfg = std::make_shared<CfgState>();
    shadow->cfg->owner = req;
    shadow->ignore_eos = true;  // ends with its owner, never on its own
    shadow->max_tokens = req->max_tokens + 8;
    shadow->trace_id = req->trace_id.empty() ? std::string() : req->trace_id + "-cfg";
    cfg_shadows_.push_back(shadow);
    add_request(shadow);  // prefills now; the owner joins once the shadow's logits are saved
    cfg_held_.push_back(std::move(req));
}

void Engine::cfg_tick_() {
    if (cfg_held_.empty() && cfg_shadows_.empty())
        return;
    // Owners whose shadow prefilled join the scheduler; cancelled ones are dropped.
    std::erase_if(cfg_held_, [&](std::shared_ptr<Request>& owner) {
        if (cfg_done(*owner))
            return true;
        const auto& shadow = owner->cfg->shadow;
        if (cfg_done(*shadow)) {  // the shadow never prefilled (cancelled, KV): unguided
            IMP_LOG_WARN("cfg: req %d runs unguided, its shadow req %d ended before its prefill", owner->id,
                         shadow->id);
            owner->cfg.reset();
        } else if (!shadow->cfg->ready) {
            return false;
        }
        scheduler_->add_request(owner);
        return true;
    });
    // Shadows of finished or cancelled owners are released.
    std::erase_if(cfg_shadows_, [&](std::shared_ptr<Request>& shadow) {
        if (cfg_done(*shadow))
            return true;
        const auto owner = shadow->cfg->owner.lock();
        if (owner && !cfg_done(*owner))
            return false;
        if (shadow->status == RequestStatus::DECODING) {
            finish_request(shadow);
        } else {
            shadow->status = RequestStatus::CANCELLED;
            kv_manager_->free_sequence(shadow->id);
            reset_ssm_state(shadow->id);
        }
        return true;
    });
}

void Engine::cfg_pair_rows_(std::vector<std::shared_ptr<Request>>& rows) {
    if (std::none_of(rows.begin(), rows.end(), [](const auto& r) { return r->cfg != nullptr; }))
        return;
    std::unordered_map<int, const Request*> in_step;
    for (const auto& r : rows)
        in_step.emplace(r->id, r.get());
    // A member runs only beside its partner at the same output length; an owner whose shadow
    // ended (KV exhaustion, cancel) goes on unguided rather than waiting forever.
    std::erase_if(rows, [&](const std::shared_ptr<Request>& r) {
        if (!r->cfg)
            return false;
        const auto p = cfg_partner(*r);
        if (r->cfg->shadow && (!p || cfg_done(*p))) {
            IMP_LOG_WARN("cfg: req %d continues unguided, its shadow ended", r->id);
            r->cfg.reset();
            return false;
        }
        return !p || !in_step.count(p->id) || p->output_tokens.size() != r->output_tokens.size();
    });
}

void Engine::cfg_combine_rows_(const std::vector<std::shared_ptr<Request>>& rows, const Tensor& logits,
                               cudaStream_t stream) const {
    if (logits.data == nullptr ||
        std::none_of(rows.begin(), rows.end(), [](const auto& r) { return r->cfg; }))
        return;
    const int64_t v = logits.shape[logits.ndim - 1];
    auto* base = static_cast<float*>(logits.data);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (!rows[i]->cfg || !rows[i]->cfg->shadow)
            continue;
        for (size_t j = 0; j < rows.size(); ++j)
            if (rows[j] == rows[i]->cfg->shadow)
                cfg_combine(base + static_cast<int64_t>(i) * v, base + static_cast<int64_t>(j) * v,
                            rows[i]->cfg->scale, static_cast<int>(v), stream);
    }
}

void Engine::cfg_mirror_tokens_(const std::vector<std::shared_ptr<Request>>& rows,
                                std::vector<int32_t>& tokens) const {
    for (size_t j = 0; j < rows.size() && j < tokens.size(); ++j) {
        if (!rows[j]->cfg || rows[j]->cfg->shadow)
            continue;  // not a shadow
        const auto owner = rows[j]->cfg->owner.lock();
        for (size_t i = 0; owner && i < rows.size(); ++i)
            if (rows[i] == owner)
                tokens[j] = tokens[i];
    }
}

void Engine::cfg_prefill_logits_(Request& req, const Tensor& logits, cudaStream_t stream) {
    if (!req.cfg || logits.data == nullptr)
        return;
    const auto v = static_cast<size_t>(logits.shape[logits.ndim - 1]);
    if (!req.cfg->shadow) {  // shadow: keep its last-row logits for the owner's first token
        req.cfg->first_logits.resize(v);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(req.cfg->first_logits.data(), logits.data, v * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        req.cfg->ready = true;
        return;
    }
    const auto& ul = req.cfg->shadow->cfg->first_logits;
    if (ul.size() != v)
        return;
    if (cfg_scratch_.count() < v)
        cfg_scratch_ = VramOwned<float>(vram_alloc_, v, "cfg_uncond_logits");
    if (!cfg_scratch_)
        return;
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(cfg_scratch_.get(), ul.data(), v * sizeof(float), cudaMemcpyHostToDevice, stream));
    cfg_combine(static_cast<float*>(logits.data), cfg_scratch_.get(), req.cfg->scale, static_cast<int>(v),
                stream);
}

void Engine::cfg_first_token_(Request& req) {
    // The owner's first token replaces the one the shadow's own prefill sampled.
    if (req.cfg && req.cfg->shadow && !req.output_tokens.empty())
        req.cfg->shadow->output_tokens.assign(1, req.output_tokens.front());
}

}  // namespace imp
