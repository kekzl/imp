#include "compute/grammar_constrain.h"
#include "core/logging.h"

#include <cfloat>

namespace imp {

namespace {

// Drive every disallowed logit to -inf. Ids at or past `n_classified` are the
// lm_head padding SafeTensors models carry beyond the tokenizer vocabulary —
// they have no token text, so they are masked wholesale rather than indexed.
__global__ void grammar_mask_kernel(float* __restrict__ logits, const uint8_t* __restrict__ allow,
                                    int vocab_size, int n_classified) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab_size)
        return;
    if (i >= n_classified || allow[i] == 0)
        logits[i] = -FLT_MAX;
}

}  // namespace

GrammarConstrainer::~GrammarConstrainer() = default;

bool GrammarConstrainer::init_grammar_only(const std::string& gbnf) {
    // The manager is pooled and reused across requests, so a new grammar lands
    // in the SAME object. Without clearing here, the cached masks of the
    // PREVIOUS grammar stay live and get served for state sets of the new one.
    mask_cache_.clear();
    initialized_ = false;
    error_.clear();
    source_ = gbnf;

    if (!matcher_.compile(gbnf, &error_)) {
        IMP_LOG_WARN("GrammarConstrainer: %s — not enforcing this grammar", error_.c_str());
        return false;
    }
    initialized_ = true;
    return true;
}

bool GrammarConstrainer::init(const std::string& gbnf, Tokenizer* tokenizer, int vocab_size) {
    if (!tokenizer)
        return false;
    if (!init_grammar_only(gbnf))
        return false;

    vocab_size_ = tokenizer->vocab_size();
    token_texts_.resize(static_cast<size_t>(vocab_size_));
    for (int i = 0; i < vocab_size_; i++)
        token_texts_[static_cast<size_t>(i)] = tokenizer->decode_token(static_cast<int32_t>(i));
    eos_ids_ = tokenizer->eos_ids();

    if (!dev_.alloc_token_allow("GrammarConstrainer", vocab_size)) {
        initialized_ = false;
        return false;
    }
    IMP_LOG_INFO("GrammarConstrainer: grammar with %zu rules active over %d tokens",
                 matcher_.grammar().rules().size(), vocab_size_);
    return true;
}

void GrammarConstrainer::reset() {
    if (!matcher_.compiled())
        return;
    matcher_.reset();
    preamble_.reset();
}

bool GrammarConstrainer::update(int32_t token_id) {
    if (!initialized_)
        return true;
    if (token_id < 0 || token_id >= static_cast<int32_t>(token_texts_.size()))
        return true;
    const std::string& text = token_texts_[static_cast<size_t>(token_id)];
    // While the preamble gate is open (a reasoning model's <think> block), the
    // tokens belong to the preamble, not to the grammar.
    if (preamble_.absorb(token_id, text))
        return true;
    for (int32_t eid : eos_ids_)
        if (token_id == eid)
            return true;  // stopping is governed by is_done(), not the grammar
    return matcher_.update_text(text);
}

const std::vector<uint8_t>& GrammarConstrainer::allow_for_current_state(int vocab_size) {
    std::vector<int32_t> key = matcher_.state_key();
    auto it = mask_cache_.find(key);
    if (it != mask_cache_.end())
        return it->second;

    // Cold for this state: walk the vocabulary once. Simulating every token
    // through the pushdown would cost more than decoding does, so reject on the
    // first byte first — that kills the overwhelming majority in O(1).
    uint8_t lead[256];
    matcher_.lead_bytes(lead);

    std::vector<uint8_t> allow(static_cast<size_t>(vocab_size), 0);
    const int n_classified = std::min(vocab_size, vocab_size_);
    size_t n_allowed = 0;
    for (int i = 0; i < n_classified; i++) {
        const std::string& text = token_texts_[static_cast<size_t>(i)];
        if (text.empty())
            continue;
        if (!lead[static_cast<unsigned char>(text[0])])
            continue;
        if (matcher_.would_accept(text)) {
            allow[static_cast<size_t>(i)] = 1;
            n_allowed++;
        }
    }

    // EOS only from a complete derivation: otherwise the model could stop
    // halfway through the format and still look "finished" to the caller.
    if (matcher_.is_done()) {
        for (int32_t eid : eos_ids_)
            if (eid >= 0 && eid < vocab_size) {
                allow[static_cast<size_t>(eid)] = 1;
                n_allowed++;
            }
    }

    // Nothing at all passes: with every logit at -inf, greedy argmax collapses
    // to token 0 and the output degenerates into "!!!!". Let it stop cleanly
    // instead — the same guard the other constrainers carry.
    if (n_allowed == 0) {
        for (int32_t eid : eos_ids_)
            if (eid >= 0 && eid < vocab_size)
                allow[static_cast<size_t>(eid)] = 1;
        IMP_LOG_WARN("GrammarConstrainer: no token continues the grammar — allowing EOS");
    }
    return mask_cache_.emplace(std::move(key), std::move(allow)).first->second;
}

void GrammarConstrainer::apply_mask(float* d_logits, int vocab_size, cudaStream_t stream) {
    if (!initialized_ || !dev_.has_token_allow())
        return;
    if (preamble_.active())
        return;

    const std::vector<uint8_t>& allow = allow_for_current_state(vocab_size);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dev_.token_allow(), allow.data(), static_cast<size_t>(vocab_size),
                                       cudaMemcpyHostToDevice, stream));

    const int threads = 256;
    const int blocks = (vocab_size + threads - 1) / threads;
    grammar_mask_kernel<<<blocks, threads, 0, stream>>>(d_logits, dev_.token_allow(), vocab_size,
                                                        std::min(vocab_size, vocab_size_));
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
