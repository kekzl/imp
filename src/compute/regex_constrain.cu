#include "compute/regex_constrain.h"
#include "core/logging.h"

#include <cfloat>
#include <set>

namespace imp {

namespace {

// Drive every disallowed logit to -inf. Ids at or past `n_classified` are the
// lm_head padding SafeTensors models carry beyond the tokenizer vocabulary
// (Qwen3-8B-NVFP4: 151936 vs 151669) — they have no token text, so they are
// masked wholesale rather than indexed.
__global__ void regex_mask_kernel(float* __restrict__ logits, const uint8_t* __restrict__ allow,
                                  int vocab_size, int n_classified) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab_size)
        return;
    if (i >= n_classified || allow[i] == 0)
        logits[i] = -FLT_MAX;
}

}  // namespace

RegexConstrainer::~RegexConstrainer() {
    if (d_token_allow_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_allow_));
        d_token_allow_ = nullptr;
    }
}

// The shared engine is permissive about syntax it cannot honour: `(?=x)` parses
// as an ordinary group and `^`/`$`/`\b` as literals, so a pattern using them
// would be enforced as something the caller never asked for. Refuse those here
// instead — a wrong grammar is worse than no grammar. (Not fixed inside
// RegexNfa on purpose: it also backs JSON-Schema `pattern`, and tightening it
// there is a separate, testable change.)
static const char* unsupported_construct(const std::string& p) {
    bool in_class = false;  // inside [...] the metacharacters are literals
    for (size_t i = 0; i < p.size(); i++) {
        if (in_class) {
            if (p[i] == '\\')
                i++;  // an escaped ] does not close the class
            else if (p[i] == ']')
                in_class = false;
            continue;
        }
        if (p[i] == '[') {
            in_class = true;
            // A leading ^ negates the class; a leading ] is a literal.
            if (i + 1 < p.size() && p[i + 1] == '^')
                i++;
            if (i + 1 < p.size() && p[i + 1] == ']')
                i++;
            continue;
        }
        if (p[i] == '\\') {
            if (i + 1 < p.size() && (p[i + 1] == 'b' || p[i + 1] == 'B'))
                return "word boundaries (\\b)";
            if (i + 1 < p.size() && p[i + 1] >= '1' && p[i + 1] <= '9')
                return "backreferences";
            i++;  // skip the escaped char
            continue;
        }
        if (p[i] == '^' || p[i] == '$')
            return "anchors (the whole output must match, so they are implicit)";
        if (p[i] == '(' && i + 1 < p.size() && p[i + 1] == '?') {
            // (?: ... ) is a plain non-capturing group and is fine.
            if (i + 2 < p.size() && p[i + 2] == ':')
                continue;
            return "lookaround / named groups";
        }
    }
    return nullptr;
}

bool RegexConstrainer::init_pattern_only(const std::string& pattern) {
    // The manager is pooled and reused across requests, so a new pattern lands
    // in the SAME object. Without clearing here, the cached masks of the
    // PREVIOUS pattern stay live and are served for state sets of the new one —
    // which showed up as one token repeating forever, because the stale mask
    // allowed a token the new FSM then refused to advance on.
    mask_cache_.clear();
    initialized_ = false;
    pattern_ = pattern;
    if (const char* bad = unsupported_construct(pattern)) {
        IMP_LOG_WARN("RegexConstrainer: pattern '%s' uses %s — not enforcing it", pattern.c_str(), bad);
        return false;
    }
    if (!nfa_.compile(pattern)) {
        IMP_LOG_WARN("RegexConstrainer: unsupported or malformed pattern '%s'", pattern.c_str());
        return false;
    }
    states_ = nfa_.start_set();

    // A pattern that cannot match anything must be refused here, not discovered
    // at decode time: the empty-allow guard would fire on the first token, the
    // request would return an empty string, and the caller would have no idea
    // why. RegexNfa accepts some of these (`a{2,1}` compiles), so check the
    // language is non-empty by searching for a reachable accepting state.
    if (!language_non_empty()) {
        IMP_LOG_WARN("RegexConstrainer: pattern '%s' matches nothing — not enforcing it", pattern.c_str());
        return false;
    }
    initialized_ = true;
    return true;
}

// Breadth-first over NFA state sets. Bounded: a pattern needing more than this
// to reach an accepting state is not something we want to enforce blindly
// either, and the bound keeps a pathological pattern from hanging the caller.
bool RegexConstrainer::language_non_empty() const {
    constexpr int kMaxDepth = 64;
    constexpr size_t kMaxSets = 4096;
    if (nfa_.accepts(states_))
        return true;
    std::set<std::vector<int>> seen{states_};
    std::vector<std::vector<int>> frontier{states_};
    for (int depth = 0; depth < kMaxDepth && !frontier.empty(); depth++) {
        std::vector<std::vector<int>> next;
        for (const auto& cur : frontier) {
            for (int b = 0; b < 256; b++) {
                std::vector<int> ns = nfa_.step(cur, static_cast<unsigned char>(b));
                if (ns.empty())
                    continue;
                if (nfa_.accepts(ns))
                    return true;
                if (seen.size() < kMaxSets && seen.insert(ns).second)
                    next.push_back(std::move(ns));
            }
        }
        frontier.swap(next);
    }
    return false;
}

bool RegexConstrainer::init(const std::string& pattern, Tokenizer* tokenizer, int vocab_size) {
    if (!tokenizer)
        return false;
    if (!init_pattern_only(pattern))
        return false;

    vocab_size_ = tokenizer->vocab_size();
    token_texts_.resize(vocab_size_);
    for (int i = 0; i < vocab_size_; i++)
        token_texts_[i] = tokenizer->decode_token(static_cast<int32_t>(i));
    eos_ids_ = tokenizer->eos_ids();

    if (cudaMalloc(&d_token_allow_, vocab_size) != cudaSuccess) {
        IMP_LOG_ERROR("RegexConstrainer: failed to allocate the device allow mask");
        d_token_allow_ = nullptr;
        initialized_ = false;
        return false;
    }
    IMP_LOG_INFO("RegexConstrainer: pattern '%s' active over %d tokens", pattern.c_str(), vocab_size_);
    return true;
}

void RegexConstrainer::reset() {
    if (!nfa_.compiled())
        return;
    states_ = nfa_.start_set();
    preamble_.reset();
}

bool RegexConstrainer::is_done() const { return initialized_ && nfa_.accepts(states_); }

bool RegexConstrainer::would_accept(const std::string& text) const {
    if (!initialized_)
        return true;
    std::vector<int> s = states_;
    for (unsigned char c : text) {
        s = nfa_.step(s, c);
        if (s.empty())
            return false;
    }
    return true;
}

bool RegexConstrainer::update_text(const std::string& text) {
    if (!initialized_)
        return true;
    std::vector<int> s = states_;
    for (unsigned char c : text) {
        s = nfa_.step(s, c);
        if (s.empty())
            return false;
    }
    states_ = std::move(s);
    return true;
}

bool RegexConstrainer::update(int32_t token_id) {
    if (!initialized_)
        return true;
    if (token_id < 0 || token_id >= static_cast<int32_t>(token_texts_.size()))
        return true;
    const std::string& text = token_texts_[static_cast<size_t>(token_id)];
    // While the preamble gate is open (a reasoning model's <think> block), the
    // tokens belong to the preamble, not to the pattern.
    if (preamble_.absorb(token_id, text))
        return true;
    for (int32_t eid : eos_ids_)
        if (token_id == eid)
            return true;  // stopping is governed by is_done(), not the FSM
    return update_text(text);
}

const std::vector<uint8_t>& RegexConstrainer::allow_for_current_state(int vocab_size) {
    auto it = mask_cache_.find(states_);
    if (it != mask_cache_.end())
        return it->second;

    // Cold for this state set: walk the vocabulary once. Typical patterns visit
    // few distinct sets, so this warms up quickly and later steps are lookups.
    std::vector<uint8_t> allow(static_cast<size_t>(vocab_size), 0);
    const int n_classified = std::min(vocab_size, vocab_size_);
    size_t n_allowed = 0;
    for (int i = 0; i < n_classified; i++) {
        const std::string& text = token_texts_[static_cast<size_t>(i)];
        if (text.empty())
            continue;
        if (would_accept(text)) {
            allow[static_cast<size_t>(i)] = 1;
            n_allowed++;
        }
    }

    // EOS only from an accepting state: otherwise the model could stop halfway
    // through the format and still look "finished" to the caller.
    if (nfa_.accepts(states_)) {
        for (int32_t eid : eos_ids_)
            if (eid >= 0 && eid < vocab_size) {
                allow[static_cast<size_t>(eid)] = 1;
                n_allowed++;
            }
    }

    // Nothing at all passes: with every logit at -inf, greedy argmax collapses
    // to token 0 and the output degenerates into "!!!!". Let it stop cleanly
    // instead — the same guard JsonConstrainer carries.
    if (n_allowed == 0) {
        for (int32_t eid : eos_ids_)
            if (eid >= 0 && eid < vocab_size)
                allow[static_cast<size_t>(eid)] = 1;
        IMP_LOG_WARN("RegexConstrainer: no token continues pattern '%s' — allowing EOS", pattern_.c_str());
    }
    return mask_cache_.emplace(states_, std::move(allow)).first->second;
}

void RegexConstrainer::apply_mask(float* d_logits, int vocab_size, cudaStream_t stream) {
    if (!initialized_ || !d_token_allow_)
        return;
    if (preamble_.active())
        return;

    const std::vector<uint8_t>& allow = allow_for_current_state(vocab_size);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_token_allow_, allow.data(), static_cast<size_t>(vocab_size),
                                       cudaMemcpyHostToDevice, stream));

    const int threads = 256;
    const int blocks = (vocab_size + threads - 1) / threads;
    regex_mask_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_token_allow_, vocab_size,
                                                      std::min(vocab_size, vocab_size_));
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
