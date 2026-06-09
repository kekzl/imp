#include "compute/schema_constrain.h"
#include "compute/json_constrain.h"  // reuse token category definitions
#include "compute/constrain_common.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <algorithm>
#include <cstring>

namespace imp {

// Max simulated frame-stack depth. Each '{'/'[' nesting level holds ~2 frames
// (container frame + value frame), so 192 frames ~= 96 nesting levels. Only
// reachable via recursive $ref schemas; hitting the cap forces closure (still
// schema-valid — any finite nesting satisfies a recursive schema).
static constexpr size_t kMaxSchemaStackDepth = 192;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

SchemaConstrainer::~SchemaConstrainer() {
    if (d_token_categories_)
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_categories_));
    if (d_token_allow_)
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_allow_));
    if (d_allowed_mask_)
        IMP_CUDA_CHECK_LOG(cudaFree(d_allowed_mask_));
}

bool SchemaConstrainer::init(const Tokenizer& tok, std::unique_ptr<SchemaNode> schema) {
    schema_ = std::move(schema);
    if (!schema_)
        return false;

    vocab_size_ = tok.vocab_size();
    eos_tokens_ = tok.eos_ids();

    // Classify all tokens (same logic as JsonConstrainer)
    token_categories_.resize(vocab_size_, 0);
    token_texts_.resize(vocab_size_);
    token_allow_.resize(vocab_size_, 1);

    for (int i = 0; i < vocab_size_; i++) {
        std::string text = tok.decode_token(i);
        token_texts_[i] = text;
        token_categories_[i] = classify_token(text);
    }

    // Upload to GPU
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_token_categories_, vocab_size_ * sizeof(uint16_t)));
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_token_categories_, token_categories_.data(),
                                  vocab_size_ * sizeof(uint16_t), cudaMemcpyHostToDevice));

    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_token_allow_, vocab_size_ * sizeof(uint8_t)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_allowed_mask_, sizeof(uint16_t)));

    reset();
    initialized_ = true;

    IMP_LOG_INFO("SchemaConstrainer: initialized with %d tokens, schema type=%d", vocab_size_,
                 static_cast<int>(schema_->type));
    return true;
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void SchemaConstrainer::reset() {
    stack_.clear();
    push_value_frame(schema_.get());
    need_token_allow_ = false;
    std::fill(token_allow_.begin(), token_allow_.end(), (uint8_t)1);
    preamble_.reset();
}

void SchemaConstrainer::push_value_frame(const SchemaNode* node) {
    SchemaFrame frame;
    // Frames always hold RESOLVED nodes: $ref indirection ends here, so the
    // phase machine below never sees SchemaType::REF.
    frame.node = resolve_schema_ref(schema_.get(), node);
    frame.phase = SchemaPhase::VALUE_START;
    stack_.push_back(std::move(frame));
}

// ---------------------------------------------------------------------------
// Property / enum helpers
// ---------------------------------------------------------------------------

const SchemaNode* SchemaConstrainer::find_property(const SchemaNode* obj, const std::string& key) const {
    for (auto& [name, schema] : obj->properties) {
        if (name == key)
            return resolve_schema_ref(schema_.get(), schema.get());
    }
    return nullptr;
}

bool SchemaConstrainer::is_valid_key_prefix(const SchemaNode* obj, const std::string& prefix,
                                            const std::set<std::string>& emitted) const {
    for (auto& [name, _] : obj->properties) {
        if (emitted.count(name))
            continue;
        if (name.size() >= prefix.size() && name.compare(0, prefix.size(), prefix) == 0)
            return true;
    }
    return false;
}

bool SchemaConstrainer::is_valid_enum_prefix(const std::vector<std::string>& values,
                                             const std::string& prefix) const {
    for (auto& v : values) {
        if (v.size() >= prefix.size() && v.compare(0, prefix.size(), prefix) == 0)
            return true;
    }
    return false;
}

bool SchemaConstrainer::token_keeps_pattern_alive(const SchemaFrame& f, const std::string& content,
                                                  std::vector<int>& out_states, int& out_len) const {
    out_states = f.regex_states;
    out_len = f.string_len;

    bool has_nfa = f.node && f.node->pattern_nfa && f.node->pattern_nfa->compiled();
    int max_len = f.node ? f.node->max_length : -1;

    for (char ch : content) {
        // Reject control chars / quote / backslash inside a constrained string.
        unsigned char uc = static_cast<unsigned char>(ch);
        if (uc < 32 || ch == '"' || ch == '\\')
            return false;
        if (max_len >= 0 && out_len + 1 > max_len)
            return false;
        if (has_nfa) {
            out_states = f.node->pattern_nfa->step(out_states, uc);
            if (out_states.empty())
                return false;  // prefix died -> no string can complete
        }
        out_len++;
    }
    return true;
}

bool SchemaConstrainer::can_close_string(const SchemaFrame& f, const std::vector<int>& states,
                                         int len) const {
    bool has_nfa = f.node && f.node->pattern_nfa && f.node->pattern_nfa->compiled();
    if (has_nfa && !f.node->pattern_nfa->accepts(states))
        return false;
    if (f.node && f.node->min_length >= 0 && len < f.node->min_length)
        return false;
    if (f.node && f.node->max_length >= 0 && len > f.node->max_length)
        return false;
    return true;
}

// ---------------------------------------------------------------------------
// Compute category mask from schema FSM state
// ---------------------------------------------------------------------------

uint16_t SchemaConstrainer::compute_category_mask() const {
    if (stack_.empty())
        return CAT_WHITESPACE;

    const auto& f = top();
    switch (f.phase) {
        case SchemaPhase::VALUE_START: {
            // No insignificant whitespace: a reasoning model whose top token is
            // a newline would otherwise stall forever (whitespace is always
            // re-allowed, never forcing structural progress). Compact JSON only
            // — whitespace *inside* string values is CAT_STRING_CHAR, untouched.
            uint16_t mask = 0;
            if (!f.node)
                return mask | CAT_VALUE_START;
            switch (f.node->type) {
                case SchemaType::OBJECT:
                    mask |= CAT_OPEN_BRACE;
                    break;
                case SchemaType::ARRAY:
                    mask |= CAT_OPEN_BRACKET;
                    break;
                case SchemaType::STRING:
                    mask |= CAT_QUOTE;
                    break;
                case SchemaType::NUMBER:
                case SchemaType::INTEGER:
                    mask |= CAT_NUMBER_START;
                    break;
                case SchemaType::BOOLEAN:
                    mask |= CAT_TRUE_START | CAT_FALSE_START;
                    break;
                case SchemaType::NULL_TYPE:
                    mask |= CAT_NULL_START;
                    break;
                case SchemaType::ENUM:
                    mask |= CAT_QUOTE;
                    break;  // enum values are strings
                case SchemaType::ANY_OF:
                    mask |= CAT_VALUE_START;
                    break;
                default:
                    mask |= CAT_VALUE_START;
                    break;
            }
            return mask;
        }

        case SchemaPhase::OBJECT_OPEN: {
            uint16_t mask = CAT_QUOTE;  // " for key (compact JSON, no whitespace)
            // Allow } only if all required keys are emitted
            bool all_required = true;
            if (f.node) {
                for (auto& req : f.node->required) {
                    if (!f.emitted_keys.count(req)) {
                        all_required = false;
                        break;
                    }
                }
            }
            if (all_required)
                mask |= CAT_CLOSE_BRACE;
            return mask;
        }

        case SchemaPhase::OBJECT_KEY:
            return CAT_STRING_CHAR | CAT_QUOTE;  // token_allow handles key constraining

        case SchemaPhase::OBJECT_AFTER_KEY:
            return CAT_COLON;

        case SchemaPhase::OBJECT_COLON:
            return CAT_COLON;

        case SchemaPhase::OBJECT_AFTER_VALUE: {
            uint16_t mask = CAT_COMMA;
            bool all_required = true;
            if (f.node) {
                for (auto& req : f.node->required) {
                    if (!f.emitted_keys.count(req)) {
                        all_required = false;
                        break;
                    }
                }
            }
            if (all_required)
                mask |= CAT_CLOSE_BRACE;
            return mask;
        }

        case SchemaPhase::ARRAY_OPEN: {
            uint16_t mask = CAT_CLOSE_BRACKET;
            // Allow value start for first item ($ref items resolve first)
            const SchemaNode* items =
                f.node ? resolve_schema_ref(schema_.get(), f.node->items.get()) : nullptr;
            if (items) {
                switch (items->type) {
                    case SchemaType::STRING:
                        mask |= CAT_QUOTE;
                        break;
                    case SchemaType::NUMBER:
                    case SchemaType::INTEGER:
                        mask |= CAT_NUMBER_START;
                        break;
                    case SchemaType::BOOLEAN:
                        mask |= CAT_TRUE_START | CAT_FALSE_START;
                        break;
                    case SchemaType::OBJECT:
                        mask |= CAT_OPEN_BRACE;
                        break;
                    case SchemaType::ARRAY:
                        mask |= CAT_OPEN_BRACKET;
                        break;
                    default:
                        mask |= CAT_VALUE_START;
                        break;
                }
            } else {
                mask |= CAT_VALUE_START;
            }
            return mask;
        }

        case SchemaPhase::ARRAY_AFTER_ITEM:
            return CAT_COMMA | CAT_CLOSE_BRACKET;

        case SchemaPhase::STRING_VALUE:
            return CAT_STRING_CHAR | CAT_QUOTE;

        case SchemaPhase::STRING_PATTERN:
            // token_allow enforces the regex / length; category just gates to
            // string content + closing quote.
            return CAT_STRING_CHAR | CAT_QUOTE;

        case SchemaPhase::STRING_ESCAPE:
            return 0xFFFF;  // any char valid after backslash

        case SchemaPhase::NUMBER_VALUE:
            return CAT_NUMBER_CONT | CAT_COMMA | CAT_CLOSE_BRACE | CAT_CLOSE_BRACKET;

        case SchemaPhase::LITERAL_VALUE:
            return CAT_LITERAL_CONT;

        case SchemaPhase::ENUM_VALUE:
            return CAT_STRING_CHAR | CAT_QUOTE;

        case SchemaPhase::DONE:
            return CAT_WHITESPACE;
    }
    return 0xFFFF;
}

// ---------------------------------------------------------------------------
// Compute per-token allow mask (for key names and enum values)
// ---------------------------------------------------------------------------

void SchemaConstrainer::compute_token_allow_mask(uint16_t cat_mask) {
    need_token_allow_ = false;

    if (stack_.empty())
        return;

    // Full per-token legality: a candidate token is allowed only if simulating
    // its entire text from the current FSM state stays legal at every char.
    // This is what the first-char category mask cannot do — it catches
    // multi-char tokens that span phase transitions (`{}` closing an object
    // with unmet required keys, `":"` as a bogus enum value, `"Why` opening a
    // non-existent key, `0.98` for an integer). The category mask still runs
    // alongside (it governs EOS / whitespace / structural first-char), so a
    // token must pass BOTH. token_legal() handles empty (EOS/special) tokens by
    // deferring to the category mask.
    //
    // Cost control (this loop runs per decode step over the whole vocab, and
    // token_legal deep-copies the frame stack per candidate — it dominated
    // json_schema decode at 151k tokens):
    //  - category prefilter: the kernel ANDs category and allow, so a token
    //    that fails the category mask is masked regardless of allow — skip
    //    its simulation entirely. In structural phases the category mask is
    //    narrow and this eliminates almost all simulations.
    //  - in-string O(1) shortcut (mirrors JsonConstrainer): in a free string
    //    value, any token without '"', '\\' or a raw control char stays
    //    inside the string by construction — sim_advance would accept every
    //    char without touching the stack, so skip the simulation. Pattern /
    //    enum / key strings still simulate (prefix & regex constraints).
    need_token_allow_ = true;
    const bool free_string = (top().phase == SchemaPhase::STRING_VALUE);
    for (int i = 0; i < vocab_size_; i++) {
        if ((token_categories_[i] & cat_mask) == 0) {
            token_allow_[i] = 0;  // masked by category — simulation irrelevant
            continue;
        }
        const std::string& text = token_texts_[i];
        if (free_string && !text.empty()) {
            bool plain = true;
            for (char c : text) {
                if (c == '"' || c == '\\' || static_cast<unsigned char>(c) < 0x20) {
                    plain = false;
                    break;
                }
            }
            if (plain) {
                token_allow_[i] = 1;
                continue;
            }
        }
        token_allow_[i] = token_legal(text) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// Apply mask to logits
// ---------------------------------------------------------------------------

void SchemaConstrainer::apply_mask(float* d_logits, int vocab_size, cudaStream_t stream) {
    if (!initialized_)
        return;

    if (preamble_.active())
        return;

    // Root value complete (stack drained): force EOS so generation stops cleanly
    // instead of trailing free text after the closing brace/bracket. Mask
    // everything except the EOS token(s) via the allow path (category = all).
    if (stack_.empty()) {
        if (eos_tokens_.empty())
            return;  // no EOS to force — leave the model unconstrained
        std::fill(token_allow_.begin(), token_allow_.end(), (uint8_t)0);
        for (int32_t e : eos_tokens_)
            if (e >= 0 && e < vocab_size_)
                token_allow_[e] = 1;
        uint16_t all_cats = 0xFFFF;
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_allowed_mask_, &all_cats, sizeof(uint16_t),
                                           cudaMemcpyHostToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_token_allow_, token_allow_.data(),
                                           vocab_size_ * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        int t = 256, b = (vocab_size + t - 1) / t;
        constrain_mask_allow_kernel<<<b, t, 0, stream>>>(d_logits, d_token_categories_, d_token_allow_,
                                                         d_allowed_mask_, vocab_size,
                                                         /*n_classified=*/vocab_size_, /*use_allow=*/true);
        return;
    }

    // Compute masks
    uint16_t cat_mask = compute_category_mask();
    compute_token_allow_mask(cat_mask);

    // Empty-allow guard: an over-tight schema/state combination (e.g.
    // {"type":"object"} without properties — the key phase knows no legal
    // key) can reject every token. All logits then go to -FLT_MAX and greedy
    // argmax degenerates to token id 0 ("!!!!" spam on byte-level BPE
    // vocabs). Force a clean EOS finish instead.
    if (need_token_allow_) {
        size_t n_allowed = 0;
        for (int i = 0; i < vocab_size_ && n_allowed == 0; i++) {
            if (token_allow_[i] && (token_categories_[i] & cat_mask) != 0)
                n_allowed++;
        }
        if (n_allowed == 0) {
            std::fill(token_allow_.begin(), token_allow_.end(), (uint8_t)0);
            for (int32_t e : eos_tokens_)
                if (e >= 0 && e < vocab_size_)
                    token_allow_[e] = 1;
            cat_mask = 0xFFFF;
            IMP_LOG_WARN("SchemaConstrainer: no token satisfies the schema in phase %d — allowing EOS",
                         static_cast<int>(top().phase));
        }
    }

    IMP_LOG_DEBUG("SchemaConstrainer::apply_mask phase=%d cat_mask=0x%04x need_allow=%d stack=%zu",
                  static_cast<int>(top().phase), cat_mask, need_token_allow_, stack_.size());

    // Upload category mask
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_allowed_mask_, &cat_mask, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));

    // Upload token allow mask if needed
    if (need_token_allow_) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_token_allow_, token_allow_.data(), vocab_size_ * sizeof(uint8_t),
                                           cudaMemcpyHostToDevice, stream));
    }

    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    // vocab_size is the LOGITS width (model vocab); the category/allow buffers
    // only cover the tokenizer vocab — padding logits are masked via
    // n_classified (SafeTensors lm_head padding, see json_constrain.cu).
    constrain_mask_allow_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_token_categories_, d_token_allow_,
                                                                d_allowed_mask_, vocab_size,
                                                                /*n_classified=*/vocab_size_,
                                                                need_token_allow_);
}

// ---------------------------------------------------------------------------
// Update FSM with sampled token
// ---------------------------------------------------------------------------

void SchemaConstrainer::update(int32_t token) {
    if (token < 0 || token >= vocab_size_ || stack_.empty())
        return;

    const auto& text = token_texts_[token];
    if (preamble_.absorb(token, text))
        return;
    SchemaPhase before = top().phase;
    for (char c : text) {
        if (!sim_advance(stack_, c))
            break;  // illegal char (mask should have prevented this) — stop early
    }
    if (!stack_.empty()) {
        IMP_LOG_DEBUG("SchemaConstrainer::update token=%d [%s] phase %d->%d stack=%zu", token, text.c_str(),
                      static_cast<int>(before), static_cast<int>(top().phase), stack_.size());
    }
}

// After a value frame pops, advance the parent frame's phase. Shared by the
// transition simulator below.
static void sim_fixup_parent(std::vector<SchemaFrame>& stk) {
    if (stk.empty())
        return;
    SchemaFrame& parent = stk.back();
    if (parent.phase == SchemaPhase::OBJECT_COLON)
        parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
    else if (parent.phase == SchemaPhase::ARRAY_OPEN ||
             parent.phase == SchemaPhase::ARRAY_AFTER_ITEM)
        parent.phase = SchemaPhase::ARRAY_AFTER_ITEM;
}

// ---------------------------------------------------------------------------
// Transition simulator — the single source of truth for the schema grammar.
// Drives the real update path (on stack_) and per-token mask legality (on a
// cloned stack). Returns false on any illegal transition, so a multi-char
// token that spans phase transitions (`{}`, `":"`, `"Why`, integer `0.98`,
// trailing comma) is rejected as a whole rather than slipping past the
// first-char category mask.
// ---------------------------------------------------------------------------
bool SchemaConstrainer::sim_advance(std::vector<SchemaFrame>& stk, char c) const {
    if (stk.empty())
        return false;  // trailing content after the root value completed

    SchemaFrame& f = stk.back();
    auto push_value = [&](const SchemaNode* node) {
        SchemaFrame nf;
        nf.node = resolve_schema_ref(schema_.get(), node);
        nf.phase = SchemaPhase::VALUE_START;
        stk.push_back(std::move(nf));
    };
    auto required_satisfied = [](const SchemaFrame& fr) {
        if (!fr.node)
            return true;
        for (auto& req : fr.node->required)
            if (!fr.emitted_keys.count(req))
                return false;
        return true;
    };
    auto has_unemitted_property = [](const SchemaFrame& fr) {
        if (!fr.node)
            return true;  // unknown object — can't tell, allow another key
        for (auto& [name, _] : fr.node->properties)
            if (!fr.emitted_keys.count(name))
                return true;
        return false;
    };
    const bool space = std::isspace(static_cast<unsigned char>(c)) != 0;

    switch (f.phase) {
        case SchemaPhase::VALUE_START: {
            if (space)
                return true;
            if (!f.node) {
                stk.pop_back();  // unconstrained value — accept opaquely
                return true;
            }
            switch (f.node->type) {
                case SchemaType::OBJECT:
                    // Depth cap: recursive $ref schemas allow unbounded nesting;
                    // refuse to open deeper than ~96 levels (any finite nesting
                    // still satisfies the schema, so this only forces closure).
                    if (c == '{') {
                        if (stk.size() >= kMaxSchemaStackDepth)
                            return false;
                        f.phase = SchemaPhase::OBJECT_OPEN;
                        return true;
                    }
                    return false;
                case SchemaType::ARRAY:
                    if (c == '[') {
                        if (stk.size() >= kMaxSchemaStackDepth)
                            return false;
                        f.phase = SchemaPhase::ARRAY_OPEN;
                        return true;
                    }
                    return false;
                case SchemaType::STRING:
                    if (c == '"') {
                        if ((f.node->pattern_nfa && f.node->pattern_nfa->compiled()) ||
                            f.node->min_length >= 0 || f.node->max_length >= 0) {
                            f.phase = SchemaPhase::STRING_PATTERN;
                            f.string_len = 0;
                            if (f.node->pattern_nfa && f.node->pattern_nfa->compiled())
                                f.regex_states = f.node->pattern_nfa->start_set();
                            else
                                f.regex_states.clear();
                        } else {
                            f.phase = SchemaPhase::STRING_VALUE;
                        }
                        return true;
                    }
                    return false;
                case SchemaType::NUMBER:
                case SchemaType::INTEGER:
                    if (c == '-' || (c >= '0' && c <= '9')) {
                        f.phase = SchemaPhase::NUMBER_VALUE;
                        f.string_len = (c >= '0' && c <= '9') ? 1 : 0;  // digit count
                        f.num_leading_zero = (c == '0');  // "0..." forbids more int digits
                        return true;
                    }
                    return false;
                case SchemaType::BOOLEAN:
                    if (c == 't') { f.phase = SchemaPhase::LITERAL_VALUE; f.literal_target = "true"; f.literal_pos = 1; return true; }
                    if (c == 'f') { f.phase = SchemaPhase::LITERAL_VALUE; f.literal_target = "false"; f.literal_pos = 1; return true; }
                    return false;
                case SchemaType::NULL_TYPE:
                    if (c == 'n') { f.phase = SchemaPhase::LITERAL_VALUE; f.literal_target = "null"; f.literal_pos = 1; return true; }
                    return false;
                case SchemaType::ENUM:
                    if (c == '"') { f.phase = SchemaPhase::ENUM_VALUE; f.enum_buffer.clear(); return true; }
                    return false;
                case SchemaType::ANY_OF:
                    // anyOf is hard to constrain precisely — accept as free string.
                    f.phase = SchemaPhase::STRING_VALUE;
                    return true;
                default:
                    return false;
            }
        }

        case SchemaPhase::OBJECT_OPEN: {
            if (space)
                return true;
            if (c == '}') {
                // After a comma a key is mandatory — closing here is a trailing
                // comma. Otherwise close only once required keys are present.
                if (f.after_comma || !required_satisfied(f))
                    return false;
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            if (c == '"') {
                f.phase = SchemaPhase::OBJECT_KEY;
                f.key_buffer.clear();
                f.after_comma = false;
                return true;
            }
            return false;
        }

        case SchemaPhase::OBJECT_KEY: {
            if (c == '"') {
                // Close the key only if it is a complete, not-yet-emitted property.
                bool complete = false;
                if (f.node) {
                    for (auto& [name, _] : f.node->properties) {
                        if (!f.emitted_keys.count(name) && name == f.key_buffer) { complete = true; break; }
                    }
                }
                if (!complete)
                    return false;
                f.current_key = f.key_buffer;
                f.emitted_keys.insert(f.current_key);
                f.phase = SchemaPhase::OBJECT_AFTER_KEY;
                return true;
            }
            if (c == '\\')
                return true;  // key escape (rare)
            if (!f.node || !is_valid_key_prefix(f.node, f.key_buffer + c, f.emitted_keys))
                return false;
            f.key_buffer += c;
            return true;
        }

        case SchemaPhase::OBJECT_AFTER_KEY: {
            if (space)
                return true;
            if (c == ':') {
                f.phase = SchemaPhase::OBJECT_COLON;
                const SchemaNode* prop = f.node ? find_property(f.node, f.current_key) : nullptr;
                push_value(prop);
                return true;
            }
            return false;
        }

        case SchemaPhase::OBJECT_COLON: {
            // The value is normally handled by the pushed sub-frame; this branch
            // only fires for robustness if a comma/brace reaches the colon frame.
            if (space)
                return true;
            if (c == ',') {
                if (!has_unemitted_property(f))
                    return false;  // no more keys possible — comma would dangle
                f.phase = SchemaPhase::OBJECT_OPEN;
                f.after_comma = true;
                return true;
            }
            if (c == '}') {
                if (!required_satisfied(f))
                    return false;
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            return false;
        }

        case SchemaPhase::OBJECT_AFTER_VALUE: {
            if (space)
                return true;
            if (c == ',') {
                if (!has_unemitted_property(f))
                    return false;  // every property emitted — comma would dangle
                f.phase = SchemaPhase::OBJECT_OPEN;
                f.after_comma = true;
                return true;
            }
            if (c == '}') {
                if (!required_satisfied(f))
                    return false;
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            return false;
        }

        case SchemaPhase::ARRAY_OPEN: {
            if (space)
                return true;
            if (c == ']') {
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            if (f.node && f.node->items) {
                push_value(f.node->items.get());
                return sim_advance(stk, c);  // process first-item char in new frame
            }
            return true;  // array without an items schema — accept opaquely
        }

        case SchemaPhase::ARRAY_AFTER_ITEM: {
            if (space)
                return true;
            if (c == ',') {
                if (f.node && f.node->items)
                    push_value(f.node->items.get());
                return true;
            }
            if (c == ']') {
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            return false;
        }

        case SchemaPhase::STRING_VALUE: {
            if (c == '\\') { f.phase = SchemaPhase::STRING_ESCAPE; return true; }
            if (c == '"') {
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            // JSON forbids raw control chars (U+0000–U+001F) inside strings —
            // they must arrive escaped (\n, \uXXXX). Multi-char tokens whose
            // first char passes the category mask (e.g. `"<newline>`) used to
            // smuggle them through, producing unparseable output.
            if (static_cast<unsigned char>(c) < 0x20)
                return false;
            return true;  // any content char
        }

        case SchemaPhase::STRING_PATTERN: {
            if (c == '"') {
                if (!can_close_string(f, f.regex_states, f.string_len))
                    return false;
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            if (static_cast<unsigned char>(c) < 0x20)
                return false;  // raw control char — see STRING_VALUE
            if (f.node && f.node->max_length >= 0 && f.string_len + 1 > f.node->max_length)
                return false;
            if (f.node && f.node->pattern_nfa && f.node->pattern_nfa->compiled()) {
                std::vector<int> next =
                    f.node->pattern_nfa->step(f.regex_states, static_cast<unsigned char>(c));
                if (next.empty())
                    return false;  // pattern prefix died
                f.regex_states = std::move(next);
            }
            f.string_len++;
            return true;
        }

        case SchemaPhase::STRING_ESCAPE: {
            if (static_cast<unsigned char>(c) < 0x20)
                return false;  // `\` + raw control char is not a legal escape
            f.phase = SchemaPhase::STRING_VALUE;
            return true;
        }

        case SchemaPhase::NUMBER_VALUE: {
            const bool is_int = f.node && f.node->type == SchemaType::INTEGER;
            if (c >= '0' && c <= '9') {
                if (f.string_len == 0) {  // first digit (came after a leading '-')
                    f.num_leading_zero = (c == '0');
                    f.string_len = 1;
                    return true;
                }
                if (f.num_leading_zero)
                    return false;  // JSON forbids leading zeros: `0` then digit
                f.string_len++;
                return true;
            }
            if (!is_int && (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')) {
                f.num_leading_zero = false;  // fractional/exponent part — int rule done
                return true;
            }
            // Any other char ends the number — legal only if >=1 digit was seen.
            if (f.string_len == 0)
                return false;
            stk.pop_back();
            sim_fixup_parent(stk);
            return sim_advance(stk, c);  // reprocess the terminator in the parent
        }

        case SchemaPhase::LITERAL_VALUE: {
            if (f.literal_pos >= static_cast<int>(f.literal_target.size()))
                return false;
            if (c != f.literal_target[f.literal_pos])
                return false;  // wrong char for true/false/null
            f.literal_pos++;
            if (f.literal_pos >= static_cast<int>(f.literal_target.size())) {
                stk.pop_back();
                sim_fixup_parent(stk);
            }
            return true;
        }

        case SchemaPhase::ENUM_VALUE: {
            if (c == '"') {
                bool exact = false;
                if (f.node) {
                    for (auto& v : f.node->enum_values)
                        if (v == f.enum_buffer) { exact = true; break; }
                }
                if (!exact)
                    return false;  // close only on an exact enum value
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            if (!f.node || !is_valid_enum_prefix(f.node->enum_values, f.enum_buffer + c))
                return false;
            f.enum_buffer += c;
            return true;
        }

        case SchemaPhase::DONE:
            return false;
    }
    return false;
}

bool SchemaConstrainer::token_legal(const std::string& text) const {
    if (text.empty())
        return true;  // EOS / special tokens — governed by the category mask
    std::vector<SchemaFrame> sim = stack_;  // deep copy of the frame stack
    for (char c : text) {
        if (!sim_advance(sim, c))
            return false;
    }
    return true;
}

}  // namespace imp
