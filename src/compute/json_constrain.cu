#include "compute/json_constrain.h"
#include "compute/constrain_common.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstring>
#include <algorithm>
#include <utility>

namespace imp {

// ============================================================================
// JsonConstrainer implementation
// ============================================================================

JsonConstrainer::~JsonConstrainer() {
    if (d_token_categories_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_categories_));
        d_token_categories_ = nullptr;
    }
    if (d_token_allow_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_allow_));
        d_token_allow_ = nullptr;
    }
    if (d_allowed_mask_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_allowed_mask_));
        d_allowed_mask_ = nullptr;
    }
}

bool JsonConstrainer::init(const Tokenizer& tok) {
    vocab_size_ = tok.vocab_size();
    token_categories_.resize(vocab_size_);
    token_texts_.resize(vocab_size_);

    // Classify each token
    for (int i = 0; i < vocab_size_; i++) {
        std::string text = tok.decode_token(static_cast<int32_t>(i));
        token_texts_[i] = text;
        token_categories_[i] = classify_token(text);
    }

    // EOS gets a dedicated category: classify_token sees the rendered text
    // ("<|im_end|>" → STRING_CHAR), so in the DONE state (whitespace-only
    // mask) EOS was blocked and a completed JSON could never terminate —
    // every json_mode request ran to max_tokens and finished with "length".
    // CAT_EOS is allowed ONLY in DONE — allowing it everywhere lets a
    // json-reluctant model emit EOS as its very first token (0 completions).
    eos_ids_ = tok.eos_ids();
    for (int32_t eid : eos_ids_) {
        if (eid >= 0 && eid < vocab_size_)
            token_categories_[eid] = CAT_EOS;  // reachable only where the mask says so
    }

    // Upload to device
    cudaError_t err = cudaMalloc(&d_token_categories_, vocab_size_ * sizeof(uint16_t));
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("JsonConstrainer: failed to allocate device categories: %s", cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(d_token_categories_, token_categories_.data(), vocab_size_ * sizeof(uint16_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("JsonConstrainer: failed to copy categories to device: %s", cudaGetErrorString(err));
        return false;
    }

    // Allocate mask buffer
    err = cudaMalloc(&d_allowed_mask_, sizeof(uint16_t));
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("JsonConstrainer: failed to allocate mask buffer: %s", cudaGetErrorString(err));
        return false;
    }

    reset();
    initialized_ = true;
    IMP_LOG_INFO("JsonConstrainer initialized (%d tokens classified)", vocab_size_);
    return true;
}

void JsonConstrainer::reset() {
    state_stack_.clear();
    current_state_ = JsonState::START;
    partial_literal_.clear();
    target_literal_.clear();
    ws_run_ = 0;
    enter_number_('0');  // clear the number sub-state (see #1104)
    num_need_digit_ = false;
    preamble_.reset();
}

// Seed the RFC 8259 number sub-state on entry. `c` is the first character of
// the number: a bare '-' still owes its first digit, a digit does not.
void JsonConstrainer::enter_number_(char c) {
    num_seen_frac_ = false;
    num_seen_exp_ = false;
    num_exp_sign_ok_ = false;
    num_need_digit_ = (c == '-');
}

uint16_t JsonConstrainer::compute_allowed_mask() const {
    // Whitespace is legal between any JSON tokens, but cap the run length —
    // see advance_char. (EOS has its own CAT_EOS bit, allowed in DONE only.)
    constexpr int kMaxWsRun = 32;

    // Force-close near the budget (#1104). A constrainer can forbid illegal
    // tokens but cannot force termination, so a model inside a long string or
    // a whitespace flood runs to max_tokens and returns a truncated document
    // that no client can parse. Once only just enough tokens remain to shut
    // everything that is open, narrow the mask to exactly the closers. The
    // estimate deliberately errs high (the string's parent frame is already on
    // the stack), because closing a few tokens early is strictly better than
    // returning unparseable output. Disabled while the budget is unknown (-1).
    force_close_active_ = false;
    if (remaining_budget_ >= 0 && current_state_ != JsonState::DONE) {
        const bool in_escape = (current_state_ == JsonState::IN_STRING_ESCAPE);
        const bool in_string = (current_state_ == JsonState::IN_STRING ||
                                current_state_ == JsonState::OBJECT_KEY);
        // state_stack_ holds only the RETURN states of *nested* values, not the
        // container we are currently inside — at `{"a"` the stack is empty
        // while a '}' is still owed. Count that container explicitly, or the
        // narrowing releases one token too early and the document is truncated
        // anyway (observed: needed=0 in AFTER_KEY with an object still open).
        const bool in_container = (current_state_ != JsonState::START && current_state_ != JsonState::DONE);
        const int needed = static_cast<int>(state_stack_.size()) + (in_container ? 1 : 0) +
                           (in_escape   ? 2
                            : in_string ? 1
                                        : 0);
        if (remaining_budget_ <= needed) {
            force_close_active_ = true;
            if (in_escape)
                return CAT_STRING_CHAR;  // finish the escape, close on the next tick
            if (in_string)
                return CAT_QUOTE;  // close the string, then the structures
            return CAT_CLOSE_BRACE | CAT_CLOSE_BRACKET;
        }
    }

    uint16_t mask = (ws_run_ < kMaxWsRun) ? CAT_WHITESPACE : 0;

    switch (current_state_) {
        case JsonState::START:
            // Must start with { or [. (Forcing object-only at the root was
            // tried and reverted: a json-reluctant model then fights the
            // mask with whitespace floods instead of emitting a minimal
            // valid document.)
            mask |= CAT_OPEN_BRACE | CAT_OPEN_BRACKET;
            break;

        case JsonState::OBJECT_START:
            // After {: expect " (key) or }
            mask |= CAT_QUOTE | CAT_CLOSE_BRACE;
            break;

        case JsonState::AFTER_KEY:
            // After key: expect :
            mask |= CAT_COLON;
            break;

        case JsonState::AFTER_COLON:
            // After :: expect any value
            mask |= CAT_VALUE_START;
            break;

        case JsonState::AFTER_VALUE:
            // After value in object: expect , or }
            mask |= CAT_COMMA | CAT_CLOSE_BRACE;
            break;

        case JsonState::ARRAY_START:
            // After [: expect value or ]
            mask |= CAT_VALUE_START | CAT_CLOSE_BRACKET;
            break;

        case JsonState::ARRAY_AFTER_VALUE:
            // After value in array: expect , or ]
            mask |= CAT_COMMA | CAT_CLOSE_BRACKET;
            break;

        case JsonState::ARRAY_NEED_VALUE:
            // After , in array: a value is mandatory — no closer (#1096).
            mask |= CAT_VALUE_START;
            break;

        case JsonState::OBJECT_NEED_KEY:
            // After , in object: a key is mandatory — no closer (#1096).
            mask |= CAT_QUOTE;
            break;

        case JsonState::IN_STRING:
            // Inside string: any string-safe char or closing "
            mask |= CAT_STRING_CHAR | CAT_QUOTE;
            break;

        case JsonState::IN_STRING_ESCAPE:
            // After backslash: any char (escape sequence)
            mask |= CAT_STRING_CHAR | CAT_QUOTE;
            break;

        case JsonState::IN_NUMBER:
            // Inside number: digit continuation, or structural that ends the number
            mask |= CAT_NUMBER_CONT | CAT_COMMA | CAT_CLOSE_BRACE | CAT_CLOSE_BRACKET;
            break;

        case JsonState::IN_LITERAL:
            // Inside literal (true/false/null): only literal continuation chars
            mask |= CAT_LITERAL_CONT;
            // If the literal is complete, also allow post-value tokens
            if (!target_literal_.empty() && partial_literal_.size() >= target_literal_.size()) {
                mask |= CAT_COMMA | CAT_CLOSE_BRACE | CAT_CLOSE_BRACKET;
            }
            break;

        case JsonState::DONE:
            // Parsing complete — only EOS (and capped whitespace). CAT_EOS
            // must stay allowed even when the WS-run cap zeroes whitespace,
            // otherwise every token is -inf and greedy argmax degenerates to
            // token id 0 ('!' on byte-level BPE vocabs).
            mask |= CAT_EOS;
            break;

        default:
            // Fallback: allow everything
            mask = 0xFFFF;
            break;
    }

    return mask;
}

bool JsonConstrainer::advance_char(char c) {
    // Skip whitespace in non-string states — but count the run. JSON allows
    // unlimited inter-token whitespace, and a model that doesn't want to emit
    // JSON exploits that as an escape hatch (greedy decode emits newlines
    // until max_tokens). compute_allowed_mask() drops CAT_WHITESPACE once the
    // run exceeds the cap, forcing a structural token (or EOS) instead.
    //
    // Returns true when the char is a legal continuation in the current FSM
    // state. update() ignores the result (tolerant, as before); the
    // whole-token simulation in apply_mask() uses it to reject tokens whose
    // FIRST char passes the category mask but whose tail violates the
    // grammar (e.g. the single token "[]." — '.' after a completed value).
    // Whitespace TERMINATES a number, it does not continue one: "1.  1" is not
    // a JSON number, yet the blanket skip below kept the FSM in IN_NUMBER and
    // let the emitted text interleave digits with spaces (#1104). Close the
    // number first, then let the whitespace be skipped in the parent state.
    if (current_state_ == JsonState::IN_NUMBER && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
        if (num_need_digit_)
            return false;  // "1." / "1e" / "-" cannot end here
        current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
        if (!state_stack_.empty())
            state_stack_.pop_back();
    }
    if (current_state_ != JsonState::IN_STRING && current_state_ != JsonState::IN_STRING_ESCAPE &&
        (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
        ws_run_++;
        return true;
    }
    ws_run_ = 0;

    switch (current_state_) {
        // Stack discipline: every opener pushes its CONTINUATION — the state
        // the parser resumes in after the construct closes — and every close
        // pops and *uses* it (empty stack -> DONE). The old code popped the
        // continuation but then peeked the grandparent's entry instead, so a
        // nested array closing inside an object left the FSM in array
        // context, accepting `,"bare-string"` + `]]` (#1067).
        case JsonState::START:
            // Root construct: continuation after it closes is DONE, which the
            // empty-stack fallback in the close handlers provides — no push.
            if (c == '{') {
                current_state_ = JsonState::OBJECT_START;
            } else if (c == '[') {
                current_state_ = JsonState::ARRAY_START;
            } else {
                return false;
            }
            break;

        case JsonState::OBJECT_START:
        case JsonState::OBJECT_NEED_KEY:
            if (c == '"') {
                current_state_ = JsonState::IN_STRING;
                state_stack_.push_back(JsonState::AFTER_KEY);
            } else if (c == '}') {
                // Legal only for an EMPTY object: after a comma this would be
                // a trailing comma (#1096).
                if (current_state_ == JsonState::OBJECT_NEED_KEY)
                    return false;
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
                if (!state_stack_.empty())
                    state_stack_.pop_back();
            } else {
                return false;
            }
            break;

        case JsonState::AFTER_KEY:
            if (c == ':') {
                current_state_ = JsonState::AFTER_COLON;
            } else {
                return false;
            }
            break;

        case JsonState::AFTER_COLON:
            // Every value form pushes AFTER_VALUE — scalars too: their end
            // handlers pop the continuation, so a missing push here made
            // them steal the enclosing container's entry.
            if (c == '"') {
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::IN_STRING;
            } else if (c == '{') {
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::OBJECT_START;
            } else if (c == '[') {
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::ARRAY_START;
            } else if (c == 't') {
                target_literal_ = "true";
                partial_literal_ = "t";
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::IN_LITERAL;
            } else if (c == 'f') {
                target_literal_ = "false";
                partial_literal_ = "f";
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::IN_LITERAL;
            } else if (c == 'n') {
                target_literal_ = "null";
                partial_literal_ = "n";
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::IN_LITERAL;
            } else if ((c >= '0' && c <= '9') || c == '-') {
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::IN_NUMBER;
                enter_number_(c);
            } else {
                return false;
            }
            break;

        case JsonState::AFTER_VALUE:
            if (c == ',') {
                // Next key in object — NOT OBJECT_START, which would also
                // accept `}` and turn `{"a":1,}` into legal output (#1096).
                current_state_ = JsonState::OBJECT_NEED_KEY;
            } else if (c == '}') {
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
                if (!state_stack_.empty())
                    state_stack_.pop_back();
            } else {
                return false;
            }
            break;

        case JsonState::ARRAY_START:
        case JsonState::ARRAY_NEED_VALUE:
            if (c == ']') {
                // Legal only for an EMPTY array: after a comma this would be
                // a trailing comma (#1096).
                if (current_state_ == JsonState::ARRAY_NEED_VALUE)
                    return false;
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
                if (!state_stack_.empty())
                    state_stack_.pop_back();
            } else if (c == '"') {
                state_stack_.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state_ = JsonState::IN_STRING;
            } else if (c == '{') {
                state_stack_.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state_ = JsonState::OBJECT_START;
            } else if (c == '[') {
                state_stack_.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state_ = JsonState::ARRAY_START;
            } else if (c == 't') {
                target_literal_ = "true";
                partial_literal_ = "t";
                state_stack_.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state_ = JsonState::IN_LITERAL;
            } else if (c == 'f') {
                target_literal_ = "false";
                partial_literal_ = "f";
                state_stack_.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state_ = JsonState::IN_LITERAL;
            } else if (c == 'n') {
                target_literal_ = "null";
                partial_literal_ = "n";
                state_stack_.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state_ = JsonState::IN_LITERAL;
            } else if ((c >= '0' && c <= '9') || c == '-') {
                state_stack_.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state_ = JsonState::IN_NUMBER;
                enter_number_(c);
            } else {
                return false;
            }
            break;

        case JsonState::ARRAY_AFTER_VALUE:
            if (c == ',') {
                // See AFTER_VALUE: ARRAY_START would also accept `]`.
                current_state_ = JsonState::ARRAY_NEED_VALUE;
            } else if (c == ']') {
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
                if (!state_stack_.empty())
                    state_stack_.pop_back();
            } else {
                return false;
            }
            break;

        case JsonState::IN_STRING:
            if (c == '\\') {
                current_state_ = JsonState::IN_STRING_ESCAPE;
            } else if (c == '"') {
                // End of string — pop to parent state
                if (!state_stack_.empty()) {
                    current_state_ = state_stack_.back();
                    state_stack_.pop_back();
                } else {
                    current_state_ = JsonState::DONE;
                }
            } else if (static_cast<unsigned char>(c) < 0x20) {
                // JSON forbids raw control chars (U+0000–U+001F) inside
                // strings — they must arrive escaped. Multi-char tokens whose
                // first char passes the category mask (e.g. `"<newline>`)
                // used to smuggle them through.
                return false;
            }
            // Otherwise stay in IN_STRING
            break;

        case JsonState::IN_STRING_ESCAPE:
            // Back to IN_STRING — but `\` + raw control char is not a legal
            // escape sequence (the escape char itself must be printable).
            if (static_cast<unsigned char>(c) < 0x20)
                return false;
            current_state_ = JsonState::IN_STRING;
            break;

        case JsonState::IN_NUMBER: {
            // RFC 8259 number grammar. The old test accepted every one of
            // '.', 'e', 'E', '+', '-' unconditionally, which made "3.5.5.5"
            // legal and left a degenerating model with no forced way out (#1104).
            bool ok = false;
            if (c >= '0' && c <= '9') {
                ok = true;
                num_need_digit_ = false;
                num_exp_sign_ok_ = false;
            } else if (c == '.') {
                ok = !num_seen_frac_ && !num_seen_exp_ && !num_need_digit_;
                if (ok) {
                    num_seen_frac_ = true;
                    num_need_digit_ = true;  // frac = "." 1*DIGIT
                }
            } else if (c == 'e' || c == 'E') {
                ok = !num_seen_exp_ && !num_need_digit_;
                if (ok) {
                    num_seen_exp_ = true;
                    num_exp_sign_ok_ = true;
                    num_need_digit_ = true;  // exp = e [sign] 1*DIGIT
                }
            } else if (c == '+' || c == '-') {
                ok = num_exp_sign_ok_;
                if (ok) {
                    num_exp_sign_ok_ = false;
                    num_need_digit_ = true;
                }
            }
            if (!ok) {
                // A number still owing a digit ("3.", "1e", "-") is incomplete:
                // nothing may terminate it, not even a structural char.
                if (num_need_digit_)
                    return false;
                // Number ended — this char is part of the parent context
                // Pop back to parent and re-process this character
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
                if (!state_stack_.empty())
                    state_stack_.pop_back();
                return advance_char(c);  // re-process in parent context
            }
            break;
        }

        case JsonState::IN_LITERAL:
            // Strict: the char must be the literal's next expected char
            // ("tru"+'x' was silently accepted before).
            if (partial_literal_.size() >= target_literal_.size() ||
                c != target_literal_[partial_literal_.size()])
                return false;
            partial_literal_ += c;
            if (partial_literal_.size() >= target_literal_.size()) {
                // Literal complete — transition to parent
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
                if (!state_stack_.empty())
                    state_stack_.pop_back();
            }
            break;

        case JsonState::DONE:
            // Any non-whitespace after a complete document is invalid.
            return false;
    }
    return true;
}

bool JsonConstrainer::sim_token_valid(const std::string& text) {
    // Snapshot → strict-advance over the whole token text → restore.
    // advance_char is the single grammar source of truth (no parallel FSM).
    const JsonState st = current_state_;
    const size_t depth = state_stack_.size();
    const std::string plit = partial_literal_;
    const std::string tlit = target_literal_;
    const int ws = ws_run_;
    // The number sub-state is part of the FSM and MUST round-trip too — a
    // simulated token that walks into a number would otherwise leave
    // num_seen_frac_/num_need_digit_ mutated on the real state (#1104).
    const bool nfrac = num_seen_frac_, nexp = num_seen_exp_;
    const bool nsign = num_exp_sign_ok_, ndig = num_need_digit_;
    std::vector<JsonState> stack_copy = state_stack_;

    bool ok = true;
    for (char c : text) {
        if (!advance_char(c)) {
            ok = false;
            break;
        }
    }

    current_state_ = st;
    state_stack_ = std::move(stack_copy);
    (void)depth;
    partial_literal_ = plit;
    target_literal_ = tlit;
    num_seen_frac_ = nfrac;
    num_seen_exp_ = nexp;
    num_exp_sign_ok_ = nsign;
    num_need_digit_ = ndig;
    ws_run_ = ws;
    return ok;
}

void JsonConstrainer::update(int32_t token) {
    if (token < 0 || token >= vocab_size_)
        return;
    const std::string& text = token_texts_[token];
    if (preamble_.absorb(token, text))
        return;
    for (char c : text) {
        advance_char(c);
    }
}

// Inside a string, a token that carries no '"' and no '\\' cannot change the
// FSM state — the old shortcut concluded from that it must be legal and skipped
// the whole-token simulation. It can still be ILLEGAL: JSON forbids raw control
// characters (U+0000-U+001F) in strings, advance_char rejects them, and the
// shortcut walked straight past that guard. A model then emitted a raw newline
// inside a string and the reply did not parse (#1104).
static bool string_token_needs_simulation(const std::string& text) {
    for (char c : text) {
        if (c == '"' || c == '\\' || static_cast<unsigned char>(c) < 0x20)
            return true;
    }
    return false;
}

void JsonConstrainer::apply_mask(float* d_logits, int vocab_size, cudaStream_t stream) {
    if (!initialized_ || !d_token_categories_ || !d_allowed_mask_)
        return;

    if (preamble_.active())
        return;

    uint16_t mask = compute_allowed_mask();

    // Whole-token validation: the category bitmask only inspects a token's
    // FIRST character class, so multi-char tokens could smuggle grammar
    // violations past it (the single token "[]." closes the document and
    // appends an illegal '.'). Simulate every category-passing candidate
    // through the FSM (advance_char strict mode) and build a per-token
    // allow list. Hot-path shortcut: inside strings, tokens without '"'
    // or '\\' can never change FSM state — skip the simulation.
    if (token_allow_.size() != static_cast<size_t>(vocab_size))
        token_allow_.assign(vocab_size, 0);
    const bool in_string =
        current_state_ == JsonState::IN_STRING || current_state_ == JsonState::IN_STRING_ESCAPE;
    size_t n_allowed = 0;
    // vocab_size is the LOGITS width (model vocab); token_categories_ /
    // token_texts_ only cover the TOKENIZER vocab (vocab_size_). SafeTensors
    // models pad the lm_head past the tokenizer vocab (Qwen3-8B-NVFP4: 151936
    // vs 151669) — iterating to vocab_size read token_texts_ out of bounds
    // (host SIGBUS, killed imp-server on the first json_mode request).
    // Padding ids stay allow=0 and the kernel masks them via n_classified.
    const int n_classified = std::min(vocab_size, vocab_size_);
    for (int i = 0; i < n_classified; i++) {
        uint8_t allow = 0;
        if ((token_categories_[i] & mask) != 0) {
            const std::string& text = token_texts_[i];
            if (token_categories_[i] == CAT_EOS) {
                allow = 1;  // EOS already gated by the mask (DONE state only)
            } else if (in_string && !string_token_needs_simulation(text)) {
                allow = 1;
            } else {
                allow = sim_token_valid(text) ? 1 : 0;
            }
        }
        token_allow_[i] = allow;
        n_allowed += allow;
    }

    // Force-close safety net: the narrowed mask offers only closers, and a
    // closer is not legal in every state — after a key the grammar demands
    // ':' and a value first. Narrowing there leaves nothing legal, which used
    // to drop straight into the EOS guard below and end the reply mid-document
    // with finish_reason="stop" (worse than the truncation it was meant to
    // prevent). Retry once with the ordinary mask: force-close may help, it
    // must never make the outcome worse.
    if (n_allowed == 0 && force_close_active_) {
        const int saved = remaining_budget_;
        remaining_budget_ = -1;  // disable narrowing for this recompute
        mask = compute_allowed_mask();
        remaining_budget_ = saved;
        for (int i = 0; i < n_classified; i++) {
            uint8_t allow = 0;
            if ((token_categories_[i] & mask) != 0) {
                const std::string& text = token_texts_[i];
                if (token_categories_[i] == CAT_EOS)
                    allow = 1;
                else if (in_string && !string_token_needs_simulation(text))
                    allow = 1;
                else
                    allow = sim_token_valid(text) ? 1 : 0;
            }
            token_allow_[i] = allow;
            n_allowed += allow;
        }
    }

    // Empty-allow guard: if NOTHING passes (over-tight schema/state combo),
    // every logit would be -FLT_MAX and greedy argmax degenerates to token
    // id 0 — the "!!!!!" spam. Force a clean finish instead by allowing EOS.
    if (n_allowed == 0) {
        for (int32_t eid : eos_ids_) {
            if (eid >= 0 && eid < vocab_size)
                token_allow_[eid] = 1;
        }
        IMP_LOG_WARN("JsonConstrainer: no token satisfies the grammar in state %d — allowing EOS",
                     std::to_underlying(current_state_));
    }

    if (!d_token_allow_) {
        if (cudaMalloc(&d_token_allow_, vocab_size) != cudaSuccess) {
            d_token_allow_ = nullptr;
            return;
        }
    }
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_token_allow_, token_allow_.data(), vocab_size, cudaMemcpyHostToDevice, stream));

    uint16_t any_mask = 0xFFFF;  // per-token allow already encodes the category mask
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_allowed_mask_, &any_mask, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    constrain_mask_allow_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_token_categories_,
                                                                d_token_allow_, d_allowed_mask_, vocab_size,
                                                                n_classified, /*use_token_allow=*/true);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
