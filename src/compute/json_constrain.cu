#include "compute/json_constrain.h"
#include "compute/constrain_common.h"
#include "core/logging.h"

#include <mutex>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstring>
#include <algorithm>
#include <utility>

namespace imp {

// ============================================================================
// JsonConstrainer implementation
// ============================================================================

JsonConstrainer::~JsonConstrainer() = default;

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
    if (!dev_.alloc_categories("JsonConstrainer", token_categories_.data(), vocab_size_))
        return false;

    if (!dev_.alloc_allowed_mask("JsonConstrainer"))
        return false;

    // Per-token allow list. Allocated HERE, not on first use (issue #1104).
    // The lazy version allocated it inside apply_mask() — i.e. mid-decode, on
    // the serving path — and on a model that loads with no free VRAM left
    // (#1103) the allocation failed and apply_mask returned WITHOUT applying
    // any mask and without logging. The request then decoded unconstrained and
    // returned prose where JSON was promised: deterministic, first request per
    // process, invisible in `imp_constrained_eager_fallback_total` because no
    // fallback was taken. The three sibling constrainers (regex, grammar,
    // schema) have always allocated this at init and failed the load loudly;
    // this one was the outlier.
    if (!dev_.alloc_token_allow("JsonConstrainer", vocab_size_))
        return false;

    reset();
    initialized_ = true;
    IMP_LOG_INFO("JsonConstrainer initialized (%d tokens classified)", vocab_size_);
    return true;
}

void JsonGrammar::reset() {
    state_stack.clear();
    current_state = JsonState::START;
    partial_literal.clear();
    target_literal.clear();
    ws_run = 0;
    enter_number('0');  // clear the number sub-state (see #1104)
    num_need_digit = false;
}

void JsonConstrainer::reset() {
    g_.reset();
    preamble_.reset();
}

// Seed the RFC 8259 number sub-state on entry. `c` is the first character of
// the number: a bare '-' still owes its first digit, a digit does not.
void JsonGrammar::enter_number(char c) {
    num_seen_frac = false;
    num_seen_exp = false;
    num_exp_sign_ok = false;
    num_need_digit = (c == '-');
}

uint16_t JsonGrammar::compute_allowed_mask() const {
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
    force_close_active = false;
    if (remaining_budget >= 0 && current_state != JsonState::DONE) {
        // A closer is not legal in every state, and demanding one where the
        // grammar forbids it is worse than not narrowing at all: the safety
        // net below then retries with the ordinary mask, the model carries on
        // freely, and the document is truncated anyway. Measured on
        // Qwen3.6-35B-A3B-NVFP4 at max_tokens=40 — the narrowing fired in
        // ARRAY_NEED_VALUE, where #1096 forbids ']' precisely so `[1,]` cannot
        // happen, so nothing was allowed and the reply came back unparseable
        // (#1291).
        //
        // So the narrowing first has to walk OUT of a state that owes
        // something, and the budget has to cover that walk. escape_mask is the
        // cheapest legal step; escape_cost is how many tokens the whole walk
        // takes, counted high on purpose — closing early beats not closing.
        uint16_t escape_mask = 0;
        int escape_cost = 0;
        switch (current_state) {
            case JsonState::IN_STRING:
                escape_mask = CAT_QUOTE;  // close it, then AFTER_VALUE takes closers
                escape_cost = 1;
                break;
            case JsonState::OBJECT_KEY:
                escape_mask = CAT_QUOTE;  // close key → AFTER_KEY still owes ':' + a value
                escape_cost = 3;
                break;
            case JsonState::IN_STRING_ESCAPE:
                escape_mask = CAT_STRING_CHAR;  // finish the escape, then close the string
                escape_cost = 2;
                break;
            case JsonState::AFTER_KEY:
                escape_mask = CAT_COLON;  // ':' then a value
                escape_cost = 2;
                break;
            case JsonState::AFTER_COLON:
            case JsonState::ARRAY_NEED_VALUE:
                // A number is the only value that both starts and ends in one
                // token — a string opens another frame, a container opens two.
                escape_mask = CAT_NUMBER_START;
                escape_cost = 1;
                break;
            case JsonState::OBJECT_NEED_KEY:
                escape_mask = CAT_QUOTE;  // "…" then ':' then a value
                escape_cost = 4;
                break;
            case JsonState::IN_LITERAL:
                // A closer ends a *complete* literal; a partial one owes its
                // tail first. 4 covers the longest ("false").
                if (target_literal.empty() || partial_literal.size() < target_literal.size()) {
                    escape_mask = CAT_LITERAL_CONT;
                    escape_cost = 4;
                }
                break;
            case JsonState::START:
                escape_mask = CAT_OPEN_BRACE;  // nothing is open yet; open then close
                escape_cost = 1;
                break;
            default:
                break;  // OBJECT_START / ARRAY_START / *_AFTER_VALUE / IN_NUMBER take a closer
        }
        // state_stack holds only the RETURN states of *nested* values, not the
        // container we are currently inside — at `{"a"` the stack is empty
        // while a '}' is still owed. Count that container explicitly, or the
        // narrowing releases one token too early and the document is truncated
        // anyway (observed: needed=0 in AFTER_KEY with an object still open).
        const bool in_container = (current_state != JsonState::START && current_state != JsonState::DONE);
        // +1 margin: the escape step can itself push a frame (a forced number
        // enters IN_NUMBER inside its container), so an estimate that is exact
        // at the moment it is taken can still land one token short. Measured
        // on the #1291 repro: without it the walk emits `-1]` and runs out
        // before the `}`. Erring high costs a token of content; erring low
        // costs the whole document.
        const int needed = static_cast<int>(state_stack.size()) + (in_container ? 1 : 0) + escape_cost + 1;
        if (remaining_budget <= needed) {
            force_close_active = true;
            if (escape_mask != 0)
                return escape_mask;
            return CAT_CLOSE_BRACE | CAT_CLOSE_BRACKET;
        }
    }

    uint16_t mask = (ws_run < kMaxWsRun) ? CAT_WHITESPACE : 0;

    switch (current_state) {
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
            if (!target_literal.empty() && partial_literal.size() >= target_literal.size()) {
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

bool JsonGrammar::advance_char(char c) {
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
    if (current_state == JsonState::IN_NUMBER && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
        if (num_need_digit)
            return false;  // "1." / "1e" / "-" cannot end here
        current_state = state_stack.empty() ? JsonState::DONE : state_stack.back();
        if (!state_stack.empty())
            state_stack.pop_back();
    }
    if (current_state != JsonState::IN_STRING && current_state != JsonState::IN_STRING_ESCAPE &&
        (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
        ws_run++;
        return true;
    }
    ws_run = 0;

    switch (current_state) {
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
                current_state = JsonState::OBJECT_START;
            } else if (c == '[') {
                current_state = JsonState::ARRAY_START;
            } else {
                return false;
            }
            break;

        case JsonState::OBJECT_START:
        case JsonState::OBJECT_NEED_KEY:
            if (c == '"') {
                current_state = JsonState::IN_STRING;
                state_stack.push_back(JsonState::AFTER_KEY);
            } else if (c == '}') {
                // Legal only for an EMPTY object: after a comma this would be
                // a trailing comma (#1096).
                if (current_state == JsonState::OBJECT_NEED_KEY)
                    return false;
                current_state = state_stack.empty() ? JsonState::DONE : state_stack.back();
                if (!state_stack.empty())
                    state_stack.pop_back();
            } else {
                return false;
            }
            break;

        case JsonState::AFTER_KEY:
            if (c == ':') {
                current_state = JsonState::AFTER_COLON;
            } else {
                return false;
            }
            break;

        case JsonState::AFTER_COLON:
            // Every value form pushes AFTER_VALUE — scalars too: their end
            // handlers pop the continuation, so a missing push here made
            // them steal the enclosing container's entry.
            if (c == '"') {
                state_stack.push_back(JsonState::AFTER_VALUE);
                current_state = JsonState::IN_STRING;
            } else if (c == '{') {
                state_stack.push_back(JsonState::AFTER_VALUE);
                current_state = JsonState::OBJECT_START;
            } else if (c == '[') {
                state_stack.push_back(JsonState::AFTER_VALUE);
                current_state = JsonState::ARRAY_START;
            } else if (c == 't') {
                target_literal = "true";
                partial_literal = "t";
                state_stack.push_back(JsonState::AFTER_VALUE);
                current_state = JsonState::IN_LITERAL;
            } else if (c == 'f') {
                target_literal = "false";
                partial_literal = "f";
                state_stack.push_back(JsonState::AFTER_VALUE);
                current_state = JsonState::IN_LITERAL;
            } else if (c == 'n') {
                target_literal = "null";
                partial_literal = "n";
                state_stack.push_back(JsonState::AFTER_VALUE);
                current_state = JsonState::IN_LITERAL;
            } else if ((c >= '0' && c <= '9') || c == '-') {
                state_stack.push_back(JsonState::AFTER_VALUE);
                current_state = JsonState::IN_NUMBER;
                enter_number(c);
            } else {
                return false;
            }
            break;

        case JsonState::AFTER_VALUE:
            if (c == ',') {
                // Next key in object — NOT OBJECT_START, which would also
                // accept `}` and turn `{"a":1,}` into legal output (#1096).
                current_state = JsonState::OBJECT_NEED_KEY;
            } else if (c == '}') {
                current_state = state_stack.empty() ? JsonState::DONE : state_stack.back();
                if (!state_stack.empty())
                    state_stack.pop_back();
            } else {
                return false;
            }
            break;

        case JsonState::ARRAY_START:
        case JsonState::ARRAY_NEED_VALUE:
            if (c == ']') {
                // Legal only for an EMPTY array: after a comma this would be
                // a trailing comma (#1096).
                if (current_state == JsonState::ARRAY_NEED_VALUE)
                    return false;
                current_state = state_stack.empty() ? JsonState::DONE : state_stack.back();
                if (!state_stack.empty())
                    state_stack.pop_back();
            } else if (c == '"') {
                state_stack.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state = JsonState::IN_STRING;
            } else if (c == '{') {
                state_stack.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state = JsonState::OBJECT_START;
            } else if (c == '[') {
                state_stack.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state = JsonState::ARRAY_START;
            } else if (c == 't') {
                target_literal = "true";
                partial_literal = "t";
                state_stack.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state = JsonState::IN_LITERAL;
            } else if (c == 'f') {
                target_literal = "false";
                partial_literal = "f";
                state_stack.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state = JsonState::IN_LITERAL;
            } else if (c == 'n') {
                target_literal = "null";
                partial_literal = "n";
                state_stack.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state = JsonState::IN_LITERAL;
            } else if ((c >= '0' && c <= '9') || c == '-') {
                state_stack.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state = JsonState::IN_NUMBER;
                enter_number(c);
            } else {
                return false;
            }
            break;

        case JsonState::ARRAY_AFTER_VALUE:
            if (c == ',') {
                // See AFTER_VALUE: ARRAY_START would also accept `]`.
                current_state = JsonState::ARRAY_NEED_VALUE;
            } else if (c == ']') {
                current_state = state_stack.empty() ? JsonState::DONE : state_stack.back();
                if (!state_stack.empty())
                    state_stack.pop_back();
            } else {
                return false;
            }
            break;

        case JsonState::IN_STRING:
            if (c == '\\') {
                current_state = JsonState::IN_STRING_ESCAPE;
            } else if (c == '"') {
                // End of string — pop to parent state
                if (!state_stack.empty()) {
                    current_state = state_stack.back();
                    state_stack.pop_back();
                } else {
                    current_state = JsonState::DONE;
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
            current_state = JsonState::IN_STRING;
            break;

        case JsonState::IN_NUMBER: {
            // RFC 8259 number grammar. The old test accepted every one of
            // '.', 'e', 'E', '+', '-' unconditionally, which made "3.5.5.5"
            // legal and left a degenerating model with no forced way out (#1104).
            bool ok = false;
            if (c >= '0' && c <= '9') {
                ok = true;
                num_need_digit = false;
                num_exp_sign_ok = false;
            } else if (c == '.') {
                ok = !num_seen_frac && !num_seen_exp && !num_need_digit;
                if (ok) {
                    num_seen_frac = true;
                    num_need_digit = true;  // frac = "." 1*DIGIT
                }
            } else if (c == 'e' || c == 'E') {
                ok = !num_seen_exp && !num_need_digit;
                if (ok) {
                    num_seen_exp = true;
                    num_exp_sign_ok = true;
                    num_need_digit = true;  // exp = e [sign] 1*DIGIT
                }
            } else if (c == '+' || c == '-') {
                ok = num_exp_sign_ok;
                if (ok) {
                    num_exp_sign_ok = false;
                    num_need_digit = true;
                }
            }
            if (!ok) {
                // A number still owing a digit ("3.", "1e", "-") is incomplete:
                // nothing may terminate it, not even a structural char.
                if (num_need_digit)
                    return false;
                // Number ended — this char is part of the parent context
                // Pop back to parent and re-process this character
                current_state = state_stack.empty() ? JsonState::DONE : state_stack.back();
                if (!state_stack.empty())
                    state_stack.pop_back();
                return advance_char(c);  // re-process in parent context
            }
            break;
        }

        case JsonState::IN_LITERAL:
            // Strict: the char must be the literal's next expected char
            // ("tru"+'x' was silently accepted before).
            if (partial_literal.size() >= target_literal.size() ||
                c != target_literal[partial_literal.size()])
                return false;
            partial_literal += c;
            if (partial_literal.size() >= target_literal.size()) {
                // Literal complete — transition to parent
                current_state = state_stack.empty() ? JsonState::DONE : state_stack.back();
                if (!state_stack.empty())
                    state_stack.pop_back();
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
    //
    // One struct copy since #1729. This used to save and restore eleven fields
    // by hand, and the number sub-state had to be added to that list after
    // #1104 found it missing: a simulated token that walked into a number left
    // num_seen_frac/num_need_digit mutated on the real state. A field added to
    // the grammar now round-trips because it is in the grammar.
    const JsonGrammar saved = g_;
    bool ok = true;
    for (char c : text) {
        if (!g_.advance_char(c)) {
            ok = false;
            break;
        }
    }
    g_ = saved;
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
    if (!initialized_ || !dev_.categories() || !dev_.allowed_mask())
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
    const bool in_string = g_.current_state == JsonState::IN_STRING ||
                           g_.current_state == JsonState::IN_STRING_ESCAPE;
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
    if (n_allowed == 0 && g_.force_close_active) {
        const int saved = g_.remaining_budget;
        g_.remaining_budget = -1;  // disable narrowing for this recompute
        mask = compute_allowed_mask();
        g_.remaining_budget = saved;
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
                     std::to_underlying(g_.current_state));
    }

    if (!dev_.has_token_allow()) {
        // Unreachable after a successful initialize(), which is the point: a
        // constrainer that cannot mask must SAY so rather than let the request
        // decode unconstrained and look like a model that ignored the schema
        // (issue #1104).
        static std::once_flag once;
        std::call_once(once, [] {
            IMP_LOG_ERROR(
                "JsonConstrainer: no device allow list — the reply will NOT be constrained. "
                "This should be impossible after initialize(); please report it.");
        });
        return;
    }
    // Clamp to what initialize() reserved. The model's lm_head can be WIDER
    // than the tokenizer (padding rows), and the host list is sized to the
    // model — but the kernel returns before touching token_allow[idx] for
    // idx >= n_classified, so the tokenizer-sized device buffer is exactly
    // enough and copying the model width would run off the end of it.
    const size_t allow_bytes = std::min(static_cast<size_t>(vocab_size), static_cast<size_t>(vocab_size_));
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(dev_.token_allow(), token_allow_.data(), allow_bytes, cudaMemcpyHostToDevice, stream));

    uint16_t any_mask = 0xFFFF;  // per-token allow already encodes the category mask
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(dev_.allowed_mask(), &any_mask, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    constrain_mask_allow_kernel<<<blocks, threads, 0, stream>>>(d_logits, dev_.categories(),
                                                                dev_.token_allow(), dev_.allowed_mask(),
                                                                vocab_size,
                                                                n_classified, /*use_token_allow=*/true);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
