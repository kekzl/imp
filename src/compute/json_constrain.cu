#include "compute/json_constrain.h"
#include "compute/constrain_common.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstring>
#include <algorithm>

namespace imp {

// ============================================================================
// JsonConstrainer implementation
// ============================================================================

JsonConstrainer::~JsonConstrainer() {
    if (d_token_categories_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_categories_));
        d_token_categories_ = nullptr;
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
    for (int32_t eid : tok.eos_ids()) {
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
    preamble_.reset();
}

uint16_t JsonConstrainer::compute_allowed_mask() const {
    // Whitespace is legal between any JSON tokens, but cap the run length —
    // see advance_char. (EOS has its own CAT_EOS bit, allowed in DONE only.)
    constexpr int kMaxWsRun = 64;
    uint16_t mask = (ws_run_ < kMaxWsRun) ? CAT_WHITESPACE : 0;

    switch (current_state_) {
        case JsonState::START:
            // Must start with { or [
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

void JsonConstrainer::advance_char(char c) {
    // Skip whitespace in non-string states — but count the run. JSON allows
    // unlimited inter-token whitespace, and a model that doesn't want to emit
    // JSON exploits that as an escape hatch (greedy decode emits newlines
    // until max_tokens). compute_allowed_mask() drops CAT_WHITESPACE once the
    // run exceeds the cap, forcing a structural token (or EOS) instead.
    if (current_state_ != JsonState::IN_STRING && current_state_ != JsonState::IN_STRING_ESCAPE &&
        (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
        ws_run_++;
        return;
    }
    ws_run_ = 0;

    switch (current_state_) {
        case JsonState::START:
            if (c == '{') {
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::OBJECT_START;
            } else if (c == '[') {
                state_stack_.push_back(JsonState::ARRAY_AFTER_VALUE);
                current_state_ = JsonState::ARRAY_START;
            }
            break;

        case JsonState::OBJECT_START:
            if (c == '"') {
                current_state_ = JsonState::IN_STRING;
                state_stack_.push_back(JsonState::AFTER_KEY);
            } else if (c == '}') {
                if (!state_stack_.empty())
                    state_stack_.pop_back();
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
            }
            break;

        case JsonState::AFTER_KEY:
            if (c == ':') {
                current_state_ = JsonState::AFTER_COLON;
            }
            break;

        case JsonState::AFTER_COLON:
            if (c == '"') {
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::IN_STRING;
            } else if (c == '{') {
                // Push the AFTER_VALUE to restore after nested object
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::OBJECT_START;
            } else if (c == '[') {
                state_stack_.push_back(JsonState::AFTER_VALUE);
                current_state_ = JsonState::ARRAY_START;
            } else if (c == 't') {
                target_literal_ = "true";
                partial_literal_ = "t";
                current_state_ = JsonState::IN_LITERAL;
            } else if (c == 'f') {
                target_literal_ = "false";
                partial_literal_ = "f";
                current_state_ = JsonState::IN_LITERAL;
            } else if (c == 'n') {
                target_literal_ = "null";
                partial_literal_ = "n";
                current_state_ = JsonState::IN_LITERAL;
            } else if ((c >= '0' && c <= '9') || c == '-') {
                current_state_ = JsonState::IN_NUMBER;
            }
            break;

        case JsonState::AFTER_VALUE:
            if (c == ',') {
                // Next key in object
                current_state_ = JsonState::OBJECT_START;
            } else if (c == '}') {
                if (!state_stack_.empty())
                    state_stack_.pop_back();
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
            }
            break;

        case JsonState::ARRAY_START:
            if (c == ']') {
                if (!state_stack_.empty())
                    state_stack_.pop_back();
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
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
                current_state_ = JsonState::IN_LITERAL;
            } else if (c == 'f') {
                target_literal_ = "false";
                partial_literal_ = "f";
                current_state_ = JsonState::IN_LITERAL;
            } else if (c == 'n') {
                target_literal_ = "null";
                partial_literal_ = "n";
                current_state_ = JsonState::IN_LITERAL;
            } else if ((c >= '0' && c <= '9') || c == '-') {
                current_state_ = JsonState::IN_NUMBER;
            }
            break;

        case JsonState::ARRAY_AFTER_VALUE:
            if (c == ',') {
                current_state_ = JsonState::ARRAY_START;
            } else if (c == ']') {
                if (!state_stack_.empty())
                    state_stack_.pop_back();
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
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
            }
            // Otherwise stay in IN_STRING
            break;

        case JsonState::IN_STRING_ESCAPE:
            // Any char after \ — back to IN_STRING
            current_state_ = JsonState::IN_STRING;
            break;

        case JsonState::IN_NUMBER:
            if (!((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')) {
                // Number ended — this char is part of the parent context
                // Pop back to parent and re-process this character
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
                if (!state_stack_.empty())
                    state_stack_.pop_back();
                advance_char(c);  // re-process
                return;
            }
            break;

        case JsonState::IN_LITERAL:
            partial_literal_ += c;
            if (partial_literal_.size() >= target_literal_.size()) {
                // Literal complete — transition to parent
                current_state_ = state_stack_.empty() ? JsonState::DONE : state_stack_.back();
                if (!state_stack_.empty())
                    state_stack_.pop_back();
            }
            break;

        case JsonState::DONE:
            break;
    }
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

void JsonConstrainer::apply_mask(float* d_logits, int vocab_size, cudaStream_t stream) {
    if (!initialized_ || !d_token_categories_ || !d_allowed_mask_)
        return;

    if (preamble_.active())
        return;

    uint16_t mask = compute_allowed_mask();

    // Upload mask to device
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_allowed_mask_, &mask, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));

    // Launch masking kernel
    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    constrain_mask_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_token_categories_, d_allowed_mask_,
                                                          vocab_size);
}

// ============================================================================
// GrammarConstrainer implementation (Part B)
// ============================================================================

GrammarConstrainer::~GrammarConstrainer() {
    if (d_token_allow_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_allow_));
        d_token_allow_ = nullptr;
    }
}

bool GrammarConstrainer::init(const Tokenizer& tok, std::shared_ptr<RegexNfa> grammar) {
    if (!grammar || !grammar->compiled()) {
        IMP_LOG_ERROR("GrammarConstrainer: null/uncompiled grammar");
        return false;
    }
    grammar_ = std::move(grammar);
    vocab_size_ = tok.vocab_size();
    token_texts_.resize(vocab_size_);
    token_allow_.resize(vocab_size_, 1);
    for (int i = 0; i < vocab_size_; i++)
        token_texts_[i] = tok.decode_token(static_cast<int32_t>(i));

    cudaError_t err = cudaMalloc(&d_token_allow_, vocab_size_ * sizeof(uint8_t));
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("GrammarConstrainer: failed to allocate allow buffer: %s", cudaGetErrorString(err));
        return false;
    }

    reset();
    initialized_ = true;
    IMP_LOG_INFO("GrammarConstrainer initialized (%d tokens)", vocab_size_);
    return true;
}

void GrammarConstrainer::reset() {
    if (grammar_)
        active_states_ = grammar_->start_set();
}

bool GrammarConstrainer::accepts_now() const {
    return grammar_ && grammar_->accepts(active_states_);
}

void GrammarConstrainer::compute_token_allow_mask() {
    for (int i = 0; i < vocab_size_; i++) {
        const std::string& text = token_texts_[i];
        if (text.empty()) {
            token_allow_[i] = 0;
            continue;
        }
        // A token is allowed iff feeding all its bytes keeps the NFA alive.
        std::vector<int> st = active_states_;
        bool alive = true;
        for (char c : text) {
            st = grammar_->step(st, static_cast<unsigned char>(c));
            if (st.empty()) {
                alive = false;
                break;
            }
        }
        token_allow_[i] = alive ? 1 : 0;
    }
}

void GrammarConstrainer::apply_mask(float* d_logits, int vocab_size, cudaStream_t stream) {
    if (!initialized_ || !grammar_)
        return;

    compute_token_allow_mask();
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_token_allow_, token_allow_.data(), vocab_size_ * sizeof(uint8_t),
                                       cudaMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    // No category gating for grammar; the per-token allow mask (built from the
    // NFA) carries the full constraint.
    grammar_mask_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_token_allow_, vocab_size);
}

void GrammarConstrainer::update(int32_t token) {
    if (!initialized_ || token < 0 || token >= vocab_size_)
        return;
    const std::string& text = token_texts_[token];
    for (char c : text)
        active_states_ = grammar_->step(active_states_, static_cast<unsigned char>(c));
}

}  // namespace imp
