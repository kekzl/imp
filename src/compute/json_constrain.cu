#include "compute/json_constrain.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstring>
#include <algorithm>

namespace imp {

// ============================================================================
// GPU kernel: apply category mask to logits
// ============================================================================

__global__ void json_mask_kernel(float* __restrict__ logits,
                                 const uint16_t* __restrict__ token_cats,
                                 const uint16_t* __restrict__ allowed_mask,
                                 int vocab_size) {
    uint16_t mask = *allowed_mask;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        if ((token_cats[idx] & mask) == 0) {
            logits[idx] = -FLT_MAX;
        }
    }
}

// ============================================================================
// Token classification
// ============================================================================

static uint16_t classify_token_text(const std::string& text) {
    if (text.empty()) return CAT_WHITESPACE;  // allow empty/special tokens

    uint16_t cat = 0;

    // Check if the entire token is a single JSON structural character
    if (text.size() == 1) {
        char c = text[0];
        switch (c) {
            case '{': cat |= CAT_OPEN_BRACE; break;
            case '}': cat |= CAT_CLOSE_BRACE; break;
            case '[': cat |= CAT_OPEN_BRACKET; break;
            case ']': cat |= CAT_CLOSE_BRACKET; break;
            case ':': cat |= CAT_COLON; break;
            case ',': cat |= CAT_COMMA; break;
            case '"': cat |= CAT_QUOTE; break;
            case ' ': case '\t': case '\n': case '\r':
                cat |= CAT_WHITESPACE; break;
            default: break;
        }
    }

    // Multi-char tokens: check if they start with structural chars
    if (text.size() >= 1) {
        char first = text[0];
        // Check if starts with { or [
        if (first == '{') cat |= CAT_OPEN_BRACE;
        if (first == '[') cat |= CAT_OPEN_BRACKET;
        if (first == '}') cat |= CAT_CLOSE_BRACE;
        if (first == ']') cat |= CAT_CLOSE_BRACKET;
        if (first == ':') cat |= CAT_COLON;
        if (first == ',') cat |= CAT_COMMA;
        if (first == '"') cat |= CAT_QUOTE;
    }

    // String content: tokens that are valid inside a JSON string
    // (any printable character, including escaped sequences)
    {
        bool all_string_safe = true;
        for (char c : text) {
            if (c == '"' || c == '\0') { all_string_safe = false; break; }
        }
        if (all_string_safe && !text.empty()) {
            cat |= CAT_STRING_CHAR;
        }
    }

    // Number: starts with digit or minus
    if (!text.empty()) {
        char first = text[0];
        if ((first >= '0' && first <= '9') || first == '-') {
            cat |= CAT_NUMBER_START;
        }
        // Number continuation
        bool all_num = true;
        for (char c : text) {
            if (!((c >= '0' && c <= '9') || c == '.' || c == 'e' ||
                  c == 'E' || c == '+' || c == '-')) {
                all_num = false;
                break;
            }
        }
        if (all_num && !text.empty()) {
            cat |= CAT_NUMBER_CONT;
        }
    }

    // Literal starts
    if (text.size() >= 1) {
        if (text[0] == 't') cat |= CAT_TRUE_START;
        if (text[0] == 'f') cat |= CAT_FALSE_START;
        if (text[0] == 'n') cat |= CAT_NULL_START;
    }

    // Literal continuation (for partial tokens like "ru", "ull", etc.)
    {
        bool is_literal_part = true;
        for (char c : text) {
            if (!(c == 'r' || c == 'u' || c == 'e' || c == 'a' ||
                  c == 'l' || c == 's')) {
                is_literal_part = false;
                break;
            }
        }
        if (is_literal_part && !text.empty()) {
            cat |= CAT_LITERAL_CONT;
        }
    }

    // Whitespace
    {
        bool all_ws = true;
        for (char c : text) {
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                all_ws = false;
                break;
            }
        }
        if (all_ws && !text.empty()) {
            cat |= CAT_WHITESPACE;
        }
    }

    return cat;
}

// ============================================================================
// JsonConstrainer implementation
// ============================================================================

JsonConstrainer::~JsonConstrainer() {
    if (d_token_categories_) {
        cudaFree(d_token_categories_);
        d_token_categories_ = nullptr;
    }
    if (d_allowed_mask_) {
        cudaFree(d_allowed_mask_);
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
        token_categories_[i] = classify_token_text(text);
    }

    // Upload to device
    cudaError_t err = cudaMalloc(&d_token_categories_,
                                  vocab_size_ * sizeof(uint16_t));
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("JsonConstrainer: failed to allocate device categories: %s",
                      cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(d_token_categories_, token_categories_.data(),
                      vocab_size_ * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("JsonConstrainer: failed to copy categories to device: %s",
                      cudaGetErrorString(err));
        return false;
    }

    // Allocate mask buffer
    err = cudaMalloc(&d_allowed_mask_, sizeof(uint16_t));
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("JsonConstrainer: failed to allocate mask buffer: %s",
                      cudaGetErrorString(err));
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
}

uint16_t JsonConstrainer::compute_allowed_mask() const {
    uint16_t mask = CAT_WHITESPACE;  // whitespace is always allowed

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
            mask |= CAT_NUMBER_CONT | CAT_COMMA |
                    CAT_CLOSE_BRACE | CAT_CLOSE_BRACKET;
            break;

        case JsonState::IN_LITERAL:
            // Inside literal (true/false/null): only literal continuation chars
            mask |= CAT_LITERAL_CONT;
            // If the literal is complete, also allow post-value tokens
            if (!target_literal_.empty() &&
                partial_literal_.size() >= target_literal_.size()) {
                mask |= CAT_COMMA | CAT_CLOSE_BRACE | CAT_CLOSE_BRACKET;
            }
            break;

        case JsonState::DONE:
            // Parsing complete — only allow EOS / whitespace
            break;

        default:
            // Fallback: allow everything
            mask = 0xFFFF;
            break;
    }

    return mask;
}

void JsonConstrainer::advance_char(char c) {
    // Skip whitespace in non-string states
    if (current_state_ != JsonState::IN_STRING &&
        current_state_ != JsonState::IN_STRING_ESCAPE &&
        (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
        return;
    }

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
                if (!state_stack_.empty()) state_stack_.pop_back();
                current_state_ = state_stack_.empty() ? JsonState::DONE :
                    state_stack_.back();
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
                if (!state_stack_.empty()) state_stack_.pop_back();
                current_state_ = state_stack_.empty() ? JsonState::DONE :
                    state_stack_.back();
            }
            break;

        case JsonState::ARRAY_START:
            if (c == ']') {
                if (!state_stack_.empty()) state_stack_.pop_back();
                current_state_ = state_stack_.empty() ? JsonState::DONE :
                    state_stack_.back();
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
                if (!state_stack_.empty()) state_stack_.pop_back();
                current_state_ = state_stack_.empty() ? JsonState::DONE :
                    state_stack_.back();
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
            if (!((c >= '0' && c <= '9') || c == '.' || c == 'e' ||
                  c == 'E' || c == '+' || c == '-')) {
                // Number ended — this char is part of the parent context
                // Pop back to parent and re-process this character
                current_state_ = state_stack_.empty() ? JsonState::DONE :
                    state_stack_.back();
                if (!state_stack_.empty()) state_stack_.pop_back();
                advance_char(c);  // re-process
                return;
            }
            break;

        case JsonState::IN_LITERAL:
            partial_literal_ += c;
            if (partial_literal_.size() >= target_literal_.size()) {
                // Literal complete — transition to parent
                current_state_ = state_stack_.empty() ? JsonState::DONE :
                    state_stack_.back();
                if (!state_stack_.empty()) state_stack_.pop_back();
            }
            break;

        case JsonState::DONE:
            break;
    }
}

void JsonConstrainer::update(int32_t token) {
    if (token < 0 || token >= vocab_size_) return;
    const std::string& text = token_texts_[token];
    for (char c : text) {
        advance_char(c);
    }
}

void JsonConstrainer::apply_mask(float* d_logits, int vocab_size, cudaStream_t stream) {
    if (!initialized_ || !d_token_categories_ || !d_allowed_mask_) return;

    uint16_t mask = compute_allowed_mask();

    // Upload mask to device
    cudaMemcpyAsync(d_allowed_mask_, &mask, sizeof(uint16_t),
                     cudaMemcpyHostToDevice, stream);

    // Launch masking kernel
    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    json_mask_kernel<<<blocks, threads, 0, stream>>>(
        d_logits, d_token_categories_, d_allowed_mask_, vocab_size);
}

} // namespace imp
