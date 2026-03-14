#include "compute/schema_constrain.h"
#include "compute/json_constrain.h"  // reuse token category definitions
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <algorithm>
#include <cstring>

namespace imp {

// ---------------------------------------------------------------------------
// GPU kernel: dual mask — category bitmask + per-token allow
// ---------------------------------------------------------------------------
__global__ void schema_mask_kernel(
    float* __restrict__ logits,
    const uint16_t* __restrict__ token_cats,
    const uint8_t* __restrict__ token_allow,
    const uint16_t* __restrict__ allowed_mask,
    int vocab_size,
    bool use_token_allow)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) return;

    uint16_t mask = *allowed_mask;
    bool cat_ok = (token_cats[idx] & mask) != 0;

    if (use_token_allow) {
        // Schema mode: token must pass category mask AND token_allow.
        // token_allow further restricts which tokens are valid (e.g., only
        // property name prefixes, not arbitrary strings).
        bool allow_ok = token_allow[idx] != 0;
        if (!cat_ok || !allow_ok) {
            logits[idx] = -FLT_MAX;
        }
    } else {
        if (!cat_ok) {
            logits[idx] = -FLT_MAX;
        }
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

SchemaConstrainer::~SchemaConstrainer() {
    if (d_token_categories_) cudaFree(d_token_categories_);
    if (d_token_allow_) cudaFree(d_token_allow_);
    if (d_allowed_mask_) cudaFree(d_allowed_mask_);
}

bool SchemaConstrainer::init(const Tokenizer& tok, std::unique_ptr<SchemaNode> schema) {
    schema_ = std::move(schema);
    if (!schema_) return false;

    vocab_size_ = tok.vocab_size();

    // Classify all tokens (same logic as JsonConstrainer)
    token_categories_.resize(vocab_size_, 0);
    token_texts_.resize(vocab_size_);
    token_allow_.resize(vocab_size_, 1);

    for (int i = 0; i < vocab_size_; i++) {
        std::string text = tok.decode_token(i);
        token_texts_[i] = text;

        if (text.empty()) continue;

        uint16_t cat = 0;
        char first = text[0];
        bool all_same = true;
        for (size_t j = 1; j < text.size(); j++) {
            if (text[j] != first) { all_same = false; break; }
        }

        // Single-character tokens get precise categories
        if (text.size() == 1) {
            switch (first) {
                case '{': cat |= CAT_OPEN_BRACE; break;
                case '}': cat |= CAT_CLOSE_BRACE; break;
                case '[': cat |= CAT_OPEN_BRACKET; break;
                case ']': cat |= CAT_CLOSE_BRACKET; break;
                case ':': cat |= CAT_COLON; break;
                case ',': cat |= CAT_COMMA; break;
                case '"': cat |= CAT_QUOTE; break;
                case 't': cat |= CAT_TRUE_START | CAT_STRING_CHAR | CAT_LITERAL_CONT; break;
                case 'f': cat |= CAT_FALSE_START | CAT_STRING_CHAR | CAT_LITERAL_CONT; break;
                case 'n': cat |= CAT_NULL_START | CAT_STRING_CHAR | CAT_LITERAL_CONT; break;
                default: break;
            }
            if (first >= '0' && first <= '9')
                cat |= CAT_NUMBER_START | CAT_NUMBER_CONT | CAT_STRING_CHAR;
            if (first == '-')
                cat |= CAT_NUMBER_START | CAT_NUMBER_CONT | CAT_STRING_CHAR;
            if (first == '.' || first == 'e' || first == 'E' || first == '+')
                cat |= CAT_NUMBER_CONT | CAT_STRING_CHAR;
            if (first == ' ' || first == '\t' || first == '\n' || first == '\r')
                cat |= CAT_WHITESPACE;
            // General string chars
            if (first >= 32 && first != '"' && first != '\\')
                cat |= CAT_STRING_CHAR;
            // Literal continuation
            if (std::strchr("ruealskl", first))
                cat |= CAT_LITERAL_CONT;
        } else {
            // Multi-character tokens
            bool is_ws = true;
            bool is_str = true;
            bool is_num = true;
            bool is_lit = true;
            for (char c : text) {
                if (c != ' ' && c != '\t' && c != '\n' && c != '\r') is_ws = false;
                if (c < 32 || c == '"' || c == '\\') is_str = false;
                if (!std::strchr("0123456789.-+eE", c)) is_num = false;
                if (!std::islower(static_cast<unsigned char>(c))) is_lit = false;
            }
            if (is_ws) cat |= CAT_WHITESPACE;
            if (is_str) cat |= CAT_STRING_CHAR;
            if (is_num) {
                cat |= CAT_NUMBER_CONT;
                if (first >= '0' && first <= '9') cat |= CAT_NUMBER_START;
                if (first == '-') cat |= CAT_NUMBER_START;
            }
            if (is_lit) {
                cat |= CAT_LITERAL_CONT;
                if (first == 't') cat |= CAT_TRUE_START;
                if (first == 'f') cat |= CAT_FALSE_START;
                if (first == 'n') cat |= CAT_NULL_START;
            }
        }

        token_categories_[i] = cat;
    }

    // Upload to GPU
    cudaMalloc(&d_token_categories_, vocab_size_ * sizeof(uint16_t));
    cudaMemcpy(d_token_categories_, token_categories_.data(),
               vocab_size_ * sizeof(uint16_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_token_allow_, vocab_size_ * sizeof(uint8_t));
    cudaMalloc(&d_allowed_mask_, sizeof(uint16_t));

    reset();
    initialized_ = true;

    IMP_LOG_INFO("SchemaConstrainer: initialized with %d tokens, schema type=%d",
                 vocab_size_, static_cast<int>(schema_->type));
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
}

void SchemaConstrainer::push_value_frame(const SchemaNode* node) {
    SchemaFrame frame;
    frame.node = node;
    frame.phase = SchemaPhase::VALUE_START;
    stack_.push_back(std::move(frame));
}

// ---------------------------------------------------------------------------
// Property / enum helpers
// ---------------------------------------------------------------------------

const SchemaNode* SchemaConstrainer::find_property(const SchemaNode* obj,
                                                     const std::string& key) const {
    for (auto& [name, schema] : obj->properties) {
        if (name == key) return schema.get();
    }
    return nullptr;
}

bool SchemaConstrainer::is_valid_key_prefix(const SchemaNode* obj, const std::string& prefix,
                                             const std::set<std::string>& emitted) const {
    for (auto& [name, _] : obj->properties) {
        if (emitted.count(name)) continue;
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

// ---------------------------------------------------------------------------
// Compute category mask from schema FSM state
// ---------------------------------------------------------------------------

uint16_t SchemaConstrainer::compute_category_mask() const {
    if (stack_.empty()) return CAT_WHITESPACE;

    const auto& f = top();
    switch (f.phase) {
        case SchemaPhase::VALUE_START: {
            uint16_t mask = CAT_WHITESPACE;
            if (!f.node) return mask | CAT_VALUE_START;
            switch (f.node->type) {
                case SchemaType::OBJECT:  mask |= CAT_OPEN_BRACE; break;
                case SchemaType::ARRAY:   mask |= CAT_OPEN_BRACKET; break;
                case SchemaType::STRING:  mask |= CAT_QUOTE; break;
                case SchemaType::NUMBER:
                case SchemaType::INTEGER: mask |= CAT_NUMBER_START; break;
                case SchemaType::BOOLEAN: mask |= CAT_TRUE_START | CAT_FALSE_START; break;
                case SchemaType::NULL_TYPE: mask |= CAT_NULL_START; break;
                case SchemaType::ENUM:    mask |= CAT_QUOTE; break;  // enum values are strings
                case SchemaType::ANY_OF:  mask |= CAT_VALUE_START; break;
                default: mask |= CAT_VALUE_START; break;
            }
            return mask;
        }

        case SchemaPhase::OBJECT_OPEN: {
            uint16_t mask = CAT_WHITESPACE | CAT_QUOTE;  // " for key
            // Allow } only if all required keys are emitted
            bool all_required = true;
            if (f.node) {
                for (auto& req : f.node->required) {
                    if (!f.emitted_keys.count(req)) { all_required = false; break; }
                }
            }
            if (all_required) mask |= CAT_CLOSE_BRACE;
            return mask;
        }

        case SchemaPhase::OBJECT_KEY:
            return CAT_STRING_CHAR | CAT_QUOTE;  // token_allow handles key constraining

        case SchemaPhase::OBJECT_AFTER_KEY:
            return CAT_COLON | CAT_WHITESPACE;

        case SchemaPhase::OBJECT_COLON:
            return CAT_COLON | CAT_WHITESPACE;

        case SchemaPhase::OBJECT_AFTER_VALUE: {
            uint16_t mask = CAT_WHITESPACE | CAT_COMMA;
            bool all_required = true;
            if (f.node) {
                for (auto& req : f.node->required) {
                    if (!f.emitted_keys.count(req)) { all_required = false; break; }
                }
            }
            if (all_required) mask |= CAT_CLOSE_BRACE;
            return mask;
        }

        case SchemaPhase::ARRAY_OPEN: {
            uint16_t mask = CAT_WHITESPACE | CAT_CLOSE_BRACKET;
            // Allow value start for first item
            if (f.node && f.node->items) {
                switch (f.node->items->type) {
                    case SchemaType::STRING: mask |= CAT_QUOTE; break;
                    case SchemaType::NUMBER:
                    case SchemaType::INTEGER: mask |= CAT_NUMBER_START; break;
                    case SchemaType::BOOLEAN: mask |= CAT_TRUE_START | CAT_FALSE_START; break;
                    case SchemaType::OBJECT: mask |= CAT_OPEN_BRACE; break;
                    case SchemaType::ARRAY: mask |= CAT_OPEN_BRACKET; break;
                    default: mask |= CAT_VALUE_START; break;
                }
            } else {
                mask |= CAT_VALUE_START;
            }
            return mask;
        }

        case SchemaPhase::ARRAY_AFTER_ITEM:
            return CAT_WHITESPACE | CAT_COMMA | CAT_CLOSE_BRACKET;

        case SchemaPhase::STRING_VALUE:
            return CAT_STRING_CHAR | CAT_QUOTE;

        case SchemaPhase::STRING_ESCAPE:
            return 0xFFFF;  // any char valid after backslash

        case SchemaPhase::NUMBER_VALUE:
            return CAT_NUMBER_CONT | CAT_WHITESPACE | CAT_COMMA |
                   CAT_CLOSE_BRACE | CAT_CLOSE_BRACKET;

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

void SchemaConstrainer::compute_token_allow_mask() {
    need_token_allow_ = false;

    if (stack_.empty()) return;
    const auto& f = top();

    if (f.phase == SchemaPhase::OBJECT_KEY && f.node) {
        // Constrain tokens to valid property name prefixes/completions
        need_token_allow_ = true;
        const auto& prefix = f.key_buffer;

        for (int i = 0; i < vocab_size_; i++) {
            const auto& text = token_texts_[i];
            if (text.empty()) { token_allow_[i] = 0; continue; }

            // Check if this token could be part of a valid key
            std::string extended = prefix + text;
            bool valid = false;

            // Check if token is the closing quote
            if (text == "\"") {
                // Allow quote only if current prefix is a complete valid key
                for (auto& [name, _] : f.node->properties) {
                    if (!f.emitted_keys.count(name) && name == prefix) {
                        valid = true;
                        break;
                    }
                }
            } else {
                // Check if extended prefix matches any remaining property name
                // Handle tokens that contain the closing quote (e.g. `name"`)
                bool has_quote = text.find('"') != std::string::npos;
                if (has_quote) {
                    // Token contains quote — check if text before quote is a complete key
                    size_t qpos = text.find('"');
                    std::string key_part = prefix + text.substr(0, qpos);
                    for (auto& [name, _] : f.node->properties) {
                        if (!f.emitted_keys.count(name) && name == key_part) {
                            valid = true;
                            break;
                        }
                    }
                } else {
                    valid = is_valid_key_prefix(f.node, extended, f.emitted_keys);
                }
            }

            token_allow_[i] = valid ? 1 : 0;
        }
    } else if (f.phase == SchemaPhase::ENUM_VALUE && f.node) {
        // Constrain tokens to valid enum value prefixes
        need_token_allow_ = true;
        const auto& prefix = f.enum_buffer;

        for (int i = 0; i < vocab_size_; i++) {
            const auto& text = token_texts_[i];
            if (text.empty()) { token_allow_[i] = 0; continue; }

            if (text == "\"") {
                // Allow closing quote only if prefix is a complete enum value
                bool complete = false;
                for (auto& v : f.node->enum_values) {
                    if (v == prefix) { complete = true; break; }
                }
                token_allow_[i] = complete ? 1 : 0;
            } else {
                std::string extended = prefix + text;
                token_allow_[i] = is_valid_enum_prefix(f.node->enum_values, extended) ? 1 : 0;
            }
        }
    } else if (f.phase == SchemaPhase::OBJECT_OPEN && f.node) {
        // When we need to open with a quote for key, also constrain which
        // quote tokens start valid keys (for multi-char tokens like `"name`)
        // Only activate if there are tokens that start with quote + key prefix
        // For single " token, category mask handles it. No need for token_allow.
    }
}

// ---------------------------------------------------------------------------
// Apply mask to logits
// ---------------------------------------------------------------------------

void SchemaConstrainer::apply_mask(float* d_logits, int vocab_size, cudaStream_t stream) {
    if (!initialized_ || stack_.empty()) return;

    // Compute masks
    uint16_t cat_mask = compute_category_mask();
    compute_token_allow_mask();

    IMP_LOG_DEBUG("SchemaConstrainer::apply_mask phase=%d cat_mask=0x%04x need_allow=%d stack=%zu",
                  static_cast<int>(top().phase), cat_mask, need_token_allow_,
                  stack_.size());

    // Upload category mask
    cudaMemcpyAsync(d_allowed_mask_, &cat_mask, sizeof(uint16_t),
                    cudaMemcpyHostToDevice, stream);

    // Upload token allow mask if needed
    if (need_token_allow_) {
        cudaMemcpyAsync(d_token_allow_, token_allow_.data(),
                        vocab_size_ * sizeof(uint8_t),
                        cudaMemcpyHostToDevice, stream);
    }

    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    schema_mask_kernel<<<blocks, threads, 0, stream>>>(
        d_logits, d_token_categories_, d_token_allow_, d_allowed_mask_,
        vocab_size, need_token_allow_);
}

// ---------------------------------------------------------------------------
// Update FSM with sampled token
// ---------------------------------------------------------------------------

void SchemaConstrainer::update(int32_t token) {
    if (token < 0 || token >= vocab_size_ || stack_.empty()) return;

    const auto& text = token_texts_[token];
    SchemaPhase before = top().phase;
    for (char c : text) {
        advance_char(c);
    }
    if (!stack_.empty()) {
        IMP_LOG_DEBUG("SchemaConstrainer::update token=%d [%s] phase %d->%d stack=%zu",
                      token, text.c_str(), static_cast<int>(before),
                      static_cast<int>(top().phase), stack_.size());
    }
}

void SchemaConstrainer::advance_char(char c) {
    if (stack_.empty()) return;

    auto& f = top();

    switch (f.phase) {
        case SchemaPhase::VALUE_START: {
            if (std::isspace(static_cast<unsigned char>(c))) return;
            if (!f.node) { stack_.pop_back(); return; }

            switch (f.node->type) {
                case SchemaType::OBJECT:
                    if (c == '{') { f.phase = SchemaPhase::OBJECT_OPEN; return; }
                    break;
                case SchemaType::ARRAY:
                    if (c == '[') { f.phase = SchemaPhase::ARRAY_OPEN; return; }
                    break;
                case SchemaType::STRING:
                    if (c == '"') { f.phase = SchemaPhase::STRING_VALUE; return; }
                    break;
                case SchemaType::NUMBER:
                case SchemaType::INTEGER:
                    if (c == '-' || (c >= '0' && c <= '9')) {
                        f.phase = SchemaPhase::NUMBER_VALUE;
                        return;
                    }
                    break;
                case SchemaType::BOOLEAN:
                    if (c == 't') {
                        f.phase = SchemaPhase::LITERAL_VALUE;
                        f.literal_target = "true";
                        f.literal_pos = 1;
                        return;
                    }
                    if (c == 'f') {
                        f.phase = SchemaPhase::LITERAL_VALUE;
                        f.literal_target = "false";
                        f.literal_pos = 1;
                        return;
                    }
                    break;
                case SchemaType::NULL_TYPE:
                    if (c == 'n') {
                        f.phase = SchemaPhase::LITERAL_VALUE;
                        f.literal_target = "null";
                        f.literal_pos = 1;
                        return;
                    }
                    break;
                case SchemaType::ENUM:
                    if (c == '"') {
                        f.phase = SchemaPhase::ENUM_VALUE;
                        f.enum_buffer.clear();
                        return;
                    }
                    break;
                case SchemaType::ANY_OF:
                    // For anyOf, we can't easily constrain — treat as unconstrained value
                    f.phase = SchemaPhase::STRING_VALUE;
                    return;
                default: break;
            }
            break;
        }

        case SchemaPhase::OBJECT_OPEN: {
            if (std::isspace(static_cast<unsigned char>(c))) return;
            if (c == '}') {
                stack_.pop_back();  // object complete
                if (!stack_.empty()) {
                    auto& parent = top();
                    if (parent.phase == SchemaPhase::OBJECT_COLON)
                        parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
                    else if (parent.phase == SchemaPhase::ARRAY_OPEN ||
                             parent.phase == SchemaPhase::ARRAY_AFTER_ITEM)
                        parent.phase = SchemaPhase::ARRAY_AFTER_ITEM;
                }
                return;
            }
            if (c == '"') {
                f.phase = SchemaPhase::OBJECT_KEY;
                f.key_buffer.clear();
                return;
            }
            break;
        }

        case SchemaPhase::OBJECT_KEY: {
            if (c == '"') {
                // Key complete
                f.current_key = f.key_buffer;
                f.emitted_keys.insert(f.current_key);
                f.phase = SchemaPhase::OBJECT_AFTER_KEY;
                return;
            }
            if (c == '\\') return;  // escape in key (rare but valid)
            f.key_buffer += c;
            return;
        }

        case SchemaPhase::OBJECT_AFTER_KEY: {
            if (std::isspace(static_cast<unsigned char>(c))) return;
            if (c == ':') {
                f.phase = SchemaPhase::OBJECT_COLON;
                // Push value frame for this property
                const SchemaNode* prop_schema = find_property(f.node, f.current_key);
                if (prop_schema) {
                    push_value_frame(prop_schema);
                } else {
                    // Unknown property — allow any value (permissive)
                    push_value_frame(nullptr);
                }
                return;
            }
            break;
        }

        case SchemaPhase::OBJECT_COLON: {
            // Value was pushed as a sub-frame — this phase is re-entered
            // when the value completes and the sub-frame is popped.
            if (std::isspace(static_cast<unsigned char>(c))) return;
            // Should not reach here in normal flow (value frame handles it)
            // but handle comma/brace for robustness
            if (c == ',') {
                f.phase = SchemaPhase::OBJECT_OPEN;  // next key
                return;
            }
            if (c == '}') {
                stack_.pop_back();
                if (!stack_.empty()) {
                    auto& parent = top();
                    if (parent.phase == SchemaPhase::OBJECT_COLON)
                        parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
                }
                return;
            }
            break;
        }

        case SchemaPhase::OBJECT_AFTER_VALUE: {
            if (std::isspace(static_cast<unsigned char>(c))) return;
            if (c == ',') {
                f.phase = SchemaPhase::OBJECT_OPEN;  // back to expecting key
                return;
            }
            if (c == '}') {
                stack_.pop_back();
                if (!stack_.empty()) {
                    auto& parent = top();
                    if (parent.phase == SchemaPhase::OBJECT_COLON)
                        parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
                    else if (parent.phase == SchemaPhase::ARRAY_OPEN ||
                             parent.phase == SchemaPhase::ARRAY_AFTER_ITEM)
                        parent.phase = SchemaPhase::ARRAY_AFTER_ITEM;
                }
                return;
            }
            break;
        }

        case SchemaPhase::ARRAY_OPEN: {
            if (std::isspace(static_cast<unsigned char>(c))) return;
            if (c == ']') {
                stack_.pop_back();
                if (!stack_.empty()) {
                    auto& parent = top();
                    if (parent.phase == SchemaPhase::OBJECT_COLON)
                        parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
                }
                return;
            }
            // Start first item — push item schema frame
            if (f.node && f.node->items) {
                push_value_frame(f.node->items.get());
                // Re-process this char in the new frame
                advance_char(c);
            }
            return;
        }

        case SchemaPhase::ARRAY_AFTER_ITEM: {
            if (std::isspace(static_cast<unsigned char>(c))) return;
            if (c == ',') {
                f.item_count++;
                // Push next item frame
                if (f.node && f.node->items) {
                    push_value_frame(f.node->items.get());
                }
                return;
            }
            if (c == ']') {
                stack_.pop_back();
                if (!stack_.empty()) {
                    auto& parent = top();
                    if (parent.phase == SchemaPhase::OBJECT_COLON)
                        parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
                }
                return;
            }
            break;
        }

        case SchemaPhase::STRING_VALUE: {
            if (c == '\\') { f.phase = SchemaPhase::STRING_ESCAPE; return; }
            if (c == '"') {
                // String value complete
                stack_.pop_back();
                if (!stack_.empty()) {
                    auto& parent = top();
                    if (parent.phase == SchemaPhase::OBJECT_COLON)
                        parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
                    else if (parent.phase == SchemaPhase::ARRAY_OPEN ||
                             parent.phase == SchemaPhase::ARRAY_AFTER_ITEM)
                        parent.phase = SchemaPhase::ARRAY_AFTER_ITEM;
                }
                return;
            }
            return;  // accumulate string content
        }

        case SchemaPhase::STRING_ESCAPE: {
            f.phase = SchemaPhase::STRING_VALUE;
            return;
        }

        case SchemaPhase::NUMBER_VALUE: {
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' ||
                c == 'E' || c == '+' || c == '-')
                return;  // continue number
            // Number ended — pop and re-process char in parent
            stack_.pop_back();
            if (!stack_.empty()) {
                auto& parent = top();
                if (parent.phase == SchemaPhase::OBJECT_COLON)
                    parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
                else if (parent.phase == SchemaPhase::ARRAY_OPEN ||
                         parent.phase == SchemaPhase::ARRAY_AFTER_ITEM)
                    parent.phase = SchemaPhase::ARRAY_AFTER_ITEM;
                advance_char(c);
            }
            return;
        }

        case SchemaPhase::LITERAL_VALUE: {
            if (f.literal_pos < static_cast<int>(f.literal_target.size())) {
                f.literal_pos++;
                if (f.literal_pos >= static_cast<int>(f.literal_target.size())) {
                    // Literal complete
                    stack_.pop_back();
                    if (!stack_.empty()) {
                        auto& parent = top();
                        if (parent.phase == SchemaPhase::OBJECT_COLON)
                            parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
                        else if (parent.phase == SchemaPhase::ARRAY_OPEN ||
                                 parent.phase == SchemaPhase::ARRAY_AFTER_ITEM)
                            parent.phase = SchemaPhase::ARRAY_AFTER_ITEM;
                    }
                }
            }
            return;
        }

        case SchemaPhase::ENUM_VALUE: {
            if (c == '"') {
                // Enum value complete
                stack_.pop_back();
                if (!stack_.empty()) {
                    auto& parent = top();
                    if (parent.phase == SchemaPhase::OBJECT_COLON)
                        parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
                    else if (parent.phase == SchemaPhase::ARRAY_OPEN ||
                             parent.phase == SchemaPhase::ARRAY_AFTER_ITEM)
                        parent.phase = SchemaPhase::ARRAY_AFTER_ITEM;
                }
                return;
            }
            f.enum_buffer += c;
            return;
        }

        case SchemaPhase::DONE:
            return;
    }
}

} // namespace imp
