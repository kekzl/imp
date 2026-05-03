#pragma once

// Shared token classification for JSON and schema constrainers.
// Both JsonConstrainer and SchemaConstrainer classify vocabulary tokens
// into category bitmasks. This header provides the shared classification
// function and the common mask kernel.

#include <cstdint>
#include <cstring>
#include <string>
#include <cfloat>
#include <cuda_runtime.h>

namespace imp {

// Token category flags are defined in json_constrain.h (JsonTokenCat enum).
// Both constrainers share those flags. This header assumes they are visible
// via the includer's own #include of json_constrain.h or schema_constrain.h.

// ---------------------------------------------------------------------------
// Shared token classification: maps token text to a category bitmask.
// ---------------------------------------------------------------------------

static inline uint16_t classify_token(const std::string& text) {
    if (text.empty())
        return CAT_WHITESPACE;  // allow empty/special tokens

    uint16_t cat = 0;
    char first = text[0];

    if (text.size() == 1) {
        // Single-character tokens get precise structural categories
        switch (first) {
            case '{':
                cat |= CAT_OPEN_BRACE;
                break;
            case '}':
                cat |= CAT_CLOSE_BRACE;
                break;
            case '[':
                cat |= CAT_OPEN_BRACKET;
                break;
            case ']':
                cat |= CAT_CLOSE_BRACKET;
                break;
            case ':':
                cat |= CAT_COLON;
                break;
            case ',':
                cat |= CAT_COMMA;
                break;
            case '"':
                cat |= CAT_QUOTE;
                break;
            case 't':
                cat |= CAT_TRUE_START | CAT_STRING_CHAR | CAT_LITERAL_CONT;
                break;
            case 'f':
                cat |= CAT_FALSE_START | CAT_STRING_CHAR | CAT_LITERAL_CONT;
                break;
            case 'n':
                cat |= CAT_NULL_START | CAT_STRING_CHAR | CAT_LITERAL_CONT;
                break;
            default:
                break;
        }
        if (first >= '0' && first <= '9')
            cat |= CAT_NUMBER_START | CAT_NUMBER_CONT | CAT_STRING_CHAR;
        if (first == '-')
            cat |= CAT_NUMBER_START | CAT_NUMBER_CONT | CAT_STRING_CHAR;
        if (first == '.' || first == 'e' || first == 'E' || first == '+')
            cat |= CAT_NUMBER_CONT | CAT_STRING_CHAR;
        if (first == ' ' || first == '\t' || first == '\n' || first == '\r')
            cat |= CAT_WHITESPACE;
        // General string chars (printable, not quote or backslash)
        if (first >= 32 && first != '"' && first != '\\')
            cat |= CAT_STRING_CHAR;
        // Literal continuation characters
        if (std::strchr("ruealskl", first))
            cat |= CAT_LITERAL_CONT;
    } else {
        // Multi-character tokens: check whole-token properties
        bool is_ws = true, is_str = true, is_num = true, is_lit = true;
        for (char c : text) {
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
                is_ws = false;
            if (c < 32 || c == '"' || c == '\\')
                is_str = false;
            if (!std::strchr("0123456789.-+eE", c))
                is_num = false;
            if (!std::islower(static_cast<unsigned char>(c)))
                is_lit = false;
        }

        // Also tag multi-char tokens that start with structural chars
        if (first == '{')
            cat |= CAT_OPEN_BRACE;
        if (first == '}')
            cat |= CAT_CLOSE_BRACE;
        if (first == '[')
            cat |= CAT_OPEN_BRACKET;
        if (first == ']')
            cat |= CAT_CLOSE_BRACKET;
        if (first == ':')
            cat |= CAT_COLON;
        if (first == ',')
            cat |= CAT_COMMA;
        if (first == '"')
            cat |= CAT_QUOTE;

        if (is_ws)
            cat |= CAT_WHITESPACE;
        if (is_str)
            cat |= CAT_STRING_CHAR;
        if (is_num) {
            cat |= CAT_NUMBER_CONT;
            if (first >= '0' && first <= '9')
                cat |= CAT_NUMBER_START;
            if (first == '-')
                cat |= CAT_NUMBER_START;
        }
        if (is_lit) {
            cat |= CAT_LITERAL_CONT;
            if (first == 't')
                cat |= CAT_TRUE_START;
            if (first == 'f')
                cat |= CAT_FALSE_START;
            if (first == 'n')
                cat |= CAT_NULL_START;
        }
    }

    return cat;
}

// ---------------------------------------------------------------------------
// Shared GPU kernel: apply category bitmask to logits.
// Sets logits to -FLT_MAX for tokens whose category doesn't match the mask.
// ---------------------------------------------------------------------------

__global__ inline void constrain_mask_kernel(float* __restrict__ logits,
                                             const uint16_t* __restrict__ token_cats,
                                             const uint16_t* __restrict__ allowed_mask, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        if ((token_cats[idx] & *allowed_mask) == 0) {
            logits[idx] = -FLT_MAX;
        }
    }
}

// ---------------------------------------------------------------------------
// Extended mask kernel with per-token allow (for schema constraining).
// Tokens must pass BOTH category mask AND token_allow when use_token_allow
// is true.
// ---------------------------------------------------------------------------

__global__ inline void constrain_mask_allow_kernel(float* __restrict__ logits,
                                                   const uint16_t* __restrict__ token_cats,
                                                   const uint8_t* __restrict__ token_allow,
                                                   const uint16_t* __restrict__ allowed_mask, int vocab_size,
                                                   bool use_token_allow) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size)
        return;

    uint16_t mask = *allowed_mask;
    bool cat_ok = (token_cats[idx] & mask) != 0;

    if (use_token_allow) {
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

}  // namespace imp
