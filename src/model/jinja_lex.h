// Jinja2 tokens: lexer output, parser input.
#pragma once

#include <string>
#include <vector>

namespace imp::jinja::detail {

enum class TokenType {
    TEXT,
    EXPR_OPEN,   // {{
    EXPR_CLOSE,  // }}
    STMT_OPEN,   // {%
    STMT_CLOSE,  // %}
    IDENT,
    STRING,
    NUMBER,
    OP,
    COMMA,
    DOT,
    LBRACKET,
    LBRACE,  // { of a dict literal
    RBRACE,
    RBRACKET,
    LPAREN,
    RPAREN,
    PIPE,
    COLON,
    ASSIGN,
    END,
};

struct Token {
    TokenType type{};
    std::string value;
    bool trim_left = false;   // whitespace control: strip left
    bool trim_right = false;  // whitespace control: strip right
    // TEXT-only: true when this token began at the start of a source line. Needed for
    // lstrip_blocks since trim_blocks usually already ate the newline, so pure indentation
    // can't tell line-indent from line-tail on its own.
    bool line_start = false;
};

// Splits template source into TEXT / {{ }} / {% %} tokens, whitespace control applied.
std::vector<Token> tokenize(const std::string& source);

}  // namespace imp::jinja::detail
