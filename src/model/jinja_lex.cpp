// Jinja2 lexer: template source -> tokens (jinja_lex.h).

#include "model/jinja_lex.h"

#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace imp::jinja::detail {

namespace {

class Lexer {
public:
    explicit Lexer(const std::string& src) : src_(src) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        size_t pos = 0;
        while (pos < src_.size()) {
            // Check for comment {# ... #}
            if (pos + 1 < src_.size() && src_[pos] == '{' && src_[pos + 1] == '#') {
                // Find closing #}
                bool trim_l = (pos + 2 < src_.size() && src_[pos + 2] == '-');
                size_t end = src_.find("#}", pos + 2);
                if (end == std::string::npos)
                    end = src_.size();
                bool trim_r = (end > 0 && src_[end - 1] == '-');
                if (trim_r)
                    end--;                   // skip the -
                end = src_.find("#}", end);  // re-find past the -
                if (end == std::string::npos)
                    end = src_.size() - 2;
                pos = end + 2;
                // Apply trim: strip trailing whitespace from last TEXT token
                if (trim_l && !tokens.empty() && tokens.back().type == TokenType::TEXT)
                    rtrim(tokens.back().value);
                // A comment is a block tag for lstrip_blocks/trim_blocks too (#1572). Skipping it left the
                // comment line's indentation and newline in the prompt (Nemotron-3-Nano put 20 spaces
                // before <|im_start|>assistant while other families rendered byte-exact).
                if (!trim_l && !tokens.empty() && tokens.back().type == TokenType::TEXT)
                    lstrip_block_indent(tokens.back().value, tokens.back().line_start);
                // Skip leading whitespace after comment
                if (trim_r) {
                    while (pos < src_.size() &&
                           (src_[pos] == ' ' || src_[pos] == '\t' || src_[pos] == '\n' || src_[pos] == '\r'))
                        pos++;
                } else {
                    if (pos < src_.size() && src_[pos] == '\r')
                        pos++;
                    if (pos < src_.size() && src_[pos] == '\n')
                        pos++;
                }
                continue;
            }

            // Check for {{ or {%
            if (pos + 1 < src_.size() && src_[pos] == '{' && (src_[pos + 1] == '{' || src_[pos + 1] == '%')) {
                bool is_expr = (src_[pos + 1] == '{');
                pos += 2;

                // Check trim marker
                bool trim_l = false;
                if (pos < src_.size() && src_[pos] == '-') {
                    trim_l = true;
                    pos++;
                }

                // Strip trailing whitespace from previous TEXT token
                if (trim_l && !tokens.empty() && tokens.back().type == TokenType::TEXT)
                    rtrim(tokens.back().value);

                // lstrip_blocks (#1572): whitespace from line-start up to a block tag is indentation, not
                // output. transformers sets lstrip_blocks=True; without this an indented {% if %} leaks
                // its indentation into the prompt.
                if (!is_expr && !trim_l && !tokens.empty() && tokens.back().type == TokenType::TEXT)
                    lstrip_block_indent(tokens.back().value, tokens.back().line_start);

                Token open;
                open.type = is_expr ? TokenType::EXPR_OPEN : TokenType::STMT_OPEN;
                open.trim_left = trim_l;
                tokens.push_back(open);

                // Tokenize contents until closing tag
                std::string close_tag = is_expr ? "}}" : "%}";
                tokenize_inner(tokens, pos, close_tag);

                continue;
            }

            // Plain text
            size_t text_start = pos;
            while (pos < src_.size()) {
                if (src_[pos] == '{' && pos + 1 < src_.size() &&
                    (src_[pos + 1] == '{' || src_[pos + 1] == '%' || src_[pos + 1] == '#'))
                    break;
                pos++;
            }
            if (pos > text_start) {
                Token t;
                t.type = TokenType::TEXT;
                t.value = src_.substr(text_start, pos - text_start);
                t.line_start = (text_start == 0 || src_[text_start - 1] == '\n');
                tokens.push_back(t);
            }
        }
        tokens.push_back({TokenType::END, ""});
        return tokens;
    }

private:
    // Drops a trailing run of spaces/tabs only when it is the WHOLE line indentation (matches
    // Jinja's lstrip_blocks). `line_start` distinguishes "\n    {% if %}" (dropped) from
    // "{{ x }} {% endfor %}" (kept, since the run isn't at the line front).
    static void lstrip_block_indent(std::string& text, bool line_start) {
        size_t i = text.size();
        while (i > 0 && (text[i - 1] == ' ' || text[i - 1] == '\t'))
            i--;
        if (i == text.size())
            return;
        if (i > 0 ? text[i - 1] == '\n' : line_start)
            text.erase(i);
    }

    // Quoted literal at pos (opening quote), escapes \n \t \r \\ \' \"; pos ends past the closing quote.
    std::string lex_string_(size_t& pos) const {
        char q = src_[pos++];
        std::string s;
        while (pos < src_.size() && src_[pos] != q) {
            if (src_[pos] == '\\' && pos + 1 < src_.size()) {
                pos++;
                switch (src_[pos]) {
                    case 'n':
                        s += '\n';
                        break;
                    case 't':
                        s += '\t';
                        break;
                    case 'r':
                        s += '\r';
                        break;
                    case '\\':
                        s += '\\';
                        break;
                    case '\'':
                        s += '\'';
                        break;
                    case '"':
                        s += '"';
                        break;
                    default:
                        s += '\\';
                        s += src_[pos];
                        break;
                }
            } else {
                s += src_[pos];
            }
            pos++;
        }
        if (pos < src_.size())
            pos++;  // skip closing quote
        return s;
    }

    void tokenize_inner(std::vector<Token>& tokens, size_t& pos, const std::string& close_tag) {
        skip_ws(pos);
        int brace_depth = 0;  // open dict literals: their '}' is not the start of "}}"
        while (pos < src_.size()) {
            if (brace_depth > 0 && src_[pos] == '}') {
                tokens.push_back({TokenType::RBRACE, "}"});
                --brace_depth;
                skip_ws(++pos);
                continue;
            }
            // Check for trim marker + close tag
            if (pos + close_tag.size() <= src_.size()) {
                bool trim_r = false;
                size_t check = pos;
                if (src_[check] == '-' && check + 1 + close_tag.size() <= src_.size()) {
                    if (src_.substr(check + 1, close_tag.size()) == close_tag) {
                        trim_r = true;
                        pos = check + 1 + close_tag.size();
                        Token close;
                        close.type = (close_tag == "}}") ? TokenType::EXPR_CLOSE : TokenType::STMT_CLOSE;
                        close.trim_right = trim_r;
                        tokens.push_back(close);
                        // Strip leading whitespace from next text
                        skip_ws_text(pos);
                        return;
                    }
                }
                if (src_.substr(pos, close_tag.size()) == close_tag) {
                    pos += close_tag.size();
                    Token close;
                    close.type = (close_tag == "}}") ? TokenType::EXPR_CLOSE : TokenType::STMT_CLOSE;
                    tokens.push_back(close);
                    // trim_blocks (#1572): the newline ending a block-tag line is part of the tag, not
                    // output.
                    // Jinja defaults this off but transformers renders with trim_blocks=True. Plain "%}" case
                    // only; "-%}" already ate trailing whitespace. Statement tags only, {{ }} is unaffected.
                    if (close_tag == "%}") {
                        if (pos < src_.size() && src_[pos] == '\r')
                            pos++;
                        if (pos < src_.size() && src_[pos] == '\n')
                            pos++;
                    }
                    return;
                }
            }

            if (src_[pos] == '{') {
                tokens.push_back({TokenType::LBRACE, "{"});
                ++brace_depth;
                skip_ws(++pos);
                continue;
            }

            // String literal
            if (src_[pos] == '\'' || src_[pos] == '"') {
                std::string s = lex_string_(pos);
                tokens.push_back({TokenType::STRING, std::move(s)});
                skip_ws(pos);
                continue;
            }

            // Number
            if (std::isdigit(static_cast<unsigned char>(src_[pos])) ||
                (src_[pos] == '-' && pos + 1 < src_.size() &&
                 std::isdigit(static_cast<unsigned char>(src_[pos + 1])) &&
                 // Only treat as negative number if not after an ident/number/string/rparen/rbracket
                 (tokens.empty() || tokens.back().type == TokenType::STMT_OPEN ||
                  tokens.back().type == TokenType::EXPR_OPEN || tokens.back().type == TokenType::COMMA ||
                  tokens.back().type == TokenType::LPAREN || tokens.back().type == TokenType::LBRACKET ||
                  tokens.back().type == TokenType::COLON || tokens.back().type == TokenType::ASSIGN ||
                  (tokens.back().type == TokenType::OP) || (tokens.back().type == TokenType::PIPE)))) {
                size_t start = pos;
                if (src_[pos] == '-')
                    pos++;
                while (pos < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos])))
                    pos++;
                if (pos < src_.size() && src_[pos] == '.') {
                    pos++;
                    while (pos < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos])))
                        pos++;
                }
                pos = scan_exponent(pos);
                tokens.push_back({TokenType::NUMBER, src_.substr(start, pos - start)});
                skip_ws(pos);
                continue;
            }

            // Identifier or keyword
            if (std::isalpha(static_cast<unsigned char>(src_[pos])) || src_[pos] == '_') {
                size_t start = pos;
                while (pos < src_.size() &&
                       (std::isalnum(static_cast<unsigned char>(src_[pos])) || src_[pos] == '_'))
                    pos++;
                std::string word = src_.substr(start, pos - start);

                // Operators that look like identifiers
                if (word == "not" || word == "and" || word == "or" || word == "in" || word == "is") {
                    // "not in" is a single operator
                    if (word == "not") {
                        size_t saved = pos;
                        skip_ws(pos);
                        if (pos + 2 <= src_.size() && src_.substr(pos, 2) == "in" &&
                            (pos + 2 >= src_.size() ||
                             !std::isalnum(static_cast<unsigned char>(src_[pos + 2])))) {
                            pos += 2;
                            tokens.push_back({TokenType::OP, "not in"});
                            skip_ws(pos);
                            continue;
                        }
                        pos = saved;
                    }
                    // "is not" is a single operator
                    if (word == "is") {
                        size_t saved = pos;
                        skip_ws(pos);
                        if (pos + 3 <= src_.size() && src_.substr(pos, 3) == "not" &&
                            (pos + 3 >= src_.size() ||
                             !std::isalnum(static_cast<unsigned char>(src_[pos + 3])))) {
                            pos += 3;
                            tokens.push_back({TokenType::OP, "is not"});
                            skip_ws(pos);
                            continue;
                        }
                        pos = saved;
                    }
                    tokens.push_back({TokenType::OP, std::move(word)});
                } else {
                    tokens.push_back({TokenType::IDENT, std::move(word)});
                }
                skip_ws(pos);
                continue;
            }

            // Operators and punctuation
            char c = src_[pos];
            switch (c) {
                case ',':
                    tokens.push_back({TokenType::COMMA, ","});
                    pos++;
                    break;
                case '.':
                    tokens.push_back({TokenType::DOT, "."});
                    pos++;
                    break;
                case '[':
                    tokens.push_back({TokenType::LBRACKET, "["});
                    pos++;
                    break;
                case ']':
                    tokens.push_back({TokenType::RBRACKET, "]"});
                    pos++;
                    break;
                case '(':
                    tokens.push_back({TokenType::LPAREN, "("});
                    pos++;
                    break;
                case ')':
                    tokens.push_back({TokenType::RPAREN, ")"});
                    pos++;
                    break;
                case '|':
                    tokens.push_back({TokenType::PIPE, "|"});
                    pos++;
                    break;
                case ':':
                    tokens.push_back({TokenType::COLON, ":"});
                    pos++;
                    break;
                case '~':
                    tokens.push_back({TokenType::OP, "~"});
                    pos++;
                    break;
                case '+':
                    tokens.push_back({TokenType::OP, "+"});
                    pos++;
                    break;
                case '-':
                    tokens.push_back({TokenType::OP, "-"});
                    pos++;
                    break;
                case '*':
                case '/': {
                    // `**` power and `//` floor division before the single-char operators.
                    const bool twice = pos + 1 < src_.size() && src_[pos + 1] == src_[pos];
                    tokens.push_back({TokenType::OP, std::string(twice ? 2 : 1, src_[pos])});
                    pos += twice ? 2 : 1;
                    break;
                }
                case '%':
                    // Could be close tag %}
                    if (pos + 1 < src_.size() && src_[pos + 1] == '}') {
                        // Handled at top of loop
                    }
                    tokens.push_back({TokenType::OP, "%"});
                    pos++;
                    break;
                case '=':
                    if (pos + 1 < src_.size() && src_[pos + 1] == '=') {
                        tokens.push_back({TokenType::OP, "=="});
                        pos += 2;
                    } else {
                        tokens.push_back({TokenType::ASSIGN, "="});
                        pos++;
                    }
                    break;
                case '!':
                    if (pos + 1 < src_.size() && src_[pos + 1] == '=') {
                        tokens.push_back({TokenType::OP, "!="});
                        pos += 2;
                    } else {
                        tokens.push_back({TokenType::OP, "!"});
                        pos++;
                    }
                    break;
                case '<':
                    if (pos + 1 < src_.size() && src_[pos + 1] == '=') {
                        tokens.push_back({TokenType::OP, "<="});
                        pos += 2;
                    } else {
                        tokens.push_back({TokenType::OP, "<"});
                        pos++;
                    }
                    break;
                case '>':
                    if (pos + 1 < src_.size() && src_[pos + 1] == '=') {
                        tokens.push_back({TokenType::OP, ">="});
                        pos += 2;
                    } else {
                        tokens.push_back({TokenType::OP, ">"});
                        pos++;
                    }
                    break;
                default:
                    pos++;  // skip unknown char
                    break;
            }
            skip_ws(pos);
        }
    }

    void skip_ws(size_t& pos) const {
        while (pos < src_.size() &&
               (src_[pos] == ' ' || src_[pos] == '\t' || src_[pos] == '\n' || src_[pos] == '\r'))
            pos++;
    }

    void skip_ws_text(size_t& pos) const {
        // After trim-right close tag: strip whitespace including one newline
        while (pos < src_.size() &&
               (src_[pos] == ' ' || src_[pos] == '\t' || src_[pos] == '\n' || src_[pos] == '\r'))
            pos++;
    }

    // Optional exponent after a number literal: 1e3, 2.5E-4.
    size_t scan_exponent(size_t pos) const {
        if (pos >= src_.size() || (src_[pos] != 'e' && src_[pos] != 'E'))
            return pos;
        size_t p = pos + 1;
        if (p < src_.size() && (src_[p] == '+' || src_[p] == '-'))
            p++;
        if (p >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[p])))
            return pos;
        while (p < src_.size() && std::isdigit(static_cast<unsigned char>(src_[p])))
            p++;
        return p;
    }

    static void rtrim(std::string& s) {
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' || s.back() == '\r'))
            s.pop_back();
    }

    const std::string& src_;
};
}  // namespace

std::vector<Token> tokenize(const std::string& source) { return Lexer(source).tokenize(); }

}  // namespace imp::jinja::detail
