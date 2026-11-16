// Jinja2 lexer and parser: template source -> AST (jinja_ast.h).

#include "model/jinja_ast.h"
#include "core/logging.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace imp::jinja::detail {

namespace {


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
                    tokens.push_back({TokenType::OP, "*"});
                    pos++;
                    break;
                case '/':
                    tokens.push_back({TokenType::OP, "/"});
                    pos++;
                    break;
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

    static void rtrim(std::string& s) {
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' || s.back() == '\r'))
            s.pop_back();
    }

    const std::string& src_;
};

// ============================================================================
// Parser
// ============================================================================

class Parser {
public:
    Parser(const std::vector<Token>& tokens) : tokens_(tokens), pos_(0) {}

    bool parse(std::vector<std::unique_ptr<Node>>& out) {
        while (!at_end()) {
            auto node = parse_node();
            if (too_deep_ || !unknown_tag_.empty())
                break;
            if (!node) {
                // Skip problematic token and continue
                if (!at_end())
                    pos_++;
                continue;
            }
            out.push_back(std::move(node));
        }
        if (too_deep_ || !unknown_tag_.empty()) {
            // Half-built nodes can hold null sub-expressions, and the
            // evaluator dereferences those without checking. The caller
            // discards a failed parse, so leave nothing behind for it.
            out.clear();
            return false;
        }
        return true;
    }

    // Empty unless parse() stopped on a tag this engine does not implement.
    const std::string& unknown_tag() const { return unknown_tag_; }

private:
    // Template nesting comes from the checkpoint (chat_template in tokenizer_config.json), not
    // the operator: parse_primary/parse_node recurse with no bound, so 100000 '(' overflows the
    // stack at load. 256 is far past any real template (deepest here: 6).
    static constexpr int kMaxParseDepth = 256;
    int depth_ = 0;
    bool too_deep_ = false;
    std::string unknown_tag_;

    struct DepthGuard {
        Parser& p;
        bool ok;
        explicit DepthGuard(Parser& parser) : p(parser), ok(parser.depth_ < kMaxParseDepth) {
            if (ok)
                ++p.depth_;
            else
                p.too_deep_ = true;
        }
        ~DepthGuard() {
            if (ok)
                --p.depth_;
        }
    };

    const Token& peek() const { return pos_ < tokens_.size() ? tokens_[pos_] : tokens_.back(); }

    const Token& advance() {
        auto& t = tokens_[pos_];
        if (pos_ < tokens_.size() - 1)
            pos_++;
        return t;
    }

    bool at_end() const { return pos_ >= tokens_.size() || tokens_[pos_].type == TokenType::END; }

    bool check(TokenType t) const { return peek().type == t; }
    bool check(TokenType t, const std::string& v) const { return peek().type == t && peek().value == v; }

    bool match(TokenType t) {
        if (check(t)) {
            advance();
            return true;
        }
        return false;
    }

    bool match(TokenType t, const std::string& v) {
        if (check(t, v)) {
            advance();
            return true;
        }
        return false;
    }

    void expect(TokenType t) {
        if (!check(t)) {
            IMP_LOG_WARN("jinja: expected token type %d, got %d ('%s')", std::to_underlying(t),
                         std::to_underlying(peek().type), peek().value.c_str());
        }
        advance();
    }

    std::unique_ptr<Node> parse_node() {
        DepthGuard guard(*this);
        if (!guard.ok)
            return nullptr;
        if (check(TokenType::TEXT)) {
            auto text = peek().value;
            advance();
            return std::make_unique<TextNode>(std::move(text));
        }
        if (check(TokenType::EXPR_OPEN)) {
            advance();  // {{
            auto expr = parse_expr();
            if (check(TokenType::EXPR_CLOSE))
                advance();
            return std::make_unique<ExprNode>(std::move(expr));
        }
        if (check(TokenType::STMT_OPEN)) {
            return parse_statement();
        }
        return nullptr;
    }

    std::unique_ptr<Node> parse_statement() {
        advance();  // {%

        if (check(TokenType::IDENT, "for"))
            return parse_for();
        if (check(TokenType::IDENT, "if"))
            return parse_if();
        if (check(TokenType::IDENT, "set"))
            return parse_set();
        if (check(TokenType::IDENT, "macro"))
            return parse_macro();
        if (check(TokenType::IDENT, "break") || check(TokenType::IDENT, "continue")) {
            const bool brk = peek().value == "break";
            advance();
            if (check(TokenType::STMT_CLOSE))
                advance();
            return std::make_unique<LoopCtlNode>(brk);
        }
        // {% generation %}/{% endgeneration %} mark the assistant span for training masks; they
        // contribute nothing to rendered text and HF's renderer ignores them too. No-op, not refusal.
        if (check(TokenType::IDENT, "generation") || check(TokenType::IDENT, "endgeneration")) {
            while (!at_end() && !check(TokenType::STMT_CLOSE))
                advance();
            if (check(TokenType::STMT_CLOSE))
                advance();
            // An empty node, not nullptr: the caller reads nullptr as "could
            // not parse this token" and skips one MORE token, which ate the
            // text right after the tag.
            return std::make_unique<TextNode>(std::string());
        }
        // Any other tag is unimplemented or an unbalanced end tag. Silently skipping it produced a
        // wrong prompt with no error and made ChatTemplate's hardcoded-fallback path unreachable
        // (#1565): parse() could not fail.
        unknown_tag_ = peek().value;
        if (unknown_tag_.empty())
            unknown_tag_ = "<empty>";
        return nullptr;
    }

    std::unique_ptr<Node> parse_for() {
        advance();  // 'for'
        auto node = std::make_unique<ForNode>();

        node->var_name = peek().value;
        advance();  // var name

        // Check for "key, value" unpacking
        if (check(TokenType::COMMA)) {
            advance();
            node->var_name2 = peek().value;
            advance();
        }

        // 'in'
        if (check(TokenType::OP, "in"))
            advance();

        node->iterable = parse_expr();

        // Optional 'recursive'
        if (check(TokenType::IDENT, "recursive")) {
            node->recursive = true;
            advance();
        }

        // Close %}
        if (check(TokenType::STMT_CLOSE))
            advance();

        // Parse body until endfor or else
        parse_body(node->body, {"endfor", "else"});

        // Check for else
        if (check(TokenType::STMT_OPEN)) {
            // Peek ahead to see if it's {% else %} or {% endfor %}
            size_t saved = pos_;
            advance();  // {%
            if (check(TokenType::IDENT, "else")) {
                advance();  // else
                if (check(TokenType::STMT_CLOSE))
                    advance();
                parse_body(node->else_body, {"endfor"});
                // Consume endfor
                if (check(TokenType::STMT_OPEN)) {
                    advance();
                    if (check(TokenType::IDENT, "endfor"))
                        advance();
                    if (check(TokenType::STMT_CLOSE))
                        advance();
                }
            } else if (check(TokenType::IDENT, "endfor")) {
                advance();
                if (check(TokenType::STMT_CLOSE))
                    advance();
            } else {
                pos_ = saved;  // restore
            }
        }

        return node;
    }

    std::unique_ptr<Node> parse_macro() {
        advance();  // 'macro'
        auto node = std::make_unique<MacroNode>();

        // macro name
        node->name = peek().value;
        advance();

        // (param1, param2=default, ...)
        if (check(TokenType::LPAREN)) {
            advance();  // '('
            while (!check(TokenType::RPAREN) && !at_end()) {
                MacroNode::Param param;
                param.name = peek().value;
                advance();
                // Optional default: =expr. The lexer emits ASSIGN for a bare '=', OP only for "==" (#1566);
                // testing OP "=" matched nothing, so '=' and the default expr were consumed as two extra
                // positional params and the named one bound to none.
                if (check(TokenType::ASSIGN)) {
                    advance();
                    param.default_value = parse_expr();
                }
                node->params.push_back(std::move(param));
                if (check(TokenType::COMMA))
                    advance();
            }
            if (check(TokenType::RPAREN))
                advance();  // ')'
        }

        if (check(TokenType::STMT_CLOSE))
            advance();

        // Parse body until endmacro
        parse_body(node->body, {"endmacro"});

        // Consume {% endmacro %}
        if (check(TokenType::STMT_OPEN)) {
            advance();
            if (check(TokenType::IDENT, "endmacro"))
                advance();
            if (check(TokenType::STMT_CLOSE))
                advance();
        }

        return node;
    }

    std::unique_ptr<Node> parse_if() {
        advance();  // 'if'
        auto node = std::make_unique<IfNode>();

        // First branch
        IfNode::Branch branch;
        branch.condition = parse_expr();
        if (check(TokenType::STMT_CLOSE))
            advance();
        parse_body(branch.body, {"endif", "elif", "else"});
        node->branches.push_back(std::move(branch));

        // elif / else branches
        while (check(TokenType::STMT_OPEN)) {
            size_t saved = pos_;
            advance();  // {%

            if (check(TokenType::IDENT, "elif")) {
                advance();
                IfNode::Branch b;
                b.condition = parse_expr();
                if (check(TokenType::STMT_CLOSE))
                    advance();
                parse_body(b.body, {"endif", "elif", "else"});
                node->branches.push_back(std::move(b));
            } else if (check(TokenType::IDENT, "else")) {
                advance();
                if (check(TokenType::STMT_CLOSE))
                    advance();
                IfNode::Branch b;
                b.condition = nullptr;  // else
                parse_body(b.body, {"endif"});
                node->branches.push_back(std::move(b));
                // Consume endif
                if (check(TokenType::STMT_OPEN)) {
                    advance();
                    if (check(TokenType::IDENT, "endif"))
                        advance();
                    if (check(TokenType::STMT_CLOSE))
                        advance();
                }
                break;
            } else if (check(TokenType::IDENT, "endif")) {
                advance();
                if (check(TokenType::STMT_CLOSE))
                    advance();
                break;
            } else {
                pos_ = saved;
                break;
            }
        }

        return node;
    }

    std::unique_ptr<Node> parse_set() {
        advance();  // 'set'
        auto node = std::make_unique<SetNode>();

        node->var_name = peek().value;
        advance();

        // Check for ns.attr = ...
        if (check(TokenType::DOT)) {
            advance();
            node->attr_name = peek().value;
            advance();
        }

        // Block form {% set x %}...{% endset %} captures the rendered body (Gemma-4's template
        // uses this for captured_content). Without this the '=' branch ran parse_expr() on '%}',
        // the body rendered inline, and the variable stayed unset (#1565).
        if (!check(TokenType::ASSIGN)) {
            node->is_block = true;
            if (check(TokenType::STMT_CLOSE))
                advance();
            parse_body(node->body, {"endset"});
            if (check(TokenType::STMT_OPEN)) {
                advance();
                if (check(TokenType::IDENT, "endset"))
                    advance();
                if (check(TokenType::STMT_CLOSE))
                    advance();
            }
            return node;
        }

        advance();  // '='

        node->value = parse_expr();
        if (check(TokenType::STMT_CLOSE))
            advance();

        return node;
    }

    void parse_body(std::vector<std::unique_ptr<Node>>& body, const std::vector<std::string>& end_keywords) {
        while (!at_end()) {
            // A tag the engine cannot parse ends the whole parse; without this
            // the body loop keeps advancing to the end of the template first.
            if (!unknown_tag_.empty() || too_deep_)
                return;
            // Check if next is a closing statement keyword
            if (check(TokenType::STMT_OPEN)) {
                size_t saved = pos_;
                advance();
                bool is_end = false;
                for (auto& kw : end_keywords) {
                    if (check(TokenType::IDENT, kw)) {
                        is_end = true;
                        break;
                    }
                }
                pos_ = saved;
                if (is_end)
                    return;
            }

            auto node = parse_node();
            if (node)
                body.push_back(std::move(node));
            else if (!at_end())
                pos_++;  // skip to avoid infinite loop
        }
    }

    // ---- Expression parsing with precedence climbing ----

    std::unique_ptr<Expr> parse_expr() {
        DepthGuard guard(*this);
        if (!guard.ok)
            return nullptr;
        auto expr = parse_ternary();
        return expr;
    }

    std::unique_ptr<Expr> parse_ternary() {
        auto expr = parse_or();
        // Check for: expr if cond else other
        // But also just 'value if cond' without else
        if (check(TokenType::IDENT, "if")) {
            advance();
            auto cond = parse_or();
            std::unique_ptr<Expr> false_val;
            if (check(TokenType::IDENT, "else")) {
                advance();
                false_val = parse_or();
            } else {
                false_val = std::make_unique<LiteralExpr>(Value(""));
            }
            auto ternary = std::make_unique<TernaryExpr>();
            ternary->true_val = std::move(expr);
            ternary->condition = std::move(cond);
            ternary->false_val = std::move(false_val);
            return ternary;
        }
        return expr;
    }

    std::unique_ptr<Expr> parse_or() {
        auto left = parse_and();
        while (check(TokenType::OP, "or")) {
            advance();
            auto right = parse_and();
            auto bin = std::make_unique<BinOpExpr>();
            bin->op = "or";
            bin->left = std::move(left);
            bin->right = std::move(right);
            left = std::move(bin);
        }
        return left;
    }

    std::unique_ptr<Expr> parse_and() {
        auto left = parse_not();
        while (check(TokenType::OP, "and")) {
            advance();
            auto right = parse_not();
            auto bin = std::make_unique<BinOpExpr>();
            bin->op = "and";
            bin->left = std::move(left);
            bin->right = std::move(right);
            left = std::move(bin);
        }
        return left;
    }

    std::unique_ptr<Expr> parse_not() {
        if (check(TokenType::OP, "not")) {
            advance();
            auto operand = parse_not();
            auto u = std::make_unique<UnaryExpr>();
            u->op = "not";
            u->operand = std::move(operand);
            return u;
        }
        return parse_comparison();
    }

    std::unique_ptr<Expr> parse_comparison() {
        auto left = parse_addition();
        while (true) {
            std::string op;
            if (check(TokenType::OP, "==") || check(TokenType::OP, "!=") || check(TokenType::OP, "<") ||
                check(TokenType::OP, ">") || check(TokenType::OP, "<=") || check(TokenType::OP, ">=") ||
                check(TokenType::OP, "in") || check(TokenType::OP, "not in")) {
                op = peek().value;
                advance();
            } else if (check(TokenType::OP, "is")) {
                advance();
                // "is defined", "is none", "is string", "is iterable", "is mapping", "is number"
                if (check(TokenType::IDENT)) {
                    std::string test_name = peek().value;
                    if (test_name == "defined" || test_name == "undefined" || test_name == "none" ||
                        test_name == "string" || test_name == "iterable" || test_name == "mapping" ||
                        test_name == "number" || test_name == "integer" || test_name == "float" ||
                        test_name == "boolean" || test_name == "sequence") {
                        advance();
                        auto bin = std::make_unique<BinOpExpr>();
                        bin->op = "is " + test_name;
                        bin->left = std::move(left);
                        bin->right = nullptr;
                        left = std::move(bin);
                        continue;
                    }
                }
                // Generic "is X" — treat as equality test
                auto right = parse_addition();
                auto bin = std::make_unique<BinOpExpr>();
                bin->op = "==";
                bin->left = std::move(left);
                bin->right = std::move(right);
                left = std::move(bin);
                continue;
            } else if (check(TokenType::OP, "is not")) {
                advance();
                if (check(TokenType::IDENT)) {
                    std::string test_name = peek().value;
                    if (test_name == "defined" || test_name == "undefined" || test_name == "none" ||
                        test_name == "string" || test_name == "iterable" || test_name == "mapping" ||
                        test_name == "number" || test_name == "integer" || test_name == "float" ||
                        test_name == "boolean" || test_name == "sequence") {
                        advance();
                        auto bin = std::make_unique<BinOpExpr>();
                        bin->op = "is not " + test_name;
                        bin->left = std::move(left);
                        bin->right = nullptr;
                        left = std::move(bin);
                        continue;
                    }
                }
                auto right = parse_addition();
                auto bin = std::make_unique<BinOpExpr>();
                bin->op = "!=";
                bin->left = std::move(left);
                bin->right = std::move(right);
                left = std::move(bin);
                continue;
            } else {
                break;
            }
            auto right = parse_addition();
            auto bin = std::make_unique<BinOpExpr>();
            bin->op = std::move(op);
            bin->left = std::move(left);
            bin->right = std::move(right);
            left = std::move(bin);
        }
        return left;
    }

    std::unique_ptr<Expr> parse_addition() {
        auto left = parse_multiplication();
        while (check(TokenType::OP, "+") || check(TokenType::OP, "-") || check(TokenType::OP, "~")) {
            auto op = peek().value;
            advance();
            auto right = parse_multiplication();
            auto bin = std::make_unique<BinOpExpr>();
            bin->op = std::move(op);
            bin->left = std::move(left);
            bin->right = std::move(right);
            left = std::move(bin);
        }
        return left;
    }

    std::unique_ptr<Expr> parse_multiplication() {
        auto left = parse_unary();
        while (check(TokenType::OP, "*") || check(TokenType::OP, "/") || check(TokenType::OP, "%")) {
            auto op = peek().value;
            advance();
            auto right = parse_unary();
            auto bin = std::make_unique<BinOpExpr>();
            bin->op = std::move(op);
            bin->left = std::move(left);
            bin->right = std::move(right);
            left = std::move(bin);
        }
        return left;
    }

    std::unique_ptr<Expr> parse_unary() {
        if (check(TokenType::OP, "-")) {
            advance();
            auto operand = parse_unary();
            auto u = std::make_unique<UnaryExpr>();
            u->op = "-";
            u->operand = std::move(operand);
            return u;
        }
        return parse_postfix();
    }

    std::unique_ptr<Expr> parse_postfix() {
        auto expr = parse_primary();
        while (true) {
            if (check(TokenType::DOT)) {
                advance();
                if (!check(TokenType::IDENT))
                    break;
                std::string attr = peek().value;
                advance();

                // Check for method call: obj.method(args)
                if (check(TokenType::LPAREN)) {
                    advance();
                    auto method = std::make_unique<MethodExpr>();
                    method->object = std::move(expr);
                    method->method = std::move(attr);
                    if (!check(TokenType::RPAREN)) {
                        method->args.push_back(parse_expr());
                        while (check(TokenType::COMMA)) {
                            advance();
                            method->args.push_back(parse_expr());
                        }
                    }
                    if (check(TokenType::RPAREN))
                        advance();
                    expr = std::move(method);
                } else {
                    auto ga = std::make_unique<GetAttrExpr>();
                    ga->object = std::move(expr);
                    ga->attr = std::move(attr);
                    expr = std::move(ga);
                }
            } else if (check(TokenType::LBRACKET)) {
                advance();
                // Check for slice notation: [start:stop:step]
                // Possible forms: [expr], [start:], [:stop], [start:stop], [::step], [start:stop:step], etc.
                std::unique_ptr<Expr> first;
                bool is_slice = false;

                // Check if starts with ':' (no start)
                if (check(TokenType::COLON)) {
                    is_slice = true;
                    first = nullptr;
                } else if (check(TokenType::RBRACKET)) {
                    // Empty brackets — shouldn't happen but handle gracefully
                    advance();
                    auto gi = std::make_unique<GetItemExpr>();
                    gi->object = std::move(expr);
                    gi->key = std::make_unique<LiteralExpr>(Value());
                    expr = std::move(gi);
                    continue;
                } else {
                    first = parse_expr();
                    if (check(TokenType::COLON)) {
                        is_slice = true;
                    }
                }

                if (is_slice) {
                    auto sl = std::make_unique<SliceExpr>();
                    sl->object = std::move(expr);
                    sl->start = std::move(first);  // may be nullptr

                    // Consume first ':'
                    if (check(TokenType::COLON))
                        advance();

                    // Parse stop (optional)
                    if (!check(TokenType::COLON) && !check(TokenType::RBRACKET)) {
                        sl->stop = parse_expr();
                    }

                    // Parse step (optional, after second ':')
                    if (check(TokenType::COLON)) {
                        advance();
                        if (!check(TokenType::RBRACKET)) {
                            sl->step = parse_expr();
                        }
                    }

                    if (check(TokenType::RBRACKET))
                        advance();
                    expr = std::move(sl);
                } else {
                    // Regular subscript
                    if (check(TokenType::RBRACKET))
                        advance();
                    auto gi = std::make_unique<GetItemExpr>();
                    gi->object = std::move(expr);
                    gi->key = std::move(first);
                    expr = std::move(gi);
                }
            } else if (check(TokenType::LPAREN)) {
                // Function call on an expression
                advance();
                auto call = std::make_unique<CallExpr>();
                call->callee = std::move(expr);
                if (!check(TokenType::RPAREN)) {
                    parse_call_args(call->args, call->kwargs);
                }
                if (check(TokenType::RPAREN))
                    advance();
                expr = std::move(call);
            } else if (check(TokenType::PIPE)) {
                advance();
                if (!check(TokenType::IDENT))
                    break;
                std::string filter_name = peek().value;
                advance();

                auto filter = std::make_unique<FilterExpr>();
                filter->value = std::move(expr);
                filter->name = std::move(filter_name);

                if (check(TokenType::LPAREN)) {
                    advance();
                    if (!check(TokenType::RPAREN)) {
                        filter->args.push_back(parse_expr());
                        while (check(TokenType::COMMA)) {
                            advance();
                            filter->args.push_back(parse_expr());
                        }
                    }
                    if (check(TokenType::RPAREN))
                        advance();
                }
                expr = std::move(filter);
            } else {
                break;
            }
        }
        return expr;
    }

    void parse_call_args(std::vector<std::unique_ptr<Expr>>& args,
                         std::vector<std::pair<std::string, std::unique_ptr<Expr>>>& kwargs) {
        while (true) {
            // Check for keyword argument: ident=expr
            if (check(TokenType::IDENT)) {
                size_t saved = pos_;
                std::string name = peek().value;
                advance();
                if (check(TokenType::ASSIGN)) {
                    advance();
                    auto val = parse_expr();
                    kwargs.emplace_back(std::move(name), std::move(val));
                    if (check(TokenType::COMMA)) {
                        advance();
                        continue;
                    }
                    break;
                }
                pos_ = saved;
            }
            args.push_back(parse_expr());
            if (check(TokenType::COMMA)) {
                advance();
                continue;
            }
            break;
        }
    }

    std::unique_ptr<Expr> parse_primary() {
        // String literal
        if (check(TokenType::STRING)) {
            auto val = peek().value;
            advance();
            return std::make_unique<LiteralExpr>(Value(std::move(val)));
        }

        // Number literal
        if (check(TokenType::NUMBER)) {
            auto val = peek().value;
            advance();
            if (val.contains('.')) {
                double d = 0;
                std::from_chars(val.data(), val.data() + val.size(), d);
                return std::make_unique<LiteralExpr>(Value(d));
            }
            int64_t i = 0;
            std::from_chars(val.data(), val.data() + val.size(), i);
            return std::make_unique<LiteralExpr>(Value(i));
        }

        // Boolean/none literals and identifiers
        if (check(TokenType::IDENT)) {
            auto name = peek().value;
            advance();

            if (name == "true" || name == "True")
                return std::make_unique<LiteralExpr>(Value(true));
            if (name == "false" || name == "False")
                return std::make_unique<LiteralExpr>(Value(false));
            if (name == "none" || name == "None")
                return std::make_unique<LiteralExpr>(Value());

            return std::make_unique<VariableExpr>(std::move(name));
        }

        // Parenthesized expression; a comma after the first element makes it a tuple, evaluated as an array
        if (check(TokenType::LPAREN)) {
            advance();
            auto expr = parse_expr();
            if (check(TokenType::COMMA)) {
                auto tuple = std::make_unique<ArrayExpr>();
                tuple->elements.push_back(std::move(expr));
                while (check(TokenType::COMMA)) {
                    advance();
                    if (check(TokenType::RPAREN))
                        break;  // trailing comma
                    tuple->elements.push_back(parse_expr());
                }
                expr = std::move(tuple);
            }
            if (check(TokenType::RPAREN))
                advance();
            return expr;
        }

        // Array literal
        if (check(TokenType::LBRACKET)) {
            advance();
            auto arr = std::make_unique<ArrayExpr>();
            if (!check(TokenType::RBRACKET)) {
                arr->elements.push_back(parse_expr());
                while (check(TokenType::COMMA)) {
                    advance();
                    if (check(TokenType::RBRACKET))
                        break;  // trailing comma
                    arr->elements.push_back(parse_expr());
                }
            }
            if (check(TokenType::RBRACKET))
                advance();
            return arr;
        }

        if (check(TokenType::LBRACE))
            return parse_dict_();

        // Fallback: empty
        return std::make_unique<LiteralExpr>(Value());
    }

    // {key: value, ...} (Cohere2 text_wrapper); trailing comma allowed.
    std::unique_ptr<Expr> parse_dict_() {
        advance();  // {
        auto dict = std::make_unique<DictExpr>();
        while (!at_end() && !check(TokenType::RBRACE)) {
            auto key = parse_expr();
            if (!check(TokenType::COLON))
                break;
            advance();
            dict->entries.emplace_back(std::move(key), parse_expr());
            if (!check(TokenType::COMMA))
                break;
            advance();
        }
        if (check(TokenType::RBRACE))
            advance();
        return dict;
    }

    const std::vector<Token>& tokens_;
    size_t pos_;
};

}  // namespace

bool parse_template(const std::string& source, std::vector<std::unique_ptr<Node>>& out, std::string& unknown_tag) {
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    if (!parser.parse(out)) {
        unknown_tag = parser.unknown_tag();
        return false;
    }
    return true;
}

}  // namespace imp::jinja::detail
