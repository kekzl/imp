// Jinja2 template engine for LLM chat templates (GGUF).
// Supports: variables, for/if/set, filters, operators, namespace(), whitespace control.
// Pure C++20, no regex, no external dependencies.

#include "model/jinja.h"
#include "core/logging.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <charconv>
#include <cmath>
#include <ctime>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace imp::jinja {

// ============================================================================
// Value implementation
// ============================================================================

bool Value::truthy() const {
    return std::visit(
        [](auto&& v) -> bool {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::monostate>)
                return false;
            else if constexpr (std::is_same_v<T, bool>)
                return v;
            else if constexpr (std::is_same_v<T, int64_t>)
                return v != 0;
            else if constexpr (std::is_same_v<T, double>)
                return v != 0.0;
            else if constexpr (std::is_same_v<T, std::string>)
                return !v.empty();
            else if constexpr (std::is_same_v<T, Array>)
                return !v.empty();
            else if constexpr (std::is_same_v<T, Object>)
                return v && !v->empty();
            else
                return false;
        },
        data_);
}

std::string Value::to_string() const {
    return std::visit(
        [](auto&& v) -> std::string {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::monostate>)
                return "None";
            else if constexpr (std::is_same_v<T, bool>)
                return v ? "True" : "False";
            else if constexpr (std::is_same_v<T, int64_t>)
                return std::to_string(v);
            else if constexpr (std::is_same_v<T, double>) {
                // Format like Python: no trailing zeros, but keep .0 for integers
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%g", v);
                return buf;
            } else if constexpr (std::is_same_v<T, std::string>)
                return v;
            else if constexpr (std::is_same_v<T, Array>) {
                std::string r = "[";
                for (size_t i = 0; i < v.size(); i++) {
                    if (i > 0)
                        r += ", ";
                    if (v[i].is_string()) {
                        r += "'";
                        r += v[i].as_string();
                        r += "'";
                    } else
                        r += v[i].to_string();
                }
                r += "]";
                return r;
            } else if constexpr (std::is_same_v<T, Object>) {
                if (!v)
                    return "{}";
                std::string r = "{";
                bool first = true;
                for (auto& [k, val] : *v) {
                    if (!first)
                        r += ", ";
                    r += "'";
                    r += k;
                    r += "': ";
                    if (val.is_string()) {
                        r += "'";
                        r += val.as_string();
                        r += "'";
                    } else
                        r += val.to_string();
                    first = false;
                }
                r += "}";
                return r;
            } else
                return "";
        },
        data_);
}

double Value::to_number() const {
    if (is_int())
        return static_cast<double>(as_int());
    if (is_double())
        return as_double();
    if (is_bool())
        return as_bool() ? 1.0 : 0.0;
    return 0.0;
}

bool Value::operator==(const Value& o) const {
    if (is_none() && o.is_none())
        return true;
    if (is_bool() && o.is_bool())
        return as_bool() == o.as_bool();
    if (is_number() && o.is_number())
        return to_number() == o.to_number();
    if (is_string() && o.is_string())
        return as_string() == o.as_string();
    if (is_array() && o.is_array()) {
        auto& a = as_array();
        auto& b = o.as_array();
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); i++)
            if (a[i] != b[i])
                return false;
        return true;
    }
    // none == false, 0 == false, etc.
    if (is_none() && o.is_bool())
        return !o.as_bool();
    if (is_bool() && o.is_none())
        return !as_bool();
    return false;
}

bool Value::operator<(const Value& o) const {
    if (is_number() && o.is_number())
        return to_number() < o.to_number();
    if (is_string() && o.is_string())
        return as_string() < o.as_string();
    return false;
}

Value Value::operator+(const Value& o) const {
    if (is_int() && o.is_int())
        return Value(as_int() + o.as_int());
    if (is_number() && o.is_number())
        return Value(to_number() + o.to_number());
    if (is_string() && o.is_string())
        return Value(as_string() + o.as_string());
    if (is_string())
        return Value(as_string() + o.to_string());
    if (o.is_string())
        return Value(to_string() + o.as_string());
    if (is_array() && o.is_array()) {
        Array r = as_array();
        auto& b = o.as_array();
        r.insert(r.end(), b.begin(), b.end());
        return Value(std::move(r));
    }
    return Value(to_number() + o.to_number());
}

int64_t Value::length() const {
    if (is_string())
        return static_cast<int64_t>(as_string().size());
    if (is_array())
        return static_cast<int64_t>(as_array().size());
    if (is_object() && as_object())
        return static_cast<int64_t>(as_object()->size());
    return 0;
}

Value Value::get(const Value& key) const {
    if (is_array()) {
        if (key.is_int()) {
            auto& arr = as_array();
            int64_t idx = key.as_int();
            if (idx < 0)
                idx += static_cast<int64_t>(arr.size());
            if (idx >= 0 && idx < static_cast<int64_t>(arr.size()))
                return arr[static_cast<size_t>(idx)];
        }
        if (key.is_string()) {
            // 'length' attribute on arrays
            if (key.as_string() == "length")
                return Value(static_cast<int64_t>(as_array().size()));
        }
        return Value();
    }
    if (is_object() && as_object()) {
        if (key.is_string()) {
            auto it = as_object()->find(key.as_string());
            if (it != as_object()->end())
                return it->second;
        }
        return Value();
    }
    if (is_string() && key.is_int()) {
        auto& s = as_string();
        int64_t idx = key.as_int();
        if (idx < 0)
            idx += static_cast<int64_t>(s.size());
        if (idx >= 0 && idx < static_cast<int64_t>(s.size()))
            return Value(std::string(1, s[static_cast<size_t>(idx)]));
    }
    return Value();
}

Value Value::get(const std::string& name) const { return get(Value(name)); }

void Value::set(const std::string& name, Value val) {
    if (is_object() && as_object()) {
        (*std::get<Object>(data_))[name] = std::move(val);
    }
}

bool Value::contains(const Value& item) const {
    if (is_array()) {
        for (auto& v : as_array())
            if (v == item)
                return true;
        return false;
    }
    if (is_string() && item.is_string()) {
        return as_string().find(item.as_string()) != std::string::npos;
    }
    if (is_object() && as_object() && item.is_string()) {
        return as_object()->count(item.as_string()) > 0;
    }
    return false;
}

Value Value::make_object() { return Value(std::make_shared<std::map<std::string, Value>>()); }

// ============================================================================
// Lexer
// ============================================================================

namespace detail {

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
                // Skip leading whitespace after comment
                if (trim_r)
                    while (pos < src_.size() &&
                           (src_[pos] == ' ' || src_[pos] == '\t' || src_[pos] == '\n' || src_[pos] == '\r'))
                        pos++;
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
                tokens.push_back(t);
            }
        }
        tokens.push_back({TokenType::END, ""});
        return tokens;
    }

private:
    void tokenize_inner(std::vector<Token>& tokens, size_t& pos, const std::string& close_tag) {
        skip_ws(pos);
        while (pos < src_.size()) {
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
                    return;
                }
            }

            // String literal
            if (src_[pos] == '\'' || src_[pos] == '"') {
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
// AST nodes
// ============================================================================

struct Expr {
    virtual ~Expr() = default;
};

struct LiteralExpr : Expr {
    Value value;
    explicit LiteralExpr(Value v) : value(std::move(v)) {}
};

struct VariableExpr : Expr {
    std::string name;
    explicit VariableExpr(std::string n) : name(std::move(n)) {}
};

struct BinOpExpr : Expr {
    std::string op;
    std::unique_ptr<Expr> left, right;
};

struct UnaryExpr : Expr {
    std::string op;
    std::unique_ptr<Expr> operand;
};

struct GetAttrExpr : Expr {
    std::unique_ptr<Expr> object;
    std::string attr;
};

struct GetItemExpr : Expr {
    std::unique_ptr<Expr> object;
    std::unique_ptr<Expr> key;
};

struct SliceExpr : Expr {
    std::unique_ptr<Expr> object;
    std::unique_ptr<Expr> start;  // nullptr = beginning
    std::unique_ptr<Expr> stop;   // nullptr = end
    std::unique_ptr<Expr> step;   // nullptr = 1
};

struct FilterExpr : Expr {
    std::unique_ptr<Expr> value;
    std::string name;
    std::vector<std::unique_ptr<Expr>> args;
};

struct CallExpr : Expr {
    std::unique_ptr<Expr> callee;
    std::vector<std::unique_ptr<Expr>> args;
    // Named args: name=value pairs
    std::vector<std::pair<std::string, std::unique_ptr<Expr>>> kwargs;
};

struct MethodExpr : Expr {
    std::unique_ptr<Expr> object;
    std::string method;
    std::vector<std::unique_ptr<Expr>> args;
};

struct TernaryExpr : Expr {
    std::unique_ptr<Expr> true_val;
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Expr> false_val;
};

struct ArrayExpr : Expr {
    std::vector<std::unique_ptr<Expr>> elements;
};

struct DictExpr : Expr {
    std::vector<std::pair<std::unique_ptr<Expr>, std::unique_ptr<Expr>>> entries;
};

// Statement/template nodes
struct Node {
    virtual ~Node() = default;
};

struct TextNode : Node {
    std::string text;
    explicit TextNode(std::string t) : text(std::move(t)) {}
};

struct ExprNode : Node {
    std::unique_ptr<Expr> expr;
    explicit ExprNode(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
};

struct ForNode : Node {
    std::string var_name;
    std::string var_name2;  // for "key, value in dict"
    std::unique_ptr<Expr> iterable;
    std::vector<std::unique_ptr<Node>> body;
    std::vector<std::unique_ptr<Node>> else_body;
    bool recursive = false;
};

struct IfNode : Node {
    // Chain of (condition, body) pairs. Last may have null condition (else).
    struct Branch {
        std::unique_ptr<Expr> condition;  // nullptr for else
        std::vector<std::unique_ptr<Node>> body;
    };
    std::vector<Branch> branches;
};

struct SetNode : Node {
    std::string var_name;   // "x" or "ns" for ns.x
    std::string attr_name;  // "x" when doing ns.x = ...
    std::unique_ptr<Expr> value;
};

struct MacroNode : Node {
    std::string name;
    struct Param {
        std::string name;
        std::unique_ptr<Expr> default_value;  // nullptr if no default
    };
    std::vector<Param> params;
    std::vector<std::unique_ptr<Node>> body;
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
            if (!node) {
                // Skip problematic token and continue
                if (!at_end())
                    pos_++;
                continue;
            }
            out.push_back(std::move(node));
        }
        return true;
    }

private:
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
        // Unknown statement — skip to %}
        while (!at_end() && !check(TokenType::STMT_CLOSE))
            advance();
        if (check(TokenType::STMT_CLOSE))
            advance();
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
                // Optional default: =expr
                if (check(TokenType::OP, "=")) {
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

        if (check(TokenType::ASSIGN))
            advance();

        node->value = parse_expr();
        if (check(TokenType::STMT_CLOSE))
            advance();

        return node;
    }

    void parse_body(std::vector<std::unique_ptr<Node>>& body, const std::vector<std::string>& end_keywords) {
        while (!at_end()) {
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
                    if (test_name == "defined" || test_name == "none" || test_name == "string" ||
                        test_name == "iterable" || test_name == "mapping" || test_name == "number" ||
                        test_name == "integer" || test_name == "float" || test_name == "boolean" ||
                        test_name == "sequence") {
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
                    if (test_name == "defined" || test_name == "none" || test_name == "string" ||
                        test_name == "iterable" || test_name == "mapping" || test_name == "number" ||
                        test_name == "integer" || test_name == "float" || test_name == "boolean" ||
                        test_name == "sequence") {
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
                    kwargs.push_back({std::move(name), std::move(val)});
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
            if (val.find('.') != std::string::npos) {
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

        // Parenthesized expression or tuple
        if (check(TokenType::LPAREN)) {
            advance();
            auto expr = parse_expr();
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

        // Dict literal
        if (check(TokenType::IDENT) || check(TokenType::STRING)) {
            // Could be dict, but we'd need { which isn't in our token stream.
            // Dicts in Jinja use {} which conflicts with tags — only reachable in expressions.
        }

        // Fallback: empty
        return std::make_unique<LiteralExpr>(Value());
    }

    const std::vector<Token>& tokens_;
    size_t pos_;
};

// ============================================================================
// Evaluator
// ============================================================================

class Evaluator {
public:
    explicit Evaluator(const Context& ctx) {
        // Push root scope
        scopes_.emplace_back();
        for (auto& [k, v] : ctx) {
            scopes_.back()[k] = v;
        }
    }

    std::string render(const std::vector<std::unique_ptr<Node>>& nodes) {
        std::string result;
        for (auto& node : nodes) {
            render_node(*node, result);
        }
        return result;
    }

private:
    using Scope = std::map<std::string, Value>;

    void push_scope() { scopes_.emplace_back(); }
    void pop_scope() {
        if (scopes_.size() > 1)
            scopes_.pop_back();
    }

    Value lookup(const std::string& name) const {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end())
                return found->second;
        }
        return Value();  // undefined = none
    }

    bool is_defined(const std::string& name) const {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            if (it->count(name))
                return true;
        }
        return false;
    }

    void set_var(const std::string& name, Value val) { scopes_.back()[name] = std::move(val); }

    // Set in nearest scope where name exists, or current scope
    void set_var_update(const std::string& name, Value val) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) {
                found->second = std::move(val);
                return;
            }
        }
        scopes_.back()[name] = std::move(val);
    }

    void render_node(const Node& node, std::string& out) {
        if (auto* text = dynamic_cast<const TextNode*>(&node)) {
            out += text->text;
        } else if (auto* expr = dynamic_cast<const ExprNode*>(&node)) {
            Value val = eval(*expr->expr);
            if (!val.is_none())
                out += val.to_string();
        } else if (auto* for_node = dynamic_cast<const ForNode*>(&node)) {
            render_for(*for_node, out);
        } else if (auto* if_node = dynamic_cast<const IfNode*>(&node)) {
            render_if(*if_node, out);
        } else if (auto* set_node = dynamic_cast<const SetNode*>(&node)) {
            render_set(*set_node);
        } else if (auto* macro_node = dynamic_cast<const MacroNode*>(&node)) {
            register_macro(*macro_node);
        }
    }

    void render_for(const ForNode& node, std::string& out) {
        Value iterable = eval(*node.iterable);

        if (iterable.is_object() && iterable.as_object()) {
            // Iterate over dict: for key, value in dict.items()
            // Or for key in dict (iterate keys)
            auto& obj = *iterable.as_object();
            if (obj.empty() && !node.else_body.empty()) {
                push_scope();
                for (auto& n : node.else_body)
                    render_node(*n, out);
                pop_scope();
                return;
            }

            int64_t idx = 0;
            int64_t len = static_cast<int64_t>(obj.size());
            push_scope();
            for (auto& [key, val] : obj) {
                // Loop variables
                auto loop = Value::make_object();
                loop.set("index", Value(idx + 1));
                loop.set("index0", Value(idx));
                loop.set("first", Value(idx == 0));
                loop.set("last", Value(idx == len - 1));
                loop.set("length", Value(len));
                set_var("loop", loop);

                if (!node.var_name2.empty()) {
                    set_var(node.var_name, Value(key));
                    set_var(node.var_name2, val);
                } else {
                    set_var(node.var_name, Value(key));
                }

                for (auto& n : node.body)
                    render_node(*n, out);
                idx++;
            }
            pop_scope();
            return;
        }

        if (!iterable.is_array()) {
            if (!node.else_body.empty()) {
                push_scope();
                for (auto& n : node.else_body)
                    render_node(*n, out);
                pop_scope();
            }
            return;
        }

        auto& arr = iterable.as_array();
        if (arr.empty() && !node.else_body.empty()) {
            push_scope();
            for (auto& n : node.else_body)
                render_node(*n, out);
            pop_scope();
            return;
        }

        int64_t len = static_cast<int64_t>(arr.size());
        push_scope();
        for (int64_t i = 0; i < len; i++) {
            auto loop = Value::make_object();
            loop.set("index", Value(i + 1));
            loop.set("index0", Value(i));
            loop.set("first", Value(i == 0));
            loop.set("last", Value(i == len - 1));
            loop.set("length", Value(len));
            set_var("loop", loop);

            if (!node.var_name2.empty()) {
                // Tuple unpacking from array of arrays: for a, b in list_of_pairs
                auto& elem = arr[static_cast<size_t>(i)];
                if (elem.is_array() && elem.as_array().size() >= 2) {
                    set_var(node.var_name, elem.as_array()[0]);
                    set_var(node.var_name2, elem.as_array()[1]);
                } else {
                    set_var(node.var_name, elem);
                    set_var(node.var_name2, Value());
                }
            } else {
                set_var(node.var_name, arr[static_cast<size_t>(i)]);
            }

            for (auto& n : node.body)
                render_node(*n, out);
        }
        pop_scope();
    }

    void render_if(const IfNode& node, std::string& out) {
        for (auto& branch : node.branches) {
            if (!branch.condition) {
                // else branch
                for (auto& n : branch.body)
                    render_node(*n, out);
                return;
            }
            Value cond = eval(*branch.condition);
            if (cond.truthy()) {
                for (auto& n : branch.body)
                    render_node(*n, out);
                return;
            }
        }
    }

    void render_set(const SetNode& node) {
        Value val = eval(*node.value);
        if (!node.attr_name.empty()) {
            // namespace set: ns.attr = val
            Value ns = lookup(node.var_name);
            if (ns.is_object()) {
                ns.set(node.attr_name, std::move(val));
                // shared_ptr — mutation visible everywhere
            }
        } else {
            set_var_update(node.var_name, std::move(val));
        }
    }

    Value eval(const Expr& expr) {
        if (auto* lit = dynamic_cast<const LiteralExpr*>(&expr)) {
            return lit->value;
        }
        if (auto* var = dynamic_cast<const VariableExpr*>(&expr)) {
            return lookup(var->name);
        }
        if (auto* bin = dynamic_cast<const BinOpExpr*>(&expr)) {
            return eval_binop(*bin);
        }
        if (auto* unary = dynamic_cast<const UnaryExpr*>(&expr)) {
            return eval_unary(*unary);
        }
        if (auto* ga = dynamic_cast<const GetAttrExpr*>(&expr)) {
            Value obj = eval(*ga->object);
            return obj.get(ga->attr);
        }
        if (auto* gi = dynamic_cast<const GetItemExpr*>(&expr)) {
            Value obj = eval(*gi->object);
            Value key = eval(*gi->key);
            return obj.get(key);
        }
        if (auto* filter = dynamic_cast<const FilterExpr*>(&expr)) {
            return eval_filter(*filter);
        }
        if (auto* call = dynamic_cast<const CallExpr*>(&expr)) {
            return eval_call(*call);
        }
        if (auto* method = dynamic_cast<const MethodExpr*>(&expr)) {
            return eval_method(*method);
        }
        if (auto* ternary = dynamic_cast<const TernaryExpr*>(&expr)) {
            Value cond = eval(*ternary->condition);
            return cond.truthy() ? eval(*ternary->true_val) : eval(*ternary->false_val);
        }
        if (auto* arr = dynamic_cast<const ArrayExpr*>(&expr)) {
            Value::Array result;
            for (auto& e : arr->elements)
                result.push_back(eval(*e));
            return Value(std::move(result));
        }
        if (auto* dict = dynamic_cast<const DictExpr*>(&expr)) {
            auto obj = Value::make_object();
            for (auto& [k, v] : dict->entries) {
                Value key = eval(*k);
                Value val = eval(*v);
                obj.set(key.to_string(), std::move(val));
            }
            return obj;
        }
        if (auto* sl = dynamic_cast<const SliceExpr*>(&expr)) {
            return eval_slice(*sl);
        }
        return Value();
    }

    Value eval_binop(const BinOpExpr& bin) {
        // Short-circuit for and/or
        if (bin.op == "and") {
            Value left = eval(*bin.left);
            if (!left.truthy())
                return left;
            return eval(*bin.right);
        }
        if (bin.op == "or") {
            Value left = eval(*bin.left);
            if (left.truthy())
                return left;
            return eval(*bin.right);
        }

        // Special ops that don't need right eval
        if (bin.op == "is defined") {
            if (auto* var = dynamic_cast<const VariableExpr*>(bin.left.get()))
                return Value(is_defined(var->name));
            // For attr access, check if result is non-none
            Value left = eval(*bin.left);
            return Value(!left.is_none());
        }
        if (bin.op == "is not defined") {
            if (auto* var = dynamic_cast<const VariableExpr*>(bin.left.get()))
                return Value(!is_defined(var->name));
            Value left = eval(*bin.left);
            return Value(left.is_none());
        }
        if (bin.op == "is none") {
            Value left = eval(*bin.left);
            return Value(left.is_none());
        }
        if (bin.op == "is not none") {
            Value left = eval(*bin.left);
            return Value(!left.is_none());
        }

        // Type tests: is string, is iterable, is mapping, is number, etc.
        auto eval_type_test = [&](const std::string& test_name, const Value& v) -> bool {
            if (test_name == "string")
                return v.is_string();
            if (test_name == "iterable")
                return v.is_array() || v.is_string() || (v.is_object() && v.as_object());
            if (test_name == "mapping")
                return v.is_object() && v.as_object();
            if (test_name == "number")
                return v.is_number();
            if (test_name == "integer")
                return v.is_int();
            if (test_name == "float")
                return v.is_double();
            if (test_name == "boolean")
                return v.is_bool();
            if (test_name == "sequence")
                return v.is_array() || v.is_string();
            if (test_name == "defined")
                return !v.is_none();
            if (test_name == "none")
                return v.is_none();
            return false;
        };

        if (bin.op.size() > 3 && bin.op.substr(0, 3) == "is " && bin.op.substr(3, 4) != "not ") {
            std::string test_name = bin.op.substr(3);
            // For "is defined", check variable existence rather than value
            if (test_name == "defined") {
                if (auto* var = dynamic_cast<const VariableExpr*>(bin.left.get()))
                    return Value(is_defined(var->name));
                Value left = eval(*bin.left);
                return Value(!left.is_none());
            }
            Value left = eval(*bin.left);
            return Value(eval_type_test(test_name, left));
        }
        if (bin.op.size() > 7 && bin.op.substr(0, 7) == "is not ") {
            std::string test_name = bin.op.substr(7);
            if (test_name == "defined") {
                if (auto* var = dynamic_cast<const VariableExpr*>(bin.left.get()))
                    return Value(!is_defined(var->name));
                Value left = eval(*bin.left);
                return Value(left.is_none());
            }
            Value left = eval(*bin.left);
            return Value(!eval_type_test(test_name, left));
        }

        Value left = eval(*bin.left);
        Value right = eval(*bin.right);

        if (bin.op == "+")
            return left + right;
        if (bin.op == "~")
            return Value(left.to_string() + right.to_string());
        if (bin.op == "-") {
            if (left.is_int() && right.is_int())
                return Value(left.as_int() - right.as_int());
            return Value(left.to_number() - right.to_number());
        }
        if (bin.op == "*") {
            if (left.is_int() && right.is_int())
                return Value(left.as_int() * right.as_int());
            return Value(left.to_number() * right.to_number());
        }
        if (bin.op == "/") {
            double d = right.to_number();
            if (d == 0.0)
                return Value(0.0);
            return Value(left.to_number() / d);
        }
        if (bin.op == "%") {
            if (left.is_int() && right.is_int()) {
                int64_t r = right.as_int();
                if (r == 0)
                    return Value(int64_t(0));
                return Value(left.as_int() % r);
            }
            double d = right.to_number();
            if (d == 0.0)
                return Value(0.0);
            return Value(std::fmod(left.to_number(), d));
        }
        if (bin.op == "==")
            return Value(left == right);
        if (bin.op == "!=")
            return Value(left != right);
        if (bin.op == "<")
            return Value(left < right);
        if (bin.op == ">")
            return Value(left > right);
        if (bin.op == "<=")
            return Value(left <= right);
        if (bin.op == ">=")
            return Value(left >= right);
        if (bin.op == "in")
            return Value(right.contains(left));
        if (bin.op == "not in")
            return Value(!right.contains(left));

        return Value();
    }

    Value eval_unary(const UnaryExpr& u) {
        Value val = eval(*u.operand);
        if (u.op == "not")
            return Value(!val.truthy());
        if (u.op == "-") {
            if (val.is_int())
                return Value(-val.as_int());
            return Value(-val.to_number());
        }
        return val;
    }

    Value eval_filter(const FilterExpr& f) {
        Value val = eval(*f.value);

        if (f.name == "trim") {
            if (!val.is_string())
                return val;
            std::string s = val.as_string();
            // Strip leading whitespace
            size_t start = 0;
            while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
                start++;
            // Strip trailing whitespace
            size_t end = s.size();
            while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
                end--;
            return Value(s.substr(start, end - start));
        }
        if (f.name == "length") {
            return Value(val.length());
        }
        if (f.name == "default" || f.name == "d") {
            if (val.is_none() || (!val.truthy() && !f.args.empty())) {
                // default(val, boolean=false): if boolean=true, use default for falsy values too
                // Simple version: return default if none
                if (val.is_none() && !f.args.empty())
                    return eval(*f.args[0]);
                if (!val.is_none())
                    return val;
                if (!f.args.empty())
                    return eval(*f.args[0]);
                return Value("");
            }
            return val;
        }
        if (f.name == "first") {
            if (val.is_array() && !val.as_array().empty())
                return val.as_array().front();
            return Value();
        }
        if (f.name == "last") {
            if (val.is_array() && !val.as_array().empty())
                return val.as_array().back();
            return Value();
        }
        if (f.name == "upper") {
            if (!val.is_string())
                return val;
            std::string s = val.as_string();
            for (auto& c : s)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            return Value(std::move(s));
        }
        if (f.name == "lower") {
            if (!val.is_string())
                return val;
            std::string s = val.as_string();
            for (auto& c : s)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return Value(std::move(s));
        }
        if (f.name == "join") {
            if (!val.is_array())
                return val;
            std::string sep;
            if (!f.args.empty()) {
                Value s = eval(*f.args[0]);
                sep = s.to_string();
            }
            std::string result;
            auto& arr = val.as_array();
            for (size_t i = 0; i < arr.size(); i++) {
                if (i > 0)
                    result += sep;
                result += arr[i].to_string();
            }
            return Value(std::move(result));
        }
        if (f.name == "string") {
            return Value(val.to_string());
        }
        if (f.name == "int") {
            return Value(static_cast<int64_t>(val.to_number()));
        }
        if (f.name == "float") {
            return Value(val.to_number());
        }
        if (f.name == "list") {
            if (val.is_array())
                return val;
            if (val.is_string()) {
                Value::Array arr;
                for (char c : val.as_string())
                    arr.push_back(Value(std::string(1, c)));
                return Value(std::move(arr));
            }
            return Value(Value::Array{});
        }
        if (f.name == "items") {
            // dict | items -> array of [key, value] pairs
            if (val.is_object() && val.as_object()) {
                Value::Array arr;
                for (auto& [k, v] : *val.as_object()) {
                    Value::Array pair;
                    pair.push_back(Value(k));
                    pair.push_back(v);
                    arr.push_back(Value(std::move(pair)));
                }
                return Value(std::move(arr));
            }
            return Value(Value::Array{});
        }
        if (f.name == "replace") {
            if (!val.is_string() || f.args.size() < 2)
                return val;
            std::string s = val.as_string();
            std::string old_str = eval(*f.args[0]).to_string();
            std::string new_str = eval(*f.args[1]).to_string();
            if (old_str.empty())
                return val;
            size_t pos = 0;
            while ((pos = s.find(old_str, pos)) != std::string::npos) {
                s.replace(pos, old_str.size(), new_str);
                pos += new_str.size();
            }
            return Value(std::move(s));
        }
        if (f.name == "title") {
            if (!val.is_string())
                return val;
            std::string s = val.as_string();
            bool cap_next = true;
            for (auto& c : s) {
                if (std::isspace(static_cast<unsigned char>(c))) {
                    cap_next = true;
                } else if (cap_next) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                    cap_next = false;
                }
            }
            return Value(std::move(s));
        }
        if (f.name == "capitalize") {
            if (!val.is_string())
                return val;
            std::string s = val.as_string();
            if (!s.empty())
                s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
            return Value(std::move(s));
        }
        if (f.name == "count") {
            return Value(val.length());
        }
        if (f.name == "reverse") {
            if (val.is_array()) {
                Value::Array arr = val.as_array();
                std::reverse(arr.begin(), arr.end());
                return Value(std::move(arr));
            }
            if (val.is_string()) {
                std::string s = val.as_string();
                std::reverse(s.begin(), s.end());
                return Value(std::move(s));
            }
            return val;
        }
        if (f.name == "tojson") {
            return Value(value_to_json(val));
        }
        if (f.name == "selectattr" || f.name == "rejectattr" || f.name == "map" || f.name == "select" ||
            f.name == "reject" || f.name == "batch" || f.name == "slice" || f.name == "sort" ||
            f.name == "unique" || f.name == "groupby") {
            // Unsupported filters — return value as-is
            IMP_LOG_WARN("jinja: unsupported filter '%s'", f.name.c_str());
            return val;
        }

        IMP_LOG_WARN("jinja: unknown filter '%s'", f.name.c_str());
        return val;
    }

    void register_macro(const MacroNode& node) { macros_[node.name] = &node; }

    Value call_macro(const MacroNode& macro, const CallExpr& call) {
        push_scope();
        // Bind positional args
        for (size_t i = 0; i < macro.params.size(); i++) {
            if (i < call.args.size()) {
                set_var(macro.params[i].name, eval(*call.args[i]));
            } else {
                // Check kwargs
                bool found = false;
                for (auto& [k, v] : call.kwargs) {
                    if (k == macro.params[i].name) {
                        set_var(k, eval(*v));
                        found = true;
                        break;
                    }
                }
                if (!found && macro.params[i].default_value) {
                    set_var(macro.params[i].name, eval(*macro.params[i].default_value));
                } else if (!found) {
                    set_var(macro.params[i].name, Value());
                }
            }
        }
        // Render macro body
        std::string result;
        for (auto& node : macro.body) {
            render_node(*node, result);
        }
        pop_scope();
        return Value(result);
    }

    Value eval_call(const CallExpr& call) {
        // Check if callee is a variable name (built-in function or macro)
        if (auto* var = dynamic_cast<const VariableExpr*>(call.callee.get())) {
            // Check macros first
            auto macro_it = macros_.find(var->name);
            if (macro_it != macros_.end()) {
                return call_macro(*macro_it->second, call);
            }
            if (var->name == "namespace") {
                auto obj = Value::make_object();
                // Initialize with keyword args
                for (auto& [k, v] : call.kwargs) {
                    obj.set(k, eval(*v));
                }
                return obj;
            }
            if (var->name == "range") {
                Value::Array arr;
                if (call.args.size() == 1) {
                    int64_t n = eval(*call.args[0]).as_int();
                    for (int64_t i = 0; i < n; i++)
                        arr.push_back(Value(i));
                } else if (call.args.size() >= 2) {
                    int64_t start = eval(*call.args[0]).as_int();
                    int64_t end = eval(*call.args[1]).as_int();
                    int64_t step = (call.args.size() >= 3) ? eval(*call.args[2]).as_int() : 1;
                    if (step > 0) {
                        for (int64_t i = start; i < end; i += step)
                            arr.push_back(Value(i));
                    } else if (step < 0) {
                        for (int64_t i = start; i > end; i += step)
                            arr.push_back(Value(i));
                    }
                }
                return Value(std::move(arr));
            }
            if (var->name == "raise_exception") {
                std::string msg = "Template error";
                if (!call.args.empty())
                    msg = eval(*call.args[0]).to_string();
                IMP_LOG_WARN("jinja: raise_exception: %s", msg.c_str());
                return Value();
            }
            if (var->name == "dict") {
                auto obj = Value::make_object();
                for (auto& [k, v] : call.kwargs) {
                    obj.set(k, eval(*v));
                }
                return obj;
            }
            if (var->name == "cycler") {
                // Minimal cycler — return first arg
                if (!call.args.empty())
                    return eval(*call.args[0]);
                return Value();
            }
            if (var->name == "joiner") {
                // Returns a callable that returns "" first call, then sep
                std::string sep = ", ";
                if (!call.args.empty())
                    sep = eval(*call.args[0]).to_string();
                // Can't truly implement callable — return empty
                return Value(sep);
            }
            if (var->name == "strftime_now") {
                // strftime_now(format_string) — returns current time formatted
                // Used by some chat templates to inject current date/time
                std::string fmt = "%Y-%m-%d %H:%M:%S";
                if (!call.args.empty())
                    fmt = eval(*call.args[0]).to_string();
                std::time_t now = std::time(nullptr);
                std::tm tm_buf{};
                localtime_r(&now, &tm_buf);  // std::localtime's static buffer races across server threads
                char buf[256];
                std::strftime(buf, sizeof(buf), fmt.c_str(), &tm_buf);
                return Value(std::string(buf));
            }
        }

        // Generic: evaluate callee and try calling it
        // Jinja templates don't have first-class callables in our subset
        Value callee = eval(*call.callee);
        return callee;
    }

    Value eval_method(const MethodExpr& m) {
        Value obj = eval(*m.object);

        // String methods
        if (obj.is_string()) {
            const std::string& s = obj.as_string();

            if (m.method == "strip") {
                std::string chars;
                if (!m.args.empty())
                    chars = eval(*m.args[0]).to_string();
                size_t start = 0;
                if (chars.empty()) {
                    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
                        start++;
                } else {
                    while (start < s.size() && chars.find(s[start]) != std::string::npos)
                        start++;
                }
                size_t end = s.size();
                if (chars.empty()) {
                    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
                        end--;
                } else {
                    while (end > start && chars.find(s[end - 1]) != std::string::npos)
                        end--;
                }
                return Value(s.substr(start, end - start));
            }
            if (m.method == "lstrip") {
                std::string chars;
                if (!m.args.empty())
                    chars = eval(*m.args[0]).to_string();
                size_t start = 0;
                if (chars.empty()) {
                    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
                        start++;
                } else {
                    while (start < s.size() && chars.find(s[start]) != std::string::npos)
                        start++;
                }
                return Value(s.substr(start));
            }
            if (m.method == "rstrip") {
                std::string chars;
                if (!m.args.empty())
                    chars = eval(*m.args[0]).to_string();
                size_t end = s.size();
                if (chars.empty()) {
                    while (end > 0 && std::isspace(static_cast<unsigned char>(s[end - 1])))
                        end--;
                } else {
                    while (end > 0 && chars.find(s[end - 1]) != std::string::npos)
                        end--;
                }
                return Value(s.substr(0, end));
            }
            if (m.method == "startswith") {
                if (m.args.empty())
                    return Value(false);
                std::string prefix = eval(*m.args[0]).to_string();
                return Value(s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0);
            }
            if (m.method == "endswith") {
                if (m.args.empty())
                    return Value(false);
                std::string suffix = eval(*m.args[0]).to_string();
                return Value(s.size() >= suffix.size() &&
                             s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0);
            }
            if (m.method == "upper") {
                std::string r = s;
                for (auto& c : r)
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                return Value(std::move(r));
            }
            if (m.method == "lower") {
                std::string r = s;
                for (auto& c : r)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                return Value(std::move(r));
            }
            if (m.method == "replace") {
                if (m.args.size() < 2)
                    return obj;
                std::string old_str = eval(*m.args[0]).to_string();
                std::string new_str = eval(*m.args[1]).to_string();
                if (old_str.empty())
                    return obj;
                std::string r = s;
                size_t pos = 0;
                while ((pos = r.find(old_str, pos)) != std::string::npos) {
                    r.replace(pos, old_str.size(), new_str);
                    pos += new_str.size();
                }
                return Value(std::move(r));
            }
            if (m.method == "split") {
                Value::Array arr;
                if (m.args.empty()) {
                    // Split on whitespace
                    std::istringstream iss(s);
                    std::string word;
                    while (iss >> word)
                        arr.push_back(Value(std::move(word)));
                } else {
                    std::string sep = eval(*m.args[0]).to_string();
                    if (sep.empty()) {
                        arr.push_back(obj);
                    } else {
                        size_t start = 0;
                        while (true) {
                            size_t pos = s.find(sep, start);
                            if (pos == std::string::npos) {
                                arr.push_back(Value(s.substr(start)));
                                break;
                            }
                            arr.push_back(Value(s.substr(start, pos - start)));
                            start = pos + sep.size();
                        }
                    }
                }
                return Value(std::move(arr));
            }
            if (m.method == "find") {
                if (m.args.empty())
                    return Value(int64_t(-1));
                std::string sub = eval(*m.args[0]).to_string();
                size_t pos = s.find(sub);
                return Value(pos != std::string::npos ? static_cast<int64_t>(pos) : int64_t(-1));
            }
            if (m.method == "count") {
                if (m.args.empty())
                    return Value(int64_t(0));
                std::string sub = eval(*m.args[0]).to_string();
                if (sub.empty())
                    return Value(int64_t(0));
                int64_t count = 0;
                size_t pos = 0;
                while ((pos = s.find(sub, pos)) != std::string::npos) {
                    count++;
                    pos += sub.size();
                }
                return Value(count);
            }
            if (m.method == "join") {
                // string.join doesn't exist in Jinja, but handle gracefully
                return obj;
            }
        }

        // Array methods
        if (obj.is_array()) {
            if (m.method == "append") {
                // Arrays are immutable in our model — return new array
                Value::Array arr = obj.as_array();
                if (!m.args.empty())
                    arr.push_back(eval(*m.args[0]));
                return Value(std::move(arr));
            }
            if (m.method == "insert") {
                Value::Array arr = obj.as_array();
                if (m.args.size() >= 2) {
                    int64_t idx = eval(*m.args[0]).as_int();
                    if (idx < 0)
                        idx = 0;
                    if (idx > static_cast<int64_t>(arr.size()))
                        idx = static_cast<int64_t>(arr.size());
                    arr.insert(arr.begin() + idx, eval(*m.args[1]));
                }
                return Value(std::move(arr));
            }
        }

        // Object/dict methods
        if (obj.is_object() && obj.as_object()) {
            if (m.method == "items") {
                Value::Array arr;
                for (auto& [k, v] : *obj.as_object()) {
                    Value::Array pair;
                    pair.push_back(Value(k));
                    pair.push_back(v);
                    arr.push_back(Value(std::move(pair)));
                }
                return Value(std::move(arr));
            }
            if (m.method == "keys") {
                Value::Array arr;
                for (auto& [k, v] : *obj.as_object()) {
                    (void)v;
                    arr.push_back(Value(k));
                }
                return Value(std::move(arr));
            }
            if (m.method == "values") {
                Value::Array arr;
                for (auto& [k, v] : *obj.as_object()) {
                    (void)k;
                    arr.push_back(v);
                }
                return Value(std::move(arr));
            }
            if (m.method == "get") {
                if (m.args.empty())
                    return Value();
                std::string key = eval(*m.args[0]).to_string();
                auto it = obj.as_object()->find(key);
                if (it != obj.as_object()->end())
                    return it->second;
                if (m.args.size() >= 2)
                    return eval(*m.args[1]);
                return Value();
            }
            if (m.method == "update") {
                // Mutate in place via shared_ptr
                if (!m.args.empty()) {
                    Value other = eval(*m.args[0]);
                    if (other.is_object() && other.as_object()) {
                        for (auto& [k, v] : *other.as_object()) {
                            (*obj.as_object())[k] = v;
                        }
                    }
                }
                return obj;
            }
        }

        IMP_LOG_WARN("jinja: unknown method '%s'", m.method.c_str());
        return Value();
    }

    Value eval_slice(const SliceExpr& sl) {
        Value obj = eval(*sl.object);
        int64_t len = obj.length();

        // Resolve start/stop/step with Python-style defaults
        int64_t step = sl.step ? static_cast<int64_t>(eval(*sl.step).to_number()) : 1;
        if (step == 0)
            step = 1;  // avoid infinite loop

        int64_t start, stop;
        if (step > 0) {
            start = sl.start ? static_cast<int64_t>(eval(*sl.start).to_number()) : 0;
            stop = sl.stop ? static_cast<int64_t>(eval(*sl.stop).to_number()) : len;
        } else {
            start = sl.start ? static_cast<int64_t>(eval(*sl.start).to_number()) : len - 1;
            stop = sl.stop ? static_cast<int64_t>(eval(*sl.stop).to_number()) : -(len + 1);
        }

        // Normalize negative indices
        if (start < 0)
            start += len;
        if (stop < 0)
            stop += len;

        // Clamp
        if (step > 0) {
            if (start < 0)
                start = 0;
            if (start > len)
                start = len;
            if (stop < 0)
                stop = 0;
            if (stop > len)
                stop = len;
        } else {
            if (start < -1)
                start = -1;
            if (start >= len)
                start = len - 1;
            if (stop < -1)
                stop = -1;
            if (stop >= len)
                stop = len - 1;
        }

        if (obj.is_array()) {
            auto& arr = obj.as_array();
            Value::Array result;
            if (step > 0) {
                for (int64_t i = start; i < stop; i += step)
                    result.push_back(arr[static_cast<size_t>(i)]);
            } else {
                for (int64_t i = start; i > stop; i += step)
                    result.push_back(arr[static_cast<size_t>(i)]);
            }
            return Value(std::move(result));
        }
        if (obj.is_string()) {
            auto& s = obj.as_string();
            std::string result;
            if (step > 0) {
                for (int64_t i = start; i < stop; i += step)
                    result += s[static_cast<size_t>(i)];
            } else {
                for (int64_t i = start; i > stop; i += step)
                    result += s[static_cast<size_t>(i)];
            }
            return Value(std::move(result));
        }
        return Value();
    }

    static std::string value_to_json(const Value& val) {
        if (val.is_none())
            return "null";
        if (val.is_bool())
            return val.as_bool() ? "true" : "false";
        if (val.is_int())
            return std::to_string(val.as_int());
        if (val.is_double()) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%g", val.as_double());
            // Ensure it looks like a number (has decimal or exponent)
            std::string s(buf);
            if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
                s.find('E') == std::string::npos)
                s += ".0";
            return s;
        }
        if (val.is_string()) {
            std::string r = "\"";
            for (char c : val.as_string()) {
                switch (c) {
                    case '"':
                        r += "\\\"";
                        break;
                    case '\\':
                        r += "\\\\";
                        break;
                    case '\n':
                        r += "\\n";
                        break;
                    case '\r':
                        r += "\\r";
                        break;
                    case '\t':
                        r += "\\t";
                        break;
                    default:
                        if (static_cast<unsigned char>(c) < 0x20) {
                            char hex[8];
                            std::snprintf(hex, sizeof(hex), "\\u%04x", static_cast<unsigned char>(c));
                            r += hex;
                        } else {
                            r += c;
                        }
                }
            }
            r += "\"";
            return r;
        }
        if (val.is_array()) {
            std::string r = "[";
            auto& arr = val.as_array();
            for (size_t i = 0; i < arr.size(); i++) {
                if (i > 0)
                    r += ", ";
                r += value_to_json(arr[i]);
            }
            r += "]";
            return r;
        }
        if (val.is_object() && val.as_object()) {
            std::string r = "{";
            bool first = true;
            for (auto& [k, v] : *val.as_object()) {
                if (!first)
                    r += ", ";
                r += "\"" + k + "\": " + value_to_json(v);
                first = false;
            }
            r += "}";
            return r;
        }
        return "null";
    }

    std::vector<Scope> scopes_;
    std::map<std::string, const MacroNode*> macros_;
};

}  // namespace detail

// ============================================================================
// Template implementation
// ============================================================================

Template::Template() = default;
Template::~Template() = default;
Template::Template(Template&&) noexcept = default;
Template& Template::operator=(Template&&) noexcept = default;

bool Template::parse(const std::string& source_in) {
    error_.clear();
    nodes_.clear();

    // Match Jinja2's keep_trailing_newline=False default — the setting HF
    // transformers and vLLM use when applying chat templates: strip a single
    // trailing newline from the template source. Without this, a template file
    // that ends in a newline (Qwen3-Coder's chat_template.jinja ends
    // "{%- endif %}\n") renders "<|im_start|>assistant\n\n"; that extra blank
    // line makes the model emit an immediate EOS (empty completion) on
    // borderline multi-turn contexts. Templates without a trailing newline
    // (Qwen3 / Modelopt) are unaffected. Strips one \n, \r\n, or \r.
    std::string source = source_in;
    if (source.size() >= 2 && source[source.size() - 2] == '\r' && source.back() == '\n')
        source.erase(source.size() - 2);
    else if (!source.empty() && (source.back() == '\n' || source.back() == '\r'))
        source.pop_back();

    detail::Lexer lexer(source);
    auto tokens = lexer.tokenize();

    detail::Parser parser(tokens);
    if (!parser.parse(nodes_)) {
        error_ = "Parse error";
        return false;
    }
    return true;
}

std::string Template::render(const Context& ctx) const {
    detail::Evaluator eval(ctx);
    return eval.render(nodes_);
}

std::string Template::render_string(const std::string& source, const Context& ctx) {
    Template tmpl;
    if (!tmpl.parse(source)) {
        IMP_LOG_WARN("jinja: failed to parse template: %s", tmpl.error().c_str());
        return "";
    }
    return tmpl.render(ctx);
}

}  // namespace imp::jinja
