// Jinja2 lexer and parser: template source -> AST (jinja_ast.h).

#include "model/jinja_ast.h"
#include "model/jinja_lex.h"
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

        // No conditional expression here: `for x in xs if cond` filters (Jinja parse_tuple,
        // with_condexpr=False).
        node->iterable = parse_or();
        if (check(TokenType::IDENT, "if")) {
            advance();
            node->filter = parse_or();
        }

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
                false_val = parse_ternary();  // 'a' if x else 'b' if y else 'c'
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

    // `x is [not] name [arg]` / `x is [not] name(arg)`: BinOpExpr op "is name" / "is not name",
    // right = the argument or null. Jinja parse_test: a bare argument follows unless it is else/or/and.
    std::unique_ptr<Expr> parse_is_test(std::unique_ptr<Expr> left, bool negated) {
        auto bin = std::make_unique<BinOpExpr>();
        bin->left = std::move(left);
        if (!check(TokenType::IDENT)) {
            bin->op = negated ? "!=" : "==";  // `x is <expr>`: equality, as before
            bin->right = parse_addition();
            return bin;
        }
        bin->op = (negated ? "is not " : "is ") + peek().value;
        advance();
        if (check(TokenType::LPAREN)) {
            advance();
            if (!check(TokenType::RPAREN))
                bin->right = parse_expr();
            if (check(TokenType::RPAREN))
                advance();
        } else if ((check(TokenType::NUMBER) || check(TokenType::STRING) || check(TokenType::LBRACKET) ||
                    check(TokenType::LBRACE) || check(TokenType::IDENT)) &&
                   !check(TokenType::IDENT, "else") && !check(TokenType::IDENT, "if") &&
                   !check(TokenType::OP, "and") && !check(TokenType::OP, "or") &&
                   !check(TokenType::IDENT, "and") && !check(TokenType::IDENT, "or")) {
            bin->right = parse_primary();
        }
        return bin;
    }

    std::unique_ptr<Expr> parse_comparison() {
        auto left = parse_addition();
        while (true) {
            if (check(TokenType::OP, "is") || check(TokenType::OP, "is not")) {
                const bool negated = check(TokenType::OP, "is not");
                advance();
                left = parse_is_test(std::move(left), negated);
                continue;
            }
            if (!(check(TokenType::OP, "==") || check(TokenType::OP, "!=") || check(TokenType::OP, "<") ||
                  check(TokenType::OP, ">") || check(TokenType::OP, "<=") || check(TokenType::OP, ">=") ||
                  check(TokenType::OP, "in") || check(TokenType::OP, "not in")))
                break;
            auto bin = std::make_unique<BinOpExpr>();
            bin->op = peek().value;
            advance();
            bin->left = std::move(left);
            bin->right = parse_addition();
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
        while (check(TokenType::OP, "*") || check(TokenType::OP, "/") || check(TokenType::OP, "//") ||
               check(TokenType::OP, "%")) {
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
        return parse_power();
    }

    // `**` binds tighter than unary minus on its left and is right-associative: -2 ** 2 == -4.
    std::unique_ptr<Expr> parse_power() {
        auto base = parse_postfix();
        if (!check(TokenType::OP, "**"))
            return base;
        advance();
        auto bin = std::make_unique<BinOpExpr>();
        bin->op = "**";
        bin->left = std::move(base);
        bin->right = parse_unary();
        return bin;
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
                    if (!check(TokenType::RPAREN))
                        parse_call_args(method->args, method->kwargs);
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
                    // Keyword args too: `v | tojson(ensure_ascii=False) if ...` lost the rest of the
                    // expression at '=' (GLM-4.7 / Flint tool-call replay quoted string args).
                    if (!check(TokenType::RPAREN))
                        parse_call_args(filter->args, filter->kwargs);
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
            if (val.find_first_of(".eE") != std::string::npos) {
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

bool parse_template(const std::string& source, std::vector<std::unique_ptr<Node>>& out,
                    std::string& unknown_tag) {
    auto tokens = tokenize(source);
    Parser parser(tokens);
    if (!parser.parse(out)) {
        unknown_tag = parser.unknown_tag();
        return false;
    }
    return true;
}

}  // namespace imp::jinja::detail
