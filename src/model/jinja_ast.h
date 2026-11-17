// Jinja AST shared by the parser (jinja_parse.cpp) and the evaluator (jinja.cpp).
#pragma once

#include "model/jinja.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace imp::jinja::detail {

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
    std::vector<std::pair<std::string, std::unique_ptr<Expr>>> kwargs;  // tojson(ensure_ascii=False)
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
    std::vector<std::pair<std::string, std::unique_ptr<Expr>>> kwargs;
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
    std::unique_ptr<Expr> filter;  // for x in xs if cond: loop.* counts the kept items only
    std::vector<std::unique_ptr<Node>> body;
    std::vector<std::unique_ptr<Node>> else_body;
    bool recursive = false;
};

// {% break %} / {% continue %} (jinja2.ext.loopcontrols, Cohere2 template).
struct LoopCtlNode : Node {
    explicit LoopCtlNode(bool brk) : is_break(brk) {}
    bool is_break;
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
    // Block form: {% set x %}...{% endset %}. The body renders into a string
    // and that string is the value; `value` is null in this form.
    std::vector<std::unique_ptr<Node>> body;
    bool is_block = false;
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

// Lexes and parses `source` (trailing newline already stripped). false on a parse error;
// `unknown_tag` names an unsupported or unbalanced tag when that was the cause.
[[nodiscard]] bool parse_template(const std::string& source, std::vector<std::unique_ptr<Node>>& out,
                                  std::string& unknown_tag);

}  // namespace imp::jinja::detail
