// Jinja2 template engine for LLM chat templates (GGUF).
// Supports: variables, for/if/set, filters, operators, namespace(), whitespace control.
// Pure C++20, no regex, no external dependencies.

#include "model/jinja_ast.h"
#include "core/logging.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <charconv>
#include <cmath>
#include <ctime>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace imp::jinja {

namespace detail {

namespace {

// ============================================================================
// Evaluator
// ============================================================================

// Evaluation is bounded like parsing (kMaxParseDepth), except macro self-recursion
// (eval_call->call_macro->render_node->eval->eval_call, no node-depth stop) and
// range(10000000000) materializing one Value per element. Both come from the checkpoint's
// chat_template (AUDIT_ARCH_2026 F1-9); evaluator throws, render() returns "" + error().
struct EvalBudgetExceeded : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Test of select/reject/selectattr/rejectattr; empty test = truthiness, unknown test = false (logged).
static bool select_test_passes(const std::string& test, const Value& v, const Value& arg) {
    if (test.empty())
        return v.truthy();
    if (test == "equalto" || test == "eq" || test == "==" || test == "sameas")
        return v == arg;
    if (test == "ne" || test == "!=")
        return !(v == arg);
    if (test == "defined" || test == "undefined" || test == "none")
        return v.is_none() == (test != "defined");
    if (test == "string")
        return v.is_string();
    if (test == "in")
        return arg.is_array() && std::ranges::any_of(arg.as_array(), [&](const Value& x) { return x == v; });
    IMP_LOG_WARN("jinja: unsupported select test '%s'", test.c_str());
    return false;
}

class Evaluator {
public:
    // 256 mirrors the parse cap; the deepest macro chain in the zoo is 3.
    static constexpr int kMaxCallDepth = 256;
    // Iterations across every loop of one render, and the longest `range`.
    // The largest real template loops over messages (< 10 000, the server's
    // cap) times a handful of tools.
    static constexpr int64_t kMaxLoopIterations = int64_t{1} << 20;

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

    int call_depth_ = 0;
    int64_t loop_iterations_ = 0;
    int loop_ctl_ = 0;  // pending {% break %} (1) / {% continue %} (2): skips nodes until the for consumes it

    // One loop iteration; false = a {% break %} ended the loop.
    bool render_loop_body_(const std::vector<std::unique_ptr<Node>>& body, std::string& out) {
        for (const auto& n : body)
            render_node(*n, out);
        return std::exchange(loop_ctl_, 0) != 1;
    }

    struct CallDepthGuard {
        Evaluator& e;
        explicit CallDepthGuard(Evaluator& ev) : e(ev) {
            if (e.call_depth_ >= kMaxCallDepth)
                throw EvalBudgetExceeded("macro call depth exceeds " + std::to_string(kMaxCallDepth));
            ++e.call_depth_;
        }
        ~CallDepthGuard() { --e.call_depth_; }
    };

    void count_iteration() {
        if (++loop_iterations_ > kMaxLoopIterations)
            throw EvalBudgetExceeded("loop iterations exceed " + std::to_string(kMaxLoopIterations));
    }

    // Length of range(start, end, step) without materialising it; the
    // subtraction is done unsigned so INT64_MIN..INT64_MAX cannot overflow.
    static uint64_t range_length(int64_t start, int64_t end, int64_t step) {
        if (step > 0 && end > start)
            return ((static_cast<uint64_t>(end) - static_cast<uint64_t>(start) - 1) /
                    static_cast<uint64_t>(step)) +
                   1;
        if (step < 0 && start > end)
            return ((static_cast<uint64_t>(start) - static_cast<uint64_t>(end) - 1) /
                    (uint64_t{0} - static_cast<uint64_t>(step))) +
                   1;
        return 0;
    }

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
            if (it->contains(name))
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
        if (loop_ctl_ != 0)
            return;
        if (auto* ctl = dynamic_cast<const LoopCtlNode*>(&node)) {
            loop_ctl_ = ctl->is_break ? 1 : 2;
        } else if (auto* text = dynamic_cast<const TextNode*>(&node)) {
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
                count_iteration();
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

                if (!render_loop_body_(node.body, out))
                    break;
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
            count_iteration();
            auto loop = Value::make_object();
            loop.set("index", Value(i + 1));
            loop.set("index0", Value(i));
            loop.set("first", Value(i == 0));
            loop.set("last", Value(i == len - 1));
            loop.set("length", Value(len));
            // Unset at the edges, as Jinja2's undefined: Qwen3.8 / Nemotron templates open the
            // user turn before the first of consecutive tool messages via loop.previtem.
            if (i > 0)
                loop.set("previtem", arr[static_cast<size_t>(i - 1)]);
            if (i + 1 < len)
                loop.set("nextitem", arr[static_cast<size_t>(i + 1)]);
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

            if (!render_loop_body_(node.body, out))
                break;
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
        if (node.is_block) {
            std::string captured;
            for (const auto& n : node.body)
                render_node(*n, captured);
            if (!node.attr_name.empty()) {
                Value ns = lookup(node.var_name);
                if (ns.is_object())
                    ns.set(node.attr_name, Value(std::move(captured)));
            } else {
                set_var_update(node.var_name, Value(std::move(captured)));
            }
            return;
        }
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

        if (bin.op.size() > 3 && bin.op.starts_with("is ") && bin.op.substr(3, 4) != "not ") {
            std::string test_name = bin.op.substr(3);
            // For "is defined", check variable existence rather than value
            if (test_name == "defined") {
                if (auto* var = dynamic_cast<const VariableExpr*>(bin.left.get()))
                    return Value(is_defined(var->name));
                Value left = eval(*bin.left);
                return Value(!left.is_none());
            }
            // "is undefined" is the exact inverse of truthiness and must not use value truthiness: a
            // variable stamped false is still defined. Falling through to generic equality made
            // `false == none` compare true, reading an explicit false as "caller said nothing".
            if (test_name == "undefined") {
                if (auto* var = dynamic_cast<const VariableExpr*>(bin.left.get()))
                    return Value(!is_defined(var->name));
                Value left = eval(*bin.left);
                return Value(left.is_none());
            }
            Value left = eval(*bin.left);
            return Value(eval_type_test(test_name, left));
        }
        if (bin.op.size() > 7 && bin.op.starts_with("is not ")) {
            std::string test_name = bin.op.substr(7);
            if (test_name == "defined") {
                if (auto* var = dynamic_cast<const VariableExpr*>(bin.left.get()))
                    return Value(!is_defined(var->name));
                Value left = eval(*bin.left);
                return Value(left.is_none());
            }
            if (test_name == "undefined") {
                if (auto* var = dynamic_cast<const VariableExpr*>(bin.left.get()))
                    return Value(is_defined(var->name));
                Value left = eval(*bin.left);
                return Value(!left.is_none());
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

        // No autoescaping here, so `safe` is the identity - but an UNKNOWN
        // filter is not: it warned and dropped the value, which is how a
        // template silently loses whatever it marked safe (#1572).
        if (f.name == "safe")
            return val;
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
                    arr.emplace_back(std::string(1, c));
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
                    pair.emplace_back(k);
                    pair.push_back(v);
                    arr.emplace_back(std::move(pair));
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
        if (f.name == "selectattr" || f.name == "rejectattr" || f.name == "select" || f.name == "reject")
            return select_filter_(f, val);
        if (f.name == "map" || f.name == "batch" || f.name == "slice" || f.name == "sort" ||
            f.name == "unique" || f.name == "groupby") {
            // Unsupported filters — return value as-is
            IMP_LOG_WARN("jinja: unsupported filter '%s'", f.name.c_str());
            return val;
        }

        IMP_LOG_WARN("jinja: unknown filter '%s'", f.name.c_str());
        return val;
    }

    // select/reject(test, arg) and selectattr/rejectattr(attr, test, arg); no test = truthiness.
    // Olmo-3 counts system turns with messages|selectattr('role', 'equalto', 'system')|list|length.
    Value select_filter_(const FilterExpr& f, const Value& val) {
        if (!val.is_array())
            return val;
        const bool by_attr = f.name == "selectattr" || f.name == "rejectattr";
        const bool keep = f.name == "selectattr" || f.name == "select";
        std::vector<Value> args;
        for (const auto& a : f.args)
            args.push_back(eval(*a));
        const size_t t0 = by_attr ? 1 : 0;
        if (by_attr && (args.empty() || !args[0].is_string()))
            return val;
        const std::string test = args.size() > t0 && args[t0].is_string() ? args[t0].as_string() : "";
        const Value arg = args.size() > t0 + 1 ? args[t0 + 1] : Value();
        Value::Array out;
        for (const auto& item : val.as_array()) {
            const Value probe = by_attr ? (item.is_object() ? item.get(args[0].as_string()) : Value()) : item;
            if (select_test_passes(test, probe, arg) == keep)
                out.push_back(item);
        }
        return Value(std::move(out));
    }

    void register_macro(const MacroNode& node) { macros_[node.name] = &node; }

    Value call_macro(const MacroNode& macro, const CallExpr& call) {
        CallDepthGuard depth(*this);
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
                int64_t start = 0, end = 0, step = 1;
                if (call.args.size() == 1) {
                    end = eval(*call.args[0]).as_int();
                } else if (call.args.size() >= 2) {
                    start = eval(*call.args[0]).as_int();
                    end = eval(*call.args[1]).as_int();
                    step = (call.args.size() >= 3) ? eval(*call.args[2]).as_int() : 1;
                }
                if (range_length(start, end, step) > static_cast<uint64_t>(kMaxLoopIterations))
                    throw EvalBudgetExceeded("range length exceeds " + std::to_string(kMaxLoopIterations));
                if (call.args.size() == 1) {
                    for (int64_t i = 0; i < end; i++)
                        arr.emplace_back(i);
                } else if (call.args.size() >= 2) {
                    if (step > 0) {
                        for (int64_t i = start; i < end; i += step)
                            arr.emplace_back(i);
                    } else if (step < 0) {
                        for (int64_t i = start; i > end; i += step)
                            arr.emplace_back(i);
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
                    while (start < s.size() && chars.contains(s[start]))
                        start++;
                }
                size_t end = s.size();
                if (chars.empty()) {
                    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
                        end--;
                } else {
                    while (end > start && chars.contains(s[end - 1]))
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
                    while (start < s.size() && chars.contains(s[start]))
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
                    while (end > 0 && chars.contains(s[end - 1]))
                        end--;
                }
                return Value(s.substr(0, end));
            }
            if (m.method == "startswith") {
                if (m.args.empty())
                    return Value(false);
                std::string prefix = eval(*m.args[0]).to_string();
                return Value(s.size() >= prefix.size() && s.starts_with(prefix));
            }
            if (m.method == "endswith") {
                if (m.args.empty())
                    return Value(false);
                std::string suffix = eval(*m.args[0]).to_string();
                return Value(s.size() >= suffix.size() && s.ends_with(suffix));
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
                        arr.emplace_back(std::move(word));
                } else {
                    std::string sep = eval(*m.args[0]).to_string();
                    if (sep.empty()) {
                        arr.push_back(obj);
                    } else {
                        size_t start = 0;
                        while (true) {
                            size_t pos = s.find(sep, start);
                            if (pos == std::string::npos) {
                                arr.emplace_back(s.substr(start));
                                break;
                            }
                            arr.emplace_back(s.substr(start, pos - start));
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
                    idx = std::max<int64_t>(idx, 0);
                    idx = std::min(idx, static_cast<int64_t>(arr.size()));
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
                    pair.emplace_back(k);
                    pair.push_back(v);
                    arr.emplace_back(std::move(pair));
                }
                return Value(std::move(arr));
            }
            if (m.method == "keys") {
                Value::Array arr;
                for (auto& [k, v] : *obj.as_object()) {
                    (void)v;
                    arr.emplace_back(k);
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
            start = std::max<int64_t>(start, 0);
            start = std::min(start, len);
            stop = std::max<int64_t>(stop, 0);
            stop = std::min(stop, len);
        } else {
            start = std::max<int64_t>(start, -1);
            if (start >= len)
                start = len - 1;
            stop = std::max<int64_t>(stop, -1);
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

    // #1607: the tojson filter walks the value tree recursively, and that tree
    // comes from a request-supplied schema. Depth is bounded at the HTTP
    // boundary; this is the second line, for a tree built any other way.
    static std::string value_to_json(const Value& val, int depth = 0) {
        constexpr int kMaxJsonDepth = 128;
        if (depth >= kMaxJsonDepth)
            return "null";
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
            if (!s.contains('.') && !s.contains('e') && !s.contains('E'))
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
                r += value_to_json(arr[i], depth + 1);
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
                r += "\"" + k + "\": " + value_to_json(v, depth + 1);
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
}  // namespace

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

    // Matches Jinja2's keep_trailing_newline=False (HF transformers/vLLM default): strips one
    // trailing \n, \r\n, or \r from the template source. Without it a template ending in a
    // newline renders an extra blank line before "assistant", causing an immediate EOS on
    // borderline multi-turn contexts.
    std::string source = source_in;
    if (source.size() >= 2 && source[source.size() - 2] == '\r' && source.back() == '\n')
        source.erase(source.size() - 2);
    else if (!source.empty() && (source.back() == '\n' || source.back() == '\r'))
        source.pop_back();

    std::string unknown_tag;
    if (!detail::parse_template(source, nodes_, unknown_tag)) {
        error_ = unknown_tag.empty() ? "Parse error" : "unsupported or unbalanced tag: {% " + unknown_tag + " %}";
        return false;
    }
    return true;
}

std::string Template::render(const Context& ctx) const {
    detail::Evaluator eval(ctx);
    try {
        return eval.render(nodes_);
    } catch (const detail::EvalBudgetExceeded& e) {
        error_ = e.what();
        IMP_LOG_WARN("jinja: render refused: %s", e.what());
        return std::string();
    }
}

}  // namespace imp::jinja
