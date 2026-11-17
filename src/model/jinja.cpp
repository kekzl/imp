// Jinja2 template engine for LLM chat templates (GGUF).
// Supports: variables, for/if/set, filters, operators, namespace(), whitespace control.
// Pure C++20, no regex, no external dependencies.

#include "model/jinja_ast.h"
#include "model/jinja_builtins.h"
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
#include <optional>
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
        // Globals of transformers' env: Llama 3.x picks today's date via `strftime_now is defined`.
        return name == "raise_exception" || name == "strftime_now" || name == "namespace" ||
               name == "range" || name == "dict" || macros_.contains(name);
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

    // Items a for loop walks: list items, dict keys ((key, value) with two loop vars), string chars.
    static Value::Array loop_items(const ForNode& node, const Value& iterable) {
        Value::Array items;
        if (iterable.is_array()) {
            items = iterable.as_array();
        } else if (iterable.is_object() && iterable.as_object()) {
            for (const auto& [k, v] : *iterable.as_object())
                items.push_back(node.var_name2.empty() ? Value(k) : Value::array({Value(k), v}));
        } else if (iterable.is_string()) {
            for (char c : iterable.as_string())
                items.emplace_back(std::string(1, c));
        }
        return items;
    }

    void bind_loop_vars(const ForNode& node, const Value& item) {
        if (node.var_name2.empty()) {
            set_var(node.var_name, item);
            return;
        }
        // Tuple unpacking: for a, b in list_of_pairs
        const bool pair = item.is_array() && item.as_array().size() >= 2;
        set_var(node.var_name, pair ? item.as_array()[0] : item);
        set_var(node.var_name2, pair ? item.as_array()[1] : Value());
    }

    // loop.* as Jinja's LoopContext; previtem / nextitem unset at the edges (Jinja2 undefined).
    static Value loop_object(const Value::Array& items, int64_t i) {
        const auto len = static_cast<int64_t>(items.size());
        auto loop = Value::make_object();
        loop.set("index", Value(i + 1));
        loop.set("index0", Value(i));
        loop.set("revindex", Value(len - i));
        loop.set("revindex0", Value(len - i - 1));
        loop.set("first", Value(i == 0));
        loop.set("last", Value(i == len - 1));
        loop.set("length", Value(len));
        loop.set("depth", Value(int64_t{1}));
        loop.set("depth0", Value(int64_t{0}));
        if (i > 0)
            loop.set("previtem", items[static_cast<size_t>(i - 1)]);
        if (i + 1 < len)
            loop.set("nextitem", items[static_cast<size_t>(i + 1)]);
        return loop;
    }

    void render_for(const ForNode& node, std::string& out) {
        Value::Array items = loop_items(node, eval(*node.iterable));
        push_scope();
        if (node.filter) {
            // `for x in xs if cond`: filtered first, so loop.index / length / last count kept items.
            Value::Array kept;
            for (const auto& item : items) {
                count_iteration();
                bind_loop_vars(node, item);
                if (eval(*node.filter).truthy())
                    kept.push_back(item);
            }
            items = std::move(kept);
        }
        if (items.empty()) {
            for (auto& n : node.else_body)
                render_node(*n, out);
            pop_scope();
            return;
        }
        for (int64_t i = 0; i < static_cast<int64_t>(items.size()); i++) {
            count_iteration();
            set_var("loop", loop_object(items, i));
            bind_loop_vars(node, items[static_cast<size_t>(i)]);
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

    // `x is [not] defined / undefined` look at the variable, not its value: a var set to none is defined.
    std::optional<bool> definedness_test(const BinOpExpr& bin, const std::string& test) {
        if (test != "defined" && test != "undefined")
            return std::nullopt;
        bool defined = false;
        if (auto* var = dynamic_cast<const VariableExpr*>(bin.left.get()))
            defined = is_defined(var->name);
        else
            defined = !eval(*bin.left).is_none();
        return test == "defined" ? defined : !defined;
    }

    Value eval_is_test(const BinOpExpr& bin) {
        const bool negated = bin.op.starts_with("is not ");
        const std::string test = bin.op.substr(negated ? 7 : 3);
        std::optional<bool> r = definedness_test(bin, test);
        if (!r) {
            const Value left = eval(*bin.left);
            std::vector<Value> args;
            if (bin.right)
                args.push_back(eval(*bin.right));
            r = run_test(test, left, args);
        }
        return Value(negated ? !*r : *r);
    }

    static Value arith(const std::string& op, const Value& left, const Value& right) {
        const bool ints = left.is_int() && right.is_int();
        if (op == "-")
            return ints ? Value(left.as_int() - right.as_int()) : Value(left.to_number() - right.to_number());
        if (op == "/") {
            const double d = right.to_number();
            return d == 0.0 ? Value(0.0) : Value(left.to_number() / d);
        }
        if (op == "//") {
            if (ints) {
                const int64_t r = right.as_int();
                if (r == 0)
                    return Value(int64_t{0});
                const int64_t q = left.as_int() / r;
                return Value((left.as_int() % r != 0 && ((left.as_int() < 0) != (r < 0))) ? q - 1 : q);
            }
            const double d = right.to_number();
            return d == 0.0 ? Value(0.0) : Value(std::floor(left.to_number() / d));
        }
        if (op == "**") {
            if (ints && right.as_int() >= 0) {
                int64_t r = 1;
                for (int64_t k = 0; k < right.as_int() && k < 64; k++)
                    r *= left.as_int();
                return Value(r);
            }
            return Value(std::pow(left.to_number(), right.to_number()));
        }
        if (op == "%") {
            if (ints) {
                const int64_t r = right.as_int();
                return r == 0 ? Value(int64_t{0}) : Value(((left.as_int() % r) + r) % r);
            }
            const double d = right.to_number();
            return d == 0.0 ? Value(0.0) : Value(left.to_number() - (d * std::floor(left.to_number() / d)));
        }
        return Value();
    }

    // 'ab' * 3, [1] * 2, 3 * 4
    static Value multiply(const Value& left, const Value& right) {
        auto repeat = [](const Value& seq, int64_t n) {
            if (seq.is_string()) {
                std::string s;
                for (int64_t k = 0; k < n; k++)
                    s += seq.as_string();
                return Value(std::move(s));
            }
            Value::Array a;
            for (int64_t k = 0; k < n; k++)
                a.insert(a.end(), seq.as_array().begin(), seq.as_array().end());
            return Value(std::move(a));
        };
        if ((left.is_string() || left.is_array()) && right.is_int())
            return repeat(left, right.as_int());
        if ((right.is_string() || right.is_array()) && left.is_int())
            return repeat(right, left.as_int());
        if (left.is_int() && right.is_int())
            return Value(left.as_int() * right.as_int());
        return Value(left.to_number() * right.to_number());
    }

    static Value compare(const std::string& op, const Value& left, const Value& right) {
        if (op == "==")
            return Value(left == right);
        if (op == "!=")
            return Value(left != right);
        if (op == "<")
            return Value(left < right);
        if (op == ">")
            return Value(left > right);
        if (op == "<=")
            return Value(left <= right);
        if (op == ">=")
            return Value(left >= right);
        if (op == "in")
            return Value(right.contains(left));
        if (op == "not in")
            return Value(!right.contains(left));
        return Value();
    }

    Value eval_binop(const BinOpExpr& bin) {
        // Short-circuit for and/or
        if (bin.op == "and" || bin.op == "or") {
            Value left = eval(*bin.left);
            if (left.truthy() == (bin.op == "or"))
                return left;
            return eval(*bin.right);
        }
        if (bin.op.starts_with("is "))
            return eval_is_test(bin);

        const Value left = eval(*bin.left);
        const Value right = eval(*bin.right);
        if (bin.op == "+")
            return left + right;
        if (bin.op == "~")
            return Value(left.to_string() + right.to_string());
        if (bin.op == "*")
            return multiply(left, right);
        if (bin.op == "-" || bin.op == "/" || bin.op == "//" || bin.op == "**" || bin.op == "%")
            return arith(bin.op, left, right);
        return compare(bin.op, left, right);
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

    void eval_args(const std::vector<std::unique_ptr<Expr>>& in,
                   const std::vector<std::pair<std::string, std::unique_ptr<Expr>>>& kw_in,
                   std::vector<Value>& args, Kwargs& kwargs) {
        for (const auto& a : in)
            args.push_back(eval(*a));
        for (const auto& [k, e] : kw_in)
            kwargs.emplace_back(k, eval(*e));
    }

    Value eval_filter(const FilterExpr& f) {
        const Value val = eval(*f.value);
        std::vector<Value> args;
        Kwargs kwargs;
        eval_args(f.args, f.kwargs, args, kwargs);
        return apply_filter(f.name, val, args, kwargs);
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

    // Pre-sandbox leniency kept: list.append/insert return a new list, dict.update mutates. HF renders
    // in an ImmutableSandboxedEnvironment where these raise, so no upstream template calls them.
    static std::optional<Value> legacy_mutator(const Value& obj, const std::string& name,
                                               const std::vector<Value>& a) {
        if (obj.is_array() && name == "append") {
            Value::Array arr = obj.as_array();
            if (!a.empty())
                arr.push_back(a[0]);
            return Value(std::move(arr));
        }
        if (obj.is_array() && name == "insert") {
            Value::Array arr = obj.as_array();
            if (a.size() >= 2 && a[0].is_int()) {
                const int64_t idx = std::clamp<int64_t>(a[0].as_int(), 0, static_cast<int64_t>(arr.size()));
                arr.insert(arr.begin() + idx, a[1]);
            }
            return Value(std::move(arr));
        }
        if (obj.is_object() && obj.as_object() && name == "update") {
            if (!a.empty() && a[0].is_object() && a[0].as_object())
                for (const auto& [k, v] : *a[0].as_object())
                    (*obj.as_object())[k] = v;
            return obj;
        }
        return std::nullopt;
    }

    Value eval_method(const MethodExpr& m) {
        const Value obj = eval(*m.object);
        std::vector<Value> args;
        Kwargs kwargs;
        eval_args(m.args, m.kwargs, args, kwargs);
        // loop.cycle('odd', 'even')
        if (m.method == "cycle" && obj.is_object() && obj.as_object() && obj.as_object()->contains("index0"))
            return args.empty()
                       ? Value()
                       : args[static_cast<size_t>(obj.get(std::string("index0")).as_int()) % args.size()];
        if (auto r = call_method(obj, m.method, args, kwargs))
            return *r;
        if (auto r = legacy_mutator(obj, m.method, args))
            return *r;
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
        error_ = unknown_tag.empty() ? "Parse error"
                                     : "unsupported or unbalanced tag: {% " + unknown_tag + " %}";
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
