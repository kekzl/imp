// Jinja2 builtins on evaluated values: filters, tests, str/list/dict methods, tojson.
// Semantics follow transformers' ImmutableSandboxedEnvironment (the env vLLM renders with).
#pragma once

#include "model/jinja.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace imp::jinja::detail {

using Kwargs = std::vector<std::pair<std::string, Value>>;

// json.dumps options; transformers' tojson defaults to ensure_ascii=False.
struct JsonOpts {
    int indent = -1;  // < 0: one line
    std::string item_sep = ", ";
    std::string key_sep = ": ";
    bool sort_keys = false;
    bool ensure_ascii = false;
};

std::string to_json(const Value& v, const JsonOpts& opts);
// Python repr of a float: 3.0, 0.1, 1e+20, 1e-05.
std::string py_float_repr(double v);

// Filter `name` on `v`; unknown filters warn and return `v`.
Value apply_filter(const std::string& name, const Value& v, const std::vector<Value>& args,
                   const Kwargs& kwargs);
// Test `name` (`x is name args...`); defined/undefined on variables are the evaluator's.
[[nodiscard]] bool run_test(const std::string& name, const Value& v, const std::vector<Value>& args);
// Method call obj.name(args); nullopt = no such method.
std::optional<Value> call_method(const Value& obj, const std::string& name, const std::vector<Value>& args,
                                 const Kwargs& kwargs);

// ---- helpers shared by jinja_builtins.cpp and jinja_methods.cpp ----
using Args = std::vector<Value>;
// Positional arg `i`, else keyword `name`, else `def`.
Value param(const Args& args, const Kwargs& kw, size_t i, std::string_view name, Value def = Value());
std::string lower_ascii(std::string s);
std::string upper_ascii(std::string s);
[[nodiscard]] bool is_ws(char c);
// str.strip family: chars empty = whitespace.
std::string strip_chars(const std::string& s, const std::string& chars, bool left, bool right);
// count < 0: all occurrences.
std::string replace_n(std::string s, const std::string& from, const std::string& to, int64_t count);
std::string title_case(std::string s);
std::string capitalize(std::string s);
int64_t to_int(const Value& v, int64_t def);
// dict -> [[key, value], ...] in insertion order.
Value pairs_of(const Value& v);

}  // namespace imp::jinja::detail
