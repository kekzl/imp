// Jinja2 builtins: filters, tests, methods, tojson (jinja_builtins.h).

#include "model/jinja_builtins.h"
#include "core/logging.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <unordered_map>

namespace imp::jinja::detail {

// Positional arg `i`, else keyword `name`, else `def`.
Value param(const Args& args, const Kwargs& kw, size_t i, std::string_view name, Value def) {
    if (i < args.size())
        return args[i];
    for (const auto& [k, v] : kw)
        if (k == name)
            return v;
    return def;
}

std::string lower_ascii(std::string s) {
    for (auto& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string upper_ascii(std::string s) {
    for (auto& c : s)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

bool is_ws(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

// str.strip family: chars empty = whitespace.
std::string strip_chars(const std::string& s, const std::string& chars, bool left, bool right) {
    auto drop = [&](char c) { return chars.empty() ? is_ws(c) : chars.contains(c); };
    size_t b = 0, e = s.size();
    while (left && b < e && drop(s[b]))
        b++;
    while (right && e > b && drop(s[e - 1]))
        e--;
    return s.substr(b, e - b);
}

std::string replace_n(std::string s, const std::string& from, const std::string& to, int64_t count) {
    if (from.empty())
        return s;
    size_t pos = 0;
    for (int64_t n = 0; count < 0 || n < count; n++) {
        pos = s.find(from, pos);
        if (pos == std::string::npos)
            break;
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Python str.title / str.capitalize on ASCII.
std::string title_case(std::string s) {
    bool prev_alpha = false;
    for (auto& c : s) {
        const bool alpha = std::isalpha(static_cast<unsigned char>(c)) != 0;
        c = static_cast<char>(prev_alpha ? std::tolower(static_cast<unsigned char>(c))
                                         : std::toupper(static_cast<unsigned char>(c)));
        prev_alpha = alpha;
    }
    return s;
}

std::string capitalize(std::string s) {
    s = lower_ascii(std::move(s));
    if (!s.empty())
        s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    return s;
}

// Attribute path "a.b" or index on an item, as Jinja's make_attrgetter.
static Value attr_of(const Value& item, const std::string& path) {
    Value cur = item;
    size_t start = 0;
    while (start <= path.size()) {
        const size_t dot = path.find('.', start);
        const std::string part = path.substr(start,
                                             dot == std::string::npos ? std::string::npos : dot - start);
        int64_t idx = 0;
        const auto [p, ec] = std::from_chars(part.data(), part.data() + part.size(), idx);
        cur = (ec == std::errc() && p == part.data() + part.size() && cur.is_array()) ? cur.get(Value(idx))
                                                                                      : cur.get(part);
        if (dot == std::string::npos)
            break;
        start = dot + 1;
    }
    return cur;
}

static Value sort_key(const Value& v, bool case_sensitive) {
    return (!case_sensitive && v.is_string()) ? Value(lower_ascii(v.as_string())) : v;
}

int64_t to_int(const Value& v, int64_t def) {
    if (v.is_int())
        return v.as_int();
    if (v.is_double())
        return static_cast<int64_t>(v.as_double());
    if (v.is_bool())
        return v.as_bool() ? 1 : 0;
    if (v.is_string()) {
        const std::string s = strip_chars(v.as_string(), "", true, true);
        int64_t out = 0;
        const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
        if (ec == std::errc() && p == s.data() + s.size())
            return out;
        char* end = nullptr;
        const double d = std::strtod(s.c_str(), &end);
        if (!s.empty() && end == s.c_str() + s.size())
            return static_cast<int64_t>(d);
    }
    return def;
}

Value pairs_of(const Value& v) {
    Value::Array arr;
    if (v.is_object() && v.as_object())
        for (const auto& [k, val] : *v.as_object())
            arr.push_back(Value::array({Value(k), val}));
    return Value(std::move(arr));
}

namespace {

// ---- filters --------------------------------------------------------------

using FilterFn = Value (*)(const Value&, const Args&, const Kwargs&);

Value f_safe(const Value& v, const Args&, const Kwargs&) { return v; }

Value f_trim(const Value& v, const Args& a, const Kwargs& kw) {
    if (!v.is_string())
        return v;
    const Value chars = param(a, kw, 0, "chars");
    return Value(strip_chars(v.as_string(), chars.is_string() ? chars.as_string() : "", true, true));
}

Value f_length(const Value& v, const Args&, const Kwargs&) { return Value(v.length()); }

// default(value='', boolean=false): Jinja treats none/undefined as missing; boolean=true also falsy.
Value f_default(const Value& v, const Args& a, const Kwargs& kw) {
    const bool boolean = param(a, kw, 1, "boolean", Value(false)).truthy();
    if (v.is_none() || (boolean && !v.truthy()))
        return param(a, kw, 0, "default_value", Value(std::string()));
    return v;
}

Value f_first(const Value& v, const Args&, const Kwargs&) {
    if (v.is_array())
        return v.as_array().empty() ? Value() : v.as_array().front();
    if (v.is_string())
        return v.as_string().empty() ? Value() : Value(v.as_string().substr(0, 1));
    return Value();
}

Value f_last(const Value& v, const Args&, const Kwargs&) {
    if (v.is_array())
        return v.as_array().empty() ? Value() : v.as_array().back();
    if (v.is_string())
        return v.as_string().empty() ? Value() : Value(v.as_string().substr(v.as_string().size() - 1));
    return Value();
}

Value f_upper(const Value& v, const Args&, const Kwargs&) {
    return v.is_string() ? Value(upper_ascii(v.as_string())) : v;
}

Value f_lower(const Value& v, const Args&, const Kwargs&) {
    return v.is_string() ? Value(lower_ascii(v.as_string())) : v;
}

Value f_join(const Value& v, const Args& a, const Kwargs& kw) {
    if (!v.is_array())
        return v.is_string() ? v : Value(std::string());
    const std::string sep = param(a, kw, 0, "d", Value(std::string())).to_string();
    const Value attr = param(a, kw, 1, "attribute");
    std::string out;
    const auto& arr = v.as_array();
    for (size_t i = 0; i < arr.size(); i++) {
        if (i > 0)
            out += sep;
        out += (attr.is_none() ? arr[i] : attr_of(arr[i], attr.to_string())).to_string();
    }
    return Value(std::move(out));
}

Value f_string(const Value& v, const Args&, const Kwargs&) { return Value(v.to_string()); }

Value f_int(const Value& v, const Args& a, const Kwargs& kw) {
    return Value(to_int(v, param(a, kw, 0, "default", Value(int64_t{0})).is_int()
                               ? param(a, kw, 0, "default", Value(int64_t{0})).as_int()
                               : 0));
}

Value f_float(const Value& v, const Args& a, const Kwargs& kw) {
    if (v.is_string()) {
        char* end = nullptr;
        const std::string s = strip_chars(v.as_string(), "", true, true);
        const double d = std::strtod(s.c_str(), &end);
        if (!s.empty() && end == s.c_str() + s.size())
            return Value(d);
        return Value(param(a, kw, 0, "default", Value(0.0)).to_number());
    }
    return Value(v.to_number());
}

Value f_list(const Value& v, const Args&, const Kwargs&) {
    if (v.is_array())
        return v;
    Value::Array arr;
    if (v.is_string()) {
        for (char c : v.as_string())
            arr.emplace_back(std::string(1, c));
    } else if (v.is_object() && v.as_object()) {
        for (const auto& [k, val] : *v.as_object())
            arr.emplace_back(k);
    }
    return Value(std::move(arr));
}

Value f_items(const Value& v, const Args&, const Kwargs&) { return pairs_of(v); }

Value f_replace(const Value& v, const Args& a, const Kwargs& kw) {
    if (!v.is_string() || a.size() < 2)
        return v;
    const Value count = param(a, kw, 2, "count");
    return Value(
        replace_n(v.as_string(), a[0].to_string(), a[1].to_string(), count.is_int() ? count.as_int() : -1));
}

Value f_title(const Value& v, const Args&, const Kwargs&) {
    return v.is_string() ? Value(title_case(v.as_string())) : v;
}

Value f_capitalize(const Value& v, const Args&, const Kwargs&) {
    return v.is_string() ? Value(capitalize(v.as_string())) : v;
}

Value f_reverse(const Value& v, const Args&, const Kwargs&) {
    if (v.is_array()) {
        Value::Array arr(v.as_array().rbegin(), v.as_array().rend());
        return Value(std::move(arr));
    }
    if (v.is_string())
        return Value(std::string(v.as_string().rbegin(), v.as_string().rend()));
    return v;
}

// tojson(indent=None, separators=None, sort_keys=False, ensure_ascii=False), as transformers.
Value f_tojson(const Value& v, const Args& a, const Kwargs& kw) {
    JsonOpts o;
    const Value indent = param(a, kw, 0, "indent");
    if (indent.is_int()) {
        o.indent = static_cast<int>(indent.as_int());
        o.item_sep = ",";
    }
    const Value seps = param(a, kw, 1, "separators");
    if (seps.is_array() && seps.as_array().size() == 2) {
        o.item_sep = seps.as_array()[0].to_string();
        o.key_sep = seps.as_array()[1].to_string();
    }
    o.sort_keys = param(a, kw, 2, "sort_keys", Value(false)).truthy();
    o.ensure_ascii = param(a, kw, 3, "ensure_ascii", Value(false)).truthy();
    return Value(to_json(v, o));
}

// select/reject(test, args...) and selectattr/rejectattr(attr, test, args...).
Value select_impl(const Value& v, const Args& a, bool by_attr, bool keep) {
    if (!v.is_array())
        return v;
    const size_t t0 = by_attr ? 1 : 0;
    if (by_attr && (a.empty() || !a[0].is_string()))
        return v;
    const std::string test = a.size() > t0 && a[t0].is_string() ? a[t0].as_string() : "";
    const Args targs(a.begin() + static_cast<std::ptrdiff_t>(std::min(a.size(), t0 + 1)), a.end());
    Value::Array out;
    for (const auto& item : v.as_array()) {
        const Value probe = by_attr ? attr_of(item, a[0].as_string()) : item;
        const bool pass = test.empty() ? probe.truthy() : run_test(test, probe, targs);
        if (pass == keep)
            out.push_back(item);
    }
    return Value(std::move(out));
}

Value f_select(const Value& v, const Args& a, const Kwargs&) { return select_impl(v, a, false, true); }
Value f_reject(const Value& v, const Args& a, const Kwargs&) { return select_impl(v, a, false, false); }
Value f_selectattr(const Value& v, const Args& a, const Kwargs&) { return select_impl(v, a, true, true); }
Value f_rejectattr(const Value& v, const Args& a, const Kwargs&) { return select_impl(v, a, true, false); }

// map(attribute='x', default=None) or map('filter', args...).
Value f_map(const Value& v, const Args& a, const Kwargs& kw) {
    if (!v.is_array())
        return Value(Value::Array{});
    Value::Array out;
    const Value attr = param({}, kw, 0, "attribute");
    for (const auto& item : v.as_array()) {
        if (!attr.is_none()) {
            Value r = attr_of(item, attr.to_string());
            out.push_back(r.is_none() ? param({}, kw, 0, "default") : r);
        } else if (!a.empty() && a[0].is_string()) {
            out.push_back(apply_filter(a[0].as_string(), item, Args(a.begin() + 1, a.end()), {}));
        } else {
            out.push_back(item);
        }
    }
    return Value(std::move(out));
}

Value::Array keyed(const Value& v, const Kwargs& kw) {
    Value::Array arr = v.is_array() ? v.as_array() : Value::Array{};
    const Value attr = param({}, kw, 0, "attribute");
    if (!attr.is_none())
        for (auto& x : arr)
            x = attr_of(x, attr.to_string());
    return arr;
}

Value f_sort(const Value& v, const Args& a, const Kwargs& kw) {
    if (!v.is_array())
        return v;
    const bool reverse = param(a, kw, 0, "reverse", Value(false)).truthy();
    const bool cs = param(a, kw, 1, "case_sensitive", Value(false)).truthy();
    const Value attr = param(a, kw, 2, "attribute");
    Value::Array arr = v.as_array();
    auto key = [&](const Value& x) {
        return sort_key(attr.is_none() ? x : attr_of(x, attr.to_string()), cs);
    };
    std::ranges::stable_sort(arr, [&](const Value& x, const Value& y) {
        return reverse ? key(y) < key(x) : key(x) < key(y);
    });
    return Value(std::move(arr));
}

Value f_unique(const Value& v, const Args& a, const Kwargs& kw) {
    if (!v.is_array())
        return v;
    const bool cs = param(a, kw, 0, "case_sensitive", Value(false)).truthy();
    const Value attr = param(a, kw, 1, "attribute");
    Value::Array out, seen;
    for (const auto& x : v.as_array()) {
        const Value k = sort_key(attr.is_none() ? x : attr_of(x, attr.to_string()), cs);
        if (std::ranges::find(seen, k) == seen.end()) {
            seen.push_back(k);
            out.push_back(x);
        }
    }
    return Value(std::move(out));
}

Value extreme(const Value& v, const Args& a, const Kwargs& kw, bool want_max) {
    if (!v.is_array() || v.as_array().empty())
        return Value();
    const bool cs = param(a, kw, 0, "case_sensitive", Value(false)).truthy();
    const Value attr = param(a, kw, 1, "attribute");
    const Value* best = &v.as_array().front();
    auto key = [&](const Value& x) {
        return sort_key(attr.is_none() ? x : attr_of(x, attr.to_string()), cs);
    };
    for (const auto& x : v.as_array())
        if (want_max ? key(*best) < key(x) : key(x) < key(*best))
            best = &x;
    return *best;
}

Value f_max(const Value& v, const Args& a, const Kwargs& kw) { return extreme(v, a, kw, true); }
Value f_min(const Value& v, const Args& a, const Kwargs& kw) { return extreme(v, a, kw, false); }

Value f_sum(const Value& v, const Args& a, const Kwargs& kw) {
    Value total = param({}, kw, 0, "start", Value(int64_t{0}));
    const Value attr = param(a, kw, 0, "attribute");
    for (const auto& x : keyed(v, attr.is_none() ? Kwargs{} : Kwargs{{"attribute", attr}}))
        total = total + x;
    return total;
}

Value f_abs(const Value& v, const Args&, const Kwargs&) {
    if (v.is_int())
        return Value(v.as_int() < 0 ? -v.as_int() : v.as_int());
    return v.is_double() ? Value(std::fabs(v.as_double())) : v;
}

// round(precision=0, method='common'|'ceil'|'floor') -> float, as Jinja.
Value f_round(const Value& v, const Args& a, const Kwargs& kw) {
    const double p = std::pow(10.0, static_cast<double>(
                                        to_int(param(a, kw, 0, "precision", Value(int64_t{0})), 0)));
    const std::string method = param(a, kw, 1, "method", Value(std::string("common"))).to_string();
    const double x = v.to_number() * p;
    const double r = method == "ceil" ? std::ceil(x) : method == "floor" ? std::floor(x) : std::round(x);
    return Value(r / p);
}

Value f_batch(const Value& v, const Args& a, const Kwargs& kw) {
    const int64_t n = to_int(param(a, kw, 0, "linecount"), 1);
    if (!v.is_array() || n <= 0)
        return Value(Value::Array{});
    const Value fill = param(a, kw, 1, "fill_with");
    Value::Array out, cur;
    for (const auto& x : v.as_array()) {
        cur.push_back(x);
        if (static_cast<int64_t>(cur.size()) == n)
            out.emplace_back(std::exchange(cur, {}));
    }
    if (!cur.empty()) {
        while (!fill.is_none() && static_cast<int64_t>(cur.size()) < n)
            cur.push_back(fill);
        out.emplace_back(std::move(cur));
    }
    return Value(std::move(out));
}

// groupby(attribute): [{grouper, list}] sorted by the key, as Jinja's _GroupTuple.
Value f_groupby(const Value& v, const Args& a, const Kwargs& kw) {
    const Value attr = param(a, kw, 0, "attribute");
    if (!v.is_array() || attr.is_none())
        return Value(Value::Array{});
    Value sorted = f_sort(v, {}, Kwargs{{"attribute", attr}, {"case_sensitive", Value(true)}});
    Value::Array out;
    for (const auto& x : sorted.as_array()) {
        const Value k = attr_of(x, attr.to_string());
        if (out.empty() || out.back().get(std::string("grouper")) != k)
            out.push_back(Value::object({{"grouper", k}, {"list", Value(Value::Array{})}}));
        Value::Array lst = out.back().get(std::string("list")).as_array();
        lst.push_back(x);
        out.back().set("list", Value(std::move(lst)));
    }
    return Value(std::move(out));
}

Value f_dictsort(const Value& v, const Args& a, const Kwargs& kw) {
    const bool cs = param(a, kw, 0, "case_sensitive", Value(false)).truthy();
    const bool by_value = param(a, kw, 1, "by", Value(std::string("key"))).to_string() == "value";
    const bool reverse = param(a, kw, 2, "reverse", Value(false)).truthy();
    Value::Array arr = pairs_of(v).as_array();
    std::ranges::stable_sort(arr, [&](const Value& x, const Value& y) {
        const Value kx = sort_key(x.as_array()[by_value ? 1 : 0], cs),
                    ky = sort_key(y.as_array()[by_value ? 1 : 0], cs);
        return reverse ? ky < kx : kx < ky;
    });
    return Value(std::move(arr));
}

// indent(width=4, first=False, blank=False).
Value f_indent(const Value& v, const Args& a, const Kwargs& kw) {
    const Value w = param(a, kw, 0, "width", Value(int64_t{4}));
    const std::string pad = w.is_string() ? w.as_string()
                                          : std::string(static_cast<size_t>(to_int(w, 4)), ' ');
    const bool first = param(a, kw, 1, "first", Value(false)).truthy();
    const bool blank = param(a, kw, 2, "blank", Value(false)).truthy();
    const std::string s = v.to_string();
    std::string out;
    size_t start = 0;
    for (bool is_first = true; start <= s.size(); is_first = false) {
        const size_t nl = s.find('\n', start);
        const std::string line = s.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
        if ((!is_first || first) && (blank || !line.empty()))
            out += pad;
        out += line;
        if (nl == std::string::npos)
            break;
        out += '\n';
        start = nl + 1;
    }
    return Value(std::move(out));
}

Value f_wordcount(const Value& v, const Args&, const Kwargs&) {
    int64_t n = 0;
    bool in_word = false;
    for (char c : v.to_string()) {
        const bool w = std::isalnum(static_cast<unsigned char>(c)) || c == '_';
        n += (w && !in_word) ? 1 : 0;
        in_word = w;
    }
    return Value(n);
}

// Python str.center: the extra pad char goes right when the slack is odd and the string length even.
Value f_center(const Value& v, const Args& a, const Kwargs& kw) {
    const std::string s = v.to_string();
    const int64_t width = to_int(param(a, kw, 0, "width", Value(int64_t{80})), 80);
    if (width <= static_cast<int64_t>(s.size()))
        return Value(s);
    const size_t marg = static_cast<size_t>(width) - s.size();
    const size_t left = (marg / 2) + (marg & static_cast<size_t>(width) & 1);
    return Value(std::string(left, ' ') + s + std::string(marg - left, ' '));
}

// printf-style "%s %d %i %f %r" with positional args.
Value f_format(const Value& v, const Args& a, const Kwargs&) {
    const std::string fmt = v.to_string();
    std::string out;
    size_t ai = 0;
    for (size_t i = 0; i < fmt.size(); i++) {
        if (fmt[i] != '%' || i + 1 >= fmt.size()) {
            out += fmt[i];
            continue;
        }
        const char spec = fmt[++i];
        if (spec == '%') {
            out += '%';
            continue;
        }
        const Value arg = ai < a.size() ? a[ai++] : Value();
        if (spec == 'd' || spec == 'i')
            out += std::to_string(to_int(arg, 0));
        else if (spec == 'r' && arg.is_string())
            out += "'" + arg.as_string() + "'";
        else if (spec == 'f') {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%f", arg.to_number());
            out += buf;
        } else
            out += arg.to_string();
    }
    return Value(std::move(out));
}

Value f_escape(const Value& v, const Args&, const Kwargs&) {
    std::string out;
    for (char c : v.to_string()) {
        switch (c) {
            case '&':
                out += "&amp;";
                break;
            case '<':
                out += "&lt;";
                break;
            case '>':
                out += "&gt;";
                break;
            case '"':
                out += "&#34;";
                break;
            case '\'':
                out += "&#39;";
                break;
            default:
                out += c;
        }
    }
    return Value(std::move(out));
}

Value f_attr(const Value& v, const Args& a, const Kwargs&) {
    return a.empty() ? Value() : v.get(a[0].to_string());
}

const std::unordered_map<std::string_view, FilterFn>& filters() {
    static const std::unordered_map<std::string_view, FilterFn> t = {
        {"safe", f_safe},
        {"trim", f_trim},
        {"length", f_length},
        {"count", f_length},
        {"default", f_default},
        {"d", f_default},
        {"first", f_first},
        {"last", f_last},
        {"upper", f_upper},
        {"lower", f_lower},
        {"join", f_join},
        {"string", f_string},
        {"int", f_int},
        {"float", f_float},
        {"list", f_list},
        {"items", f_items},
        {"replace", f_replace},
        {"title", f_title},
        {"capitalize", f_capitalize},
        {"reverse", f_reverse},
        {"tojson", f_tojson},
        {"select", f_select},
        {"reject", f_reject},
        {"selectattr", f_selectattr},
        {"rejectattr", f_rejectattr},
        {"map", f_map},
        {"sort", f_sort},
        {"unique", f_unique},
        {"max", f_max},
        {"min", f_min},
        {"sum", f_sum},
        {"abs", f_abs},
        {"round", f_round},
        {"batch", f_batch},
        {"groupby", f_groupby},
        {"dictsort", f_dictsort},
        {"indent", f_indent},
        {"wordcount", f_wordcount},
        {"center", f_center},
        {"format", f_format},
        {"escape", f_escape},
        {"e", f_escape},
        {"forceescape", f_escape},
        {"attr", f_attr},
    };
    return t;
}

// ---- tests ----------------------------------------------------------------

using TestFn = bool (*)(const Value&, const Args&);

Value arg0(const Args& a) { return a.empty() ? Value() : a[0]; }

bool t_none(const Value& v, const Args&) { return v.is_none(); }
bool t_defined(const Value& v, const Args&) { return !v.is_none(); }
bool t_boolean(const Value& v, const Args&) { return v.is_bool(); }
bool t_true(const Value& v, const Args&) { return v.is_bool() && v.as_bool(); }
bool t_false(const Value& v, const Args&) { return v.is_bool() && !v.as_bool(); }
bool t_integer(const Value& v, const Args&) { return v.is_int(); }
bool t_float(const Value& v, const Args&) { return v.is_double(); }
bool t_number(const Value& v, const Args&) { return v.is_number() || v.is_bool(); }
bool t_string(const Value& v, const Args&) { return v.is_string(); }
bool t_mapping(const Value& v, const Args&) { return v.is_object() && v.as_object(); }
bool t_iterable(const Value& v, const Args&) { return v.is_array() || v.is_string() || t_mapping(v, {}); }
bool t_sequence(const Value& v, const Args&) { return v.is_array() || v.is_string() || t_mapping(v, {}); }
bool t_callable(const Value&, const Args&) { return false; }
bool t_escaped(const Value&, const Args&) { return false; }
bool t_lower(const Value& v, const Args&) {
    return v.is_string() && lower_ascii(v.as_string()) == v.as_string();
}
bool t_upper(const Value& v, const Args&) {
    return v.is_string() && upper_ascii(v.as_string()) == v.as_string();
}
bool t_even(const Value& v, const Args&) { return v.is_int() && v.as_int() % 2 == 0; }
bool t_odd(const Value& v, const Args&) { return v.is_int() && v.as_int() % 2 != 0; }
bool t_divisibleby(const Value& v, const Args& a) {
    const int64_t n = to_int(arg0(a), 0);
    return n != 0 && v.is_int() && v.as_int() % n == 0;
}
bool t_eq(const Value& v, const Args& a) { return v == arg0(a); }
bool t_ne(const Value& v, const Args& a) { return v != arg0(a); }
bool t_lt(const Value& v, const Args& a) { return v < arg0(a); }
bool t_le(const Value& v, const Args& a) { return v <= arg0(a); }
bool t_gt(const Value& v, const Args& a) { return v > arg0(a); }
bool t_ge(const Value& v, const Args& a) { return v >= arg0(a); }
bool t_in(const Value& v, const Args& a) { return arg0(a).contains(v); }
bool t_sameas(const Value& v, const Args& a) {
    const Value o = arg0(a);
    return (v.is_none() && o.is_none()) || (v.is_bool() && o.is_bool() && v.as_bool() == o.as_bool()) ||
           (v.is_object() && o.is_object() && v.as_object() == o.as_object());
}

const std::unordered_map<std::string_view, TestFn>& tests() {
    static const std::unordered_map<std::string_view, TestFn> t = {
        {"none", t_none},
        {"defined", t_defined},
        {"boolean", t_boolean},
        {"true", t_true},
        {"false", t_false},
        {"integer", t_integer},
        {"float", t_float},
        {"number", t_number},
        {"string", t_string},
        {"mapping", t_mapping},
        {"iterable", t_iterable},
        {"sequence", t_sequence},
        {"callable", t_callable},
        {"escaped", t_escaped},
        {"lower", t_lower},
        {"upper", t_upper},
        {"even", t_even},
        {"odd", t_odd},
        {"divisibleby", t_divisibleby},
        {"eq", t_eq},
        {"equalto", t_eq},
        {"==", t_eq},
        {"ne", t_ne},
        {"!=", t_ne},
        {"lt", t_lt},
        {"lessthan", t_lt},
        {"<", t_lt},
        {"le", t_le},
        {"<=", t_le},
        {"gt", t_gt},
        {"greaterthan", t_gt},
        {">", t_gt},
        {"ge", t_ge},
        {">=", t_ge},
        {"in", t_in},
        {"sameas", t_sameas},
    };
    return t;
}

}  // namespace

Value apply_filter(const std::string& name, const Value& v, const std::vector<Value>& args,
                   const Kwargs& kwargs) {
    const auto& t = filters();
    if (auto it = t.find(name); it != t.end())
        return it->second(v, args, kwargs);
    IMP_LOG_WARN("jinja: unknown filter '%s'", name.c_str());
    return v;
}

bool run_test(const std::string& name, const Value& v, const std::vector<Value>& args) {
    const auto& t = tests();
    if (auto it = t.find(name); it != t.end())
        return it->second(v, args);
    IMP_LOG_WARN("jinja: unknown test '%s'", name.c_str());
    return false;
}

}  // namespace imp::jinja::detail
