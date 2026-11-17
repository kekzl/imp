// Jinja2 str / list / dict methods (jinja_builtins.h).

#include "model/jinja_builtins.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace imp::jinja::detail {

namespace {

Value arg0(const Args& a) { return a.empty() ? Value() : a[0]; }

// ---- methods --------------------------------------------------------------

using MethodFn = Value (*)(const Value&, const Args&, const Kwargs&);

std::string sarg(const Args& a, size_t i) { return i < a.size() ? a[i].to_string() : std::string(); }

Value m_strip(const Value& v, const Args& a, const Kwargs&) {
    return Value(strip_chars(v.as_string(), a.empty() || a[0].is_none() ? "" : sarg(a, 0), true, true));
}
Value m_lstrip(const Value& v, const Args& a, const Kwargs&) {
    return Value(strip_chars(v.as_string(), a.empty() || a[0].is_none() ? "" : sarg(a, 0), true, false));
}
Value m_rstrip(const Value& v, const Args& a, const Kwargs&) {
    return Value(strip_chars(v.as_string(), a.empty() || a[0].is_none() ? "" : sarg(a, 0), false, true));
}

// startswith/endswith accept a str or a tuple (list) of str.
Value affix(const Value& v, const Args& a, bool prefix) {
    if (a.empty())
        return Value(false);
    const std::string& s = v.as_string();
    const Args cands = a[0].is_array() ? a[0].as_array() : Args{a[0]};
    for (const auto& c : cands) {
        const std::string x = c.to_string();
        if (prefix ? s.starts_with(x) : s.ends_with(x))
            return Value(true);
    }
    return Value(false);
}
Value m_startswith(const Value& v, const Args& a, const Kwargs&) { return affix(v, a, true); }
Value m_endswith(const Value& v, const Args& a, const Kwargs&) { return affix(v, a, false); }
Value m_upper(const Value& v, const Args&, const Kwargs&) { return Value(upper_ascii(v.as_string())); }
Value m_lower(const Value& v, const Args&, const Kwargs&) { return Value(lower_ascii(v.as_string())); }
Value m_title(const Value& v, const Args&, const Kwargs&) { return Value(title_case(v.as_string())); }
Value m_capitalize(const Value& v, const Args&, const Kwargs&) { return Value(capitalize(v.as_string())); }

Value m_replace(const Value& v, const Args& a, const Kwargs&) {
    if (a.size() < 2)
        return v;
    return Value(replace_n(v.as_string(), sarg(a, 0), sarg(a, 1), a.size() > 2 ? to_int(a[2], -1) : -1));
}

// str.split(sep=None, maxsplit=-1); str.rsplit splits from the right.
Value split_impl(const Value& v, const Args& a, const Kwargs& kw, bool from_right) {
    const std::string& s = v.as_string();
    const Value sepv = param(a, kw, 0, "sep");
    int64_t maxsplit = to_int(param(a, kw, 1, "maxsplit", Value(int64_t{-1})), -1);
    Value::Array parts;
    if (sepv.is_none()) {
        std::string cur;
        for (char c : s) {
            if (is_ws(c)) {
                if (!cur.empty())
                    parts.emplace_back(std::exchange(cur, {}));
            } else {
                cur += c;
            }
        }
        if (!cur.empty())
            parts.emplace_back(std::move(cur));
        return Value(std::move(parts));
    }
    const std::string sep = sepv.to_string();
    if (sep.empty())
        return Value(Value::Array{v});
    if (!from_right) {
        size_t start = 0;
        for (size_t pos; maxsplit != 0 && (pos = s.find(sep, start)) != std::string::npos; maxsplit--) {
            parts.emplace_back(s.substr(start, pos - start));
            start = pos + sep.size();
        }
        parts.emplace_back(s.substr(start));
        return Value(std::move(parts));
    }
    size_t end = s.size();
    for (size_t pos;
         maxsplit != 0 && end >= sep.size() && (pos = s.rfind(sep, end - sep.size())) != std::string::npos;
         maxsplit--) {
        parts.insert(parts.begin(), Value(s.substr(pos + sep.size(), end - pos - sep.size())));
        end = pos;
    }
    parts.insert(parts.begin(), Value(s.substr(0, end)));
    return Value(std::move(parts));
}
Value m_split(const Value& v, const Args& a, const Kwargs& kw) { return split_impl(v, a, kw, false); }
Value m_rsplit(const Value& v, const Args& a, const Kwargs& kw) { return split_impl(v, a, kw, true); }

Value m_find(const Value& v, const Args& a, const Kwargs&) {
    const size_t pos = v.as_string().find(sarg(a, 0));
    return Value(pos == std::string::npos ? int64_t{-1} : static_cast<int64_t>(pos));
}
Value m_rfind(const Value& v, const Args& a, const Kwargs&) {
    const size_t pos = v.as_string().rfind(sarg(a, 0));
    return Value(pos == std::string::npos ? int64_t{-1} : static_cast<int64_t>(pos));
}

Value m_str_count(const Value& v, const Args& a, const Kwargs&) {
    const std::string sub = sarg(a, 0);
    const std::string& s = v.as_string();
    if (sub.empty())
        return Value(static_cast<int64_t>(s.size() + 1));
    int64_t n = 0;
    for (size_t pos = 0; (pos = s.find(sub, pos)) != std::string::npos; pos += sub.size())
        n++;
    return Value(n);
}

// sep.join(iterable)
Value m_join(const Value& v, const Args& a, const Kwargs&) {
    if (a.empty() || !a[0].is_array())
        return a.empty() ? Value(std::string()) : Value(a[0].to_string());
    std::string out;
    for (size_t i = 0; i < a[0].as_array().size(); i++) {
        if (i > 0)
            out += v.as_string();
        out += a[0].as_array()[i].to_string();
    }
    return Value(std::move(out));
}

// "{} {0} {name}".format(...), no format specs.
Value m_format(const Value& v, const Args& a, const Kwargs& kw) {
    const std::string& f = v.as_string();
    std::string out;
    size_t auto_idx = 0;
    for (size_t i = 0; i < f.size(); i++) {
        if ((f[i] == '{' || f[i] == '}') && i + 1 < f.size() && f[i + 1] == f[i]) {
            out += f[i++];
            continue;
        }
        const size_t close = f[i] == '{' ? f.find('}', i) : std::string::npos;
        if (close == std::string::npos) {
            out += f[i];
            continue;
        }
        const std::string field = f.substr(i + 1, close - i - 1);
        int64_t idx = 0;
        const auto [p, ec] = std::from_chars(field.data(), field.data() + field.size(), idx);
        if (field.empty())
            out += param(a, {}, auto_idx++, "").to_string();
        else if (ec == std::errc() && p == field.data() + field.size())
            out += param(a, {}, static_cast<size_t>(idx), "").to_string();
        else
            out += param({}, kw, 0, field).to_string();
        i = close;
    }
    return Value(std::move(out));
}

template <int (*Pred)(int)>
Value char_class(const Value& v, const Args&, const Kwargs&) {
    const std::string& s = v.as_string();
    return Value(!s.empty() &&
                 std::ranges::all_of(s, [](char c) { return Pred(static_cast<unsigned char>(c)) != 0; }));
}

Value m_isupper(const Value& v, const Args&, const Kwargs&) {
    const std::string& s = v.as_string();
    return Value(
        std::ranges::any_of(s, [](char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }) &&
        upper_ascii(s) == s);
}
Value m_islower(const Value& v, const Args&, const Kwargs&) {
    const std::string& s = v.as_string();
    return Value(
        std::ranges::any_of(s, [](char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }) &&
        lower_ascii(s) == s);
}

const std::unordered_map<std::string_view, MethodFn>& str_methods() {
    static const std::unordered_map<std::string_view, MethodFn> t = {
        {"strip", m_strip},
        {"lstrip", m_lstrip},
        {"rstrip", m_rstrip},
        {"startswith", m_startswith},
        {"endswith", m_endswith},
        {"upper", m_upper},
        {"lower", m_lower},
        {"title", m_title},
        {"capitalize", m_capitalize},
        {"replace", m_replace},
        {"split", m_split},
        {"rsplit", m_rsplit},
        {"find", m_find},
        {"rfind", m_rfind},
        {"count", m_str_count},
        {"join", m_join},
        {"format", m_format},
        {"isdigit", char_class<std::isdigit>},
        {"isalpha", char_class<std::isalpha>},
        {"isalnum", char_class<std::isalnum>},
        {"isspace", char_class<std::isspace>},
        {"isupper", m_isupper},
        {"islower", m_islower},
    };
    return t;
}

Value m_list_index(const Value& v, const Args& a, const Kwargs&) {
    const auto& arr = v.as_array();
    const auto it = std::ranges::find(arr, arg0(a));
    return it == arr.end() ? Value() : Value(static_cast<int64_t>(it - arr.begin()));
}
Value m_list_count(const Value& v, const Args& a, const Kwargs&) {
    return Value(static_cast<int64_t>(std::ranges::count(v.as_array(), arg0(a))));
}

Value m_items(const Value& v, const Args&, const Kwargs&) { return pairs_of(v); }
Value m_keys(const Value& v, const Args&, const Kwargs&) {
    Value::Array arr;
    for (const auto& [k, val] : *v.as_object())
        arr.emplace_back(k);
    return Value(std::move(arr));
}
Value m_values(const Value& v, const Args&, const Kwargs&) {
    Value::Array arr;
    for (const auto& [k, val] : *v.as_object())
        arr.push_back(val);
    return Value(std::move(arr));
}
Value m_get(const Value& v, const Args& a, const Kwargs&) {
    const auto it = v.as_object()->find(sarg(a, 0));
    return it != v.as_object()->end() ? it->second : (a.size() > 1 ? a[1] : Value());
}

}  // namespace

std::optional<Value> call_method(const Value& obj, const std::string& name, const std::vector<Value>& args,
                                 const Kwargs& kwargs) {
    if (obj.is_string()) {
        const auto& t = str_methods();
        if (auto it = t.find(name); it != t.end())
            return it->second(obj, args, kwargs);
    } else if (obj.is_array()) {
        if (name == "index")
            return m_list_index(obj, args, kwargs);
        if (name == "count")
            return m_list_count(obj, args, kwargs);
    } else if (obj.is_object() && obj.as_object()) {
        if (name == "items")
            return m_items(obj, args, kwargs);
        if (name == "keys")
            return m_keys(obj, args, kwargs);
        if (name == "values")
            return m_values(obj, args, kwargs);
        if (name == "get")
            return m_get(obj, args, kwargs);
    }
    return std::nullopt;
}

}  // namespace imp::jinja::detail
