// Jinja2 Value: truthiness, printing, item access.

#include "model/jinja.h"
#include "model/jinja_builtins.h"
#include "core/logging.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace imp::jinja {

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
            else if constexpr (std::is_same_v<T, double>)
                return detail::py_float_repr(v);  // 3.0, 0.1, 1e+20 as Python prints them
            else if constexpr (std::is_same_v<T, std::string>)
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

namespace {

// Python dict equality: same keys, equal values, order ignored.
bool objects_equal(const Value::Object& a, const Value::Object& b) {
    if (!a || !b)
        return a == b;
    if (a->size() != b->size())
        return false;
    for (const auto& [k, v] : *a) {
        auto it = b->find(k);
        if (it == b->end() || it->second != v)
            return false;
    }
    return true;
}

}  // namespace

bool Value::operator==(const Value& o) const {
    if (is_none() && o.is_none())
        return true;
    if (is_bool() && o.is_bool())
        return as_bool() == o.as_bool();
    // bool is an int in Python: 0 == false, 1 == true
    if ((is_number() || is_bool()) && (o.is_number() || o.is_bool()))
        return to_number() == o.to_number();
    if (is_string() && o.is_string())
        return as_string() == o.as_string();
    if (is_array() && o.is_array())
        return std::ranges::equal(as_array(), o.as_array());
    if (is_object() && o.is_object())
        return objects_equal(as_object(), o.as_object());
    // none == false (kept: `is undefined` paths rely on it)
    if ((is_none() && o.is_bool()) || (is_bool() && o.is_none()))
        return !(is_bool() ? as_bool() : o.as_bool());
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
        return as_string().contains(item.as_string());
    }
    if (is_object() && as_object() && item.is_string()) {
        return as_object()->contains(item.as_string());
    }
    return false;
}

Value Value::make_object() { return Value(std::make_shared<ObjectMap>()); }

}  // namespace imp::jinja
