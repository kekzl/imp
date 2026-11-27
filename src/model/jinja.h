#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace imp::jinja {

// Forward declarations for AST nodes (defined in jinja_ast.h)
namespace detail {
struct Node;
struct Expr;
}  // namespace detail

class ObjectMap;

// ---------------------------------------------------------------------------
// Value — dynamically typed value used in template contexts
// ---------------------------------------------------------------------------

class Value {
public:
    using Array = std::vector<Value>;
    // Insertion-ordered like a Python dict: tool schemas render in the client's key order.
    using Object = std::shared_ptr<ObjectMap>;

    // Underlying storage
    using Storage = std::variant<std::monostate,  // none
                                 bool,            // boolean
                                 int64_t,         // integer
                                 double,          // floating point
                                 std::string,     // string
                                 Array,           // list
                                 Object           // object / namespace
                                 >;

    // Constructors
    Value() : data_(std::monostate{}) {}
    Value(std::monostate) : data_(std::monostate{}) {}
    Value(bool v) : data_(v) {}
    Value(int v) : data_(static_cast<int64_t>(v)) {}
    Value(int64_t v) : data_(v) {}
    Value(double v) : data_(v) {}
    Value(const char* v) : data_(std::string(v)) {}
    Value(std::string v) : data_(std::move(v)) {}
    Value(std::string_view v) : data_(std::string(v)) {}
    Value(Array v) : data_(std::move(v)) {}
    Value(Object v) : data_(std::move(v)) {}

    // Type checks
    bool is_none() const { return std::holds_alternative<std::monostate>(data_); }
    bool is_bool() const { return std::holds_alternative<bool>(data_); }
    bool is_int() const { return std::holds_alternative<int64_t>(data_); }
    bool is_double() const { return std::holds_alternative<double>(data_); }
    bool is_string() const { return std::holds_alternative<std::string>(data_); }
    bool is_array() const { return std::holds_alternative<Array>(data_); }
    bool is_object() const { return std::holds_alternative<Object>(data_); }

    bool is_number() const { return is_int() || is_double(); }

    // Accessors (unchecked — caller must verify type)
    [[nodiscard]] bool as_bool() const { return std::get<bool>(data_); }
    int64_t as_int() const { return std::get<int64_t>(data_); }
    double as_double() const { return std::get<double>(data_); }
    const std::string& as_string() const { return std::get<std::string>(data_); }
    const Array& as_array() const { return std::get<Array>(data_); }
    // C++23 deducing this: one overload serves const and non-const callers.
    template <typename Self>
    auto&& as_object(this Self& self) {
        return std::get<Object>(self.data_);
    }

    // Truthiness (Python/Jinja2 rules)
    [[nodiscard]] bool truthy() const;

    // Convert to string for output
    std::string to_string() const;

    // Convert to number (int64_t or double, for arithmetic)
    double to_number() const;

    // Comparison
    bool operator==(const Value& other) const;
    bool operator!=(const Value& other) const { return !(*this == other); }
    bool operator<(const Value& other) const;
    bool operator>(const Value& other) const { return other < *this; }
    bool operator<=(const Value& other) const { return !(other < *this); }
    bool operator>=(const Value& other) const { return !(*this < other); }

    // Arithmetic / concatenation
    Value operator+(const Value& other) const;

    // Length (string or array)
    int64_t length() const;

    // Subscript access: a[index] or a["key"]
    Value get(const Value& key) const;

    // Attribute access: a.name
    Value get(const std::string& name) const;

    // Set attribute on object
    void set(const std::string& name, Value val);

    // Check if value contains item (for 'in' operator)
    [[nodiscard]] bool contains(const Value& item) const;

    // Size (alias for length)
    int64_t size() const { return length(); }

    // Truthiness alias
    bool is_truthy() const { return truthy(); }

    // Create an empty namespace/object
    static Value make_object();

    // Convenience: create array from initializer list
    static Value array(std::initializer_list<Value> items) { return Value(Array(items)); }

    // Convenience: create object from initializer list
    static Value object(std::initializer_list<std::pair<std::string, Value>> items);

    const Storage& storage() const { return data_; }

private:
    Storage data_;
};

// Python-dict semantics: insertion order kept, assigning an existing key keeps its slot.
// Linear lookup: template objects (messages, tool schemas, namespaces) hold < 100 keys.
class ObjectMap {
public:
    using value_type = std::pair<std::string, Value>;
    using iterator = std::vector<value_type>::iterator;
    using const_iterator = std::vector<value_type>::const_iterator;

    iterator begin() { return items_.begin(); }
    iterator end() { return items_.end(); }
    const_iterator begin() const { return items_.begin(); }
    const_iterator end() const { return items_.end(); }
    size_t size() const { return items_.size(); }
    bool empty() const { return items_.empty(); }

    iterator find(std::string_view key) {
        for (auto it = items_.begin(); it != items_.end(); ++it)
            if (it->first == key)
                return it;
        return items_.end();
    }
    const_iterator find(std::string_view key) const { return const_cast<ObjectMap*>(this)->find(key); }
    [[nodiscard]] bool contains(std::string_view key) const {
        return std::ranges::any_of(items_, [&](const value_type& kv) { return kv.first == key; });
    }
    size_t count(std::string_view key) const { return contains(key) ? 1 : 0; }

    Value& operator[](const std::string& key) {
        auto it = find(key);
        if (it != end())
            return it->second;
        return items_.emplace_back(key, Value()).second;
    }
    std::pair<iterator, bool> emplace(std::string key, Value val) {
        auto it = find(key);
        if (it != end())
            return {it, false};
        items_.emplace_back(std::move(key), std::move(val));
        return {items_.end() - 1, true};
    }
    size_t erase(std::string_view key) {
        auto it = find(key);
        if (it == end())
            return 0;
        items_.erase(it);
        return 1;
    }

private:
    std::vector<value_type> items_;
};

inline Value Value::object(std::initializer_list<std::pair<std::string, Value>> items) {
    auto obj = std::make_shared<ObjectMap>();
    for (const auto& [k, v] : items)
        obj->emplace(k, v);
    return Value(obj);
}

// ---------------------------------------------------------------------------
// Context — variable lookup with scoping
// ---------------------------------------------------------------------------

using Context = std::map<std::string, Value>;

// ---------------------------------------------------------------------------
// Template — parse and render Jinja2 templates
// ---------------------------------------------------------------------------

class Template {
public:
    Template();
    ~Template();

    Template(Template&&) noexcept;
    Template& operator=(Template&&) noexcept;

    // Non-copyable (AST uses unique_ptr)
    Template(const Template&) = delete;
    Template& operator=(const Template&) = delete;

    // Parse a Jinja2 template string. Returns true on success.
    [[nodiscard]] bool parse(const std::string& source);

    // Renders the parsed template; returns "" on error. A render exceeding the evaluation
    // budget (macro depth, loop iterations) is an error, not a crash: error() names the budget.
    std::string render(const Context& ctx) const;

    // Last error message (set on parse/render failure)
    const std::string& error() const { return error_; }

private:
    std::vector<std::unique_ptr<detail::Node>> nodes_;
    mutable std::string error_;  // render() is const and still reports
};

}  // namespace imp::jinja
