#pragma once

// Minimal JSON parser shared across model loaders. Handles UTF-16 surrogate pairs in
// \uXXXX, emits UTF-8. Scalars: string/number/bool(as NUMBER 0.0/1.0)/null. No streaming;
// whole document materialized as a JValue tree.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace imp {

enum class JType { NUL, STRING, NUMBER, ARRAY, OBJECT };

struct JValue;

// Not std::pair: vector<T> accepts an incomplete T (needed for the recursive JSON tree),
// but pair<string,JValue> is complete with an incomplete member, a different permission -
// libstdc++'s defaulted dtor instantiation there fails to build under clang (IMP_FUZZERS=ON).
struct JMember;

struct JValue {
    JType type = JType::NUL;
    std::string str_val;
    double num_val = 0.0;
    std::vector<JValue> arr;
    std::vector<JMember> obj;

    int64_t as_int() const { return static_cast<int64_t>(num_val); }
};

struct JMember {
    std::string key;
    JValue value;

    JMember() = default;
    JMember(std::string k, JValue v) : key(std::move(k)), value(std::move(v)) {}
};

class JsonParser {
public:
    // Depth cap bounds the stack, not the heap: a JValue is ~104 bytes against 2 bytes of "0,"
    // text, so a 128 MiB SafeTensors header alone admits ~6.6 GiB of tree (AUDIT_arch_2026 F1-10).
    // 8M nodes is ~870 MiB; the largest real input (151k-vocab tokenizer.json) is ~0.4M.
    static constexpr size_t kMaxNodes = size_t{1} << 23;

    // A view, not (pointer,length): every caller already had both halves of one object.
    // `max_nodes` is the seam the node-budget test uses; production callers take the default.
    explicit JsonParser(std::string_view data, size_t max_nodes = kMaxNodes);

    JValue parse();
    bool ok() const { return !error_; }

private:
    const char* data_;
    size_t len_;
    size_t pos_;
    bool error_ = false;
    int depth_ = 0;
    size_t nodes_ = 0;
    size_t max_nodes_;

    // Recursion is bounded because the input isn't: a SafeTensors header may declare up to
    // 128 MiB, each '[' is one parse_value frame (~240 B stack). An 8 MiB stack survives
    // ~30000 levels, SIGSEGVs before 40000; 512 is far past any real config (deepest here: 7).
    static constexpr int kMaxDepth = 512;

    // RAII depth counter: `parse_value` has eight returns and the count has to
    // be right on all of them.
    struct DepthGuard {
        JsonParser& p;
        explicit DepthGuard(JsonParser& parser) : p(parser) { ++p.depth_; }
        ~DepthGuard() { --p.depth_; }
    };

    char peek() const;
    char advance();
    void skip_ws();
    bool expect(char c);
    static int hex_digit(char c);
    uint32_t parse_u4();
    static void append_codepoint_utf8(std::string& s, uint32_t cp);

    JValue parse_value();
    JValue parse_string_value();
    std::string parse_string_raw();
    JValue parse_number();
    JValue parse_object();
    JValue parse_array();
    JValue parse_bool();
    JValue parse_null();
};

// Object lookup by key. Returns nullptr if absent.
const JValue* jobj_find(const JValue& obj, const std::string& key);

// Typed field accessors. Return false (and leave `out` untouched) on missing
// key or wrong type.
template <typename T>
bool jobj_get_int(const JValue& obj, const std::string& key, T& out) {
    const JValue* v = jobj_find(obj, key);
    if (!v || v->type != JType::NUMBER)
        return false;
    out = static_cast<T>(v->num_val);
    return true;
}

bool jobj_get_float(const JValue& obj, const std::string& key, float& out);
bool jobj_get_string(const JValue& obj, const std::string& key, std::string& out);

// Slurp a file into a string. Empty string on error.
std::string read_file(const std::string& path);

// Parse a JSON file into a JValue. Returns false on I/O or parse error, or if
// the root is not an object.
bool parse_json_file(const std::string& path, JValue& out);

}  // namespace imp
