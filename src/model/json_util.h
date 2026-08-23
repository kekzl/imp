#pragma once

// Minimal JSON parser shared across model loaders (hf_config_loader,
// safetensors_loader, tokenizer, …). Handles UTF-16 surrogate pairs in
// `\uXXXX` escapes and emits proper UTF-8.
//
// Supported scalars: string, number, bool (encoded as NUMBER 0.0/1.0), null.
// Containers: object (preserves insertion order) and array. No streaming;
// the entire document is materialized as a tree of `JValue`.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace imp {

enum class JType { NUL, STRING, NUMBER, ARRAY, OBJECT };

struct JValue {
    JType type = JType::NUL;
    std::string str_val;
    double num_val = 0.0;
    std::vector<JValue> arr;
    std::vector<std::pair<std::string, JValue>> obj;

    int64_t as_int() const { return static_cast<int64_t>(num_val); }
};

class JsonParser {
public:
    // A view, not (pointer, length): every caller already had both halves of
    // one object and had to spell the pair out.
    explicit JsonParser(std::string_view data);

    JValue parse();
    bool ok() const { return !error_; }

private:
    const char* data_;
    size_t len_;
    size_t pos_;
    bool error_ = false;
    int depth_ = 0;

    // Recursion is bounded because the input is not: a SafeTensors header may
    // declare up to 128 MiB, and every '[' is one `parse_value` frame at ~240
    // bytes of stack. Measured on an 8 MiB stack, this parser survives 30 000
    // levels and takes SIGSEGV somewhere before 40 000, so a 128 MiB header of
    // '[' crashes the loader before a weight is read. 512 is far past any real
    // config.json or tokenizer.json; the deepest in this tree is 7.
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
