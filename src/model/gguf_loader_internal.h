#pragma once

// ============================================================================
// GGUF loader — internal shared declarations.
//
// File-local helpers/tables that are needed by more than one of the gguf_*.cpp
// translation units (gguf_loader.cpp, gguf_parse.cpp, gguf_tensor_assign.cpp).
// NOT part of the public loader API (model/gguf_loader.h) — do not include
// outside the gguf_* TUs.
// ============================================================================

#include "model/gguf_loader.h"
#include "model/model.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace imp {

// ---- Binary reader over mmap'd memory ----

class BinaryReader {
public:
    BinaryReader(const uint8_t* data, size_t size) : data_(data), size_(size), pos_(0) {}

    size_t pos() const { return pos_; }
    size_t remaining() const { return size_ - pos_; }
    const uint8_t* ptr() const { return data_ + pos_; }
    bool failed() const { return failed_; }
    void fail() { failed_ = true; }

    // Bounds check for an n-byte read at the current cursor. Guards against
    // pos_ + n overflowing on attacker-controlled u64 lengths (a wrapped sum
    // would otherwise compare <= size_ and admit an out-of-bounds read).
    bool check(size_t n) const { return n <= size_ - pos_; }

    void skip(size_t n) {
        if (!check(n)) {
            failed_ = true;
            return;
        }
        pos_ += n;
    }

    void align(size_t alignment) {
        size_t rem = pos_ % alignment;
        if (rem != 0) {
            size_t pad = alignment - rem;
            if (!check(pad)) {
                failed_ = true;
                return;
            }
            pos_ += pad;
        }
    }

    template <typename T>
    T read() {
        if (!check(sizeof(T))) {
            failed_ = true;
            return T{};
        }
        T val;
        std::memcpy(&val, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return val;
    }

    uint8_t read_u8() { return read<uint8_t>(); }
    int8_t read_i8() { return read<int8_t>(); }
    uint16_t read_u16() { return read<uint16_t>(); }
    int16_t read_i16() { return read<int16_t>(); }
    uint32_t read_u32() { return read<uint32_t>(); }
    int32_t read_i32() { return read<int32_t>(); }
    uint64_t read_u64() { return read<uint64_t>(); }
    int64_t read_i64() { return read<int64_t>(); }
    float read_f32() { return read<float>(); }
    double read_f64() { return read<double>(); }

    std::string read_string() {
        uint64_t len = read_u64();
        // A length that runs past EOF (incl. an absurd 2^60-style value) is a
        // hard parse error, not a recoverable empty string: returning "" here
        // would leave the cursor desynced and let the caller keep parsing
        // garbage. Flag failure so the surrounding loop stops.
        if (failed_ || len > remaining()) {
            failed_ = true;
            return "";
        }
        std::string s(reinterpret_cast<const char*>(data_ + pos_), static_cast<size_t>(len));
        pos_ += static_cast<size_t>(len);
        return s;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
    bool failed_ = false;
};

// ---- GGUF metadata value (variant-like) ----

struct GGUFValue {
    GGUFValueType type = GGUFValueType::UINT32;
    uint64_t uint_val = 0;
    int64_t int_val = 0;
    double float_val = 0.0;
    std::string str_val;
    std::vector<std::string> str_array;
    std::vector<float> float_array;
    std::vector<int32_t> int_array;
};

// ---- Shared parsing / assignment helpers (defined in gguf_parse.cpp /
//      gguf_tensor_assign.cpp; consumed by load_gguf in gguf_loader.cpp) ----

GGUFValue read_gguf_value(BinaryReader& r, GGUFValueType type);
uint64_t val_uint(const GGUFValue& v);
double val_float(const GGUFValue& v);

void parse_tensor_infos(BinaryReader& reader, uint64_t tensor_count,
                        std::vector<GGUFTensorInfo>& out);

bool gguf_tensor_in_bounds(const GGUFTensorInfo& info);

bool assign_tensor(Model& model, const std::string& name, const Tensor& tensor, GgufWireType gtype);

}  // namespace imp
