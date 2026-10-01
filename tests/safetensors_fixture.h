#pragma once

// Reads a small .safetensors fixture into host Tensors (F32 / F16 / BF16) that point into one
// owned buffer. Tests only: no mmap, no shard index.

#include "core/tensor.h"
#include "model/json_util.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp_test {

struct SafetensorsFixture {
    std::vector<uint8_t> bytes;
    std::unordered_map<std::string, imp::Tensor> tensors;

    // Empty `tensors` on any failure.
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        if (bytes.size() < 8)
            return false;
        uint64_t hlen = 0;
        std::memcpy(&hlen, bytes.data(), 8);
        if (8 + hlen > bytes.size())
            return false;
        imp::JsonParser parser(std::string_view(reinterpret_cast<const char*>(bytes.data()) + 8, hlen));
        const imp::JValue root = parser.parse();
        if (!parser.ok() || root.type != imp::JType::OBJECT)
            return false;
        uint8_t* data = bytes.data() + 8 + hlen;
        for (const auto& m : root.obj) {
            if (m.key == "__metadata__")
                continue;
            const imp::JValue* dt = imp::jobj_find(m.value, "dtype");
            const imp::JValue* sh = imp::jobj_find(m.value, "shape");
            const imp::JValue* off = imp::jobj_find(m.value, "data_offsets");
            if (!dt || !sh || !off || off->arr.size() != 2)
                return false;
            imp::QType q = dt->str_val == "F32"    ? imp::QType::F32
                           : dt->str_val == "F16"  ? imp::QType::F16
                           : dt->str_val == "BF16" ? imp::QType::BF16
                                                   : imp::QType::NONE;
            if (q == imp::QType::NONE)
                return false;
            int64_t shape[imp::kMaxDims] = {};
            int nd = 0;
            for (const auto& d : sh->arr)
                if (nd < imp::kMaxDims)
                    shape[nd++] = d.as_int();
            if (nd == 0)
                shape[nd++] = 1;
            tensors[m.key] = imp::Tensor(data + off->arr[0].as_int(), q, nd, shape, false);
        }
        return true;
    }

    // F32 values of one tensor (F32 fixtures only).
    std::vector<float> floats(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end() || it->second.qtype != imp::QType::F32)
            return {};
        const float* p = static_cast<const float*>(it->second.data);
        return std::vector<float>(p, p + it->second.numel());
    }
};

}  // namespace imp_test
