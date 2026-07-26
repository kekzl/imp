#include "model/safetensors_writer.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace imp {

size_t safetensors_dtype_size(const std::string& dtype) {
    if (dtype == "F64" || dtype == "I64" || dtype == "U64")
        return 8;
    if (dtype == "F32" || dtype == "I32" || dtype == "U32")
        return 4;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16")
        return 2;
    if (dtype == "F8_E4M3" || dtype == "F8_E5M2" || dtype == "I8" || dtype == "U8" || dtype == "BOOL")
        return 1;
    return 0;
}

namespace {

// The header is JSON, so the few characters JSON reserves must be escaped.
// Tensor names come from a model's own naming, but a checkpoint that fails to
// round-trip because of an unescaped quote would be a silent corruption.
void append_json_string(std::string& out, const std::string& s) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

}  // namespace

std::string write_safetensors(const std::string& path, const std::vector<SafeTensorsOut>& tensors,
                              const std::vector<std::pair<std::string, std::string>>& metadata) {
    if (tensors.empty())
        return "refusing to write a checkpoint with no tensors";

    // Validate everything BEFORE creating the file: a rejected write should
    // leave no trace at all.
    for (const auto& t : tensors) {
        if (t.name.empty())
            return "tensor with an empty name";
        if (t.name == "__metadata__")
            return "'__metadata__' is reserved and cannot be a tensor name";
        if (t.data == nullptr && t.nbytes != 0)
            return "tensor '" + t.name + "' has null data but nbytes=" + std::to_string(t.nbytes);
        const size_t width = safetensors_dtype_size(t.dtype);
        if (width == 0)
            return "tensor '" + t.name + "' has unsupported dtype '" + t.dtype + "'";
        // Element count from the shape must match the buffer. For sub-byte data
        // (NVFP4 packed into U8) the shape is the packed shape, so this still
        // holds — that is exactly why the caller passes packed dims.
        size_t elems = 1;
        for (int64_t d : t.shape) {
            if (d < 0)
                return "tensor '" + t.name + "' has a negative dimension";
            elems *= static_cast<size_t>(d);
        }
        if (elems * width != t.nbytes)
            return "tensor '" + t.name + "' shape implies " + std::to_string(elems * width) +
                   " bytes but nbytes=" + std::to_string(t.nbytes);
    }

    // Build the header. Offsets are relative to the start of the data block.
    std::string header = "{";
    if (!metadata.empty()) {
        header += "\"__metadata__\":{";
        for (size_t i = 0; i < metadata.size(); i++) {
            if (i)
                header += ',';
            append_json_string(header, metadata[i].first);
            header += ':';
            append_json_string(header, metadata[i].second);
        }
        header += "},";
    }
    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        const auto& t = tensors[i];
        if (i)
            header += ',';
        append_json_string(header, t.name);
        header += ":{\"dtype\":";
        append_json_string(header, t.dtype);
        header += ",\"shape\":[";
        for (size_t d = 0; d < t.shape.size(); d++) {
            if (d)
                header += ',';
            header += std::to_string(t.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + t.nbytes) +
                  "]}";
        offset += t.nbytes;
    }
    header += '}';

    // Pad so the data block starts 8-byte aligned. Trailing spaces are legal
    // JSON whitespace and every reader tolerates them.
    const size_t pad = (8 - ((8 + header.size()) % 8)) % 8;
    header.append(pad, ' ');

    // Write to a temporary next to the target, then rename: an interrupted
    // write must never leave a file that parses but holds truncated tensors.
    const std::string tmp = path + ".partial";
    std::error_code ec;
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f)
        return "cannot open '" + tmp + "' for writing: " + std::strerror(errno);

    auto fail = [&](const std::string& why) {
        fclose(f);
        std::filesystem::remove(tmp, ec);
        return why;
    };

    const uint64_t header_len = header.size();
    unsigned char len_le[8];
    for (int i = 0; i < 8; i++)
        len_le[i] = static_cast<unsigned char>((header_len >> (8 * i)) & 0xFF);
    if (fwrite(len_le, 1, 8, f) != 8)
        return fail("short write on the header length");
    if (fwrite(header.data(), 1, header.size(), f) != header.size())
        return fail("short write on the header");
    for (const auto& t : tensors) {
        if (t.nbytes == 0)
            continue;
        if (fwrite(t.data, 1, t.nbytes, f) != t.nbytes)
            return fail("short write on tensor '" + t.name + "' (disk full?)");
    }
    if (fflush(f) != 0)
        return fail("flush failed: " + std::string(std::strerror(errno)));
    if (fclose(f) != 0) {
        std::filesystem::remove(tmp, ec);
        return "close failed: " + std::string(std::strerror(errno));
    }

    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return "rename to '" + path + "' failed: " + ec.message();
    }
    return "";
}

}  // namespace imp
