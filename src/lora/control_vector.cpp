#include "lora/control_vector.h"

#include "model/gguf_loader.h"
#include "model/gguf_loader_internal.h"

#include <charconv>
#include <cstring>
#include <fstream>
#include <iterator>

namespace imp {

namespace {
bool fail(std::string* err, const std::string& msg) {
    if (err)
        *err = msg;
    return false;
}
}  // namespace

bool parse_control_vector_gguf(std::span<const uint8_t> bytes, int d_model, int n_layers,
                               ControlVectorHost& out, std::string* err) {
    if (d_model <= 0 || n_layers <= 0)
        return fail(err, "control vector: model dims unknown");
    BinaryReader r(bytes);
    if (r.read_u32() != GGUF_MAGIC)
        return fail(err, "control vector: not a GGUF file");
    const uint32_t version = r.read_u32();
    if (version < 2 || version > 3)
        return fail(err, "control vector: unsupported GGUF version " + std::to_string(version));
    const uint64_t n_tensors = r.read_u64();
    const uint64_t n_kv = r.read_u64();
    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < n_kv && !r.failed(); ++i) {
        const std::string key = r.read_string();
        const GGUFValue v = read_gguf_value(r, static_cast<GGUFValueType>(r.read_u32()));
        if (key == "general.alignment" && val_uint(v) != 0)
            alignment = static_cast<size_t>(val_uint(v));
    }
    std::vector<GGUFTensorInfo> infos;
    if (!r.failed())
        parse_tensor_infos(r, n_tensors, infos);
    if (r.failed())
        return fail(err, "control vector: truncated GGUF header");
    r.align(alignment);
    const size_t data_start = r.pos();
    if (r.failed() || data_start > bytes.size())
        return fail(err, "control vector: truncated GGUF header");

    out.d_model = d_model;
    out.dirs.assign(static_cast<size_t>(n_layers) * d_model, 0.0f);
    out.layer_mask.assign(static_cast<size_t>(n_layers), 0);
    int found = 0;
    for (const auto& t : infos) {
        constexpr std::string_view kPrefix = "direction.";
        if (!t.name.starts_with(kPrefix))
            continue;
        int layer = -1;
        const char* first = t.name.data() + kPrefix.size();
        const char* last = t.name.data() + t.name.size();
        if (std::from_chars(first, last, layer).ptr != last || layer < 1 || layer >= n_layers)
            return fail(err,
                        "control vector: " + t.name + " outside layers 1.." + std::to_string(n_layers - 1));
        if (t.type != GgufWireType::F32)
            return fail(err, "control vector: " + t.name + " is not F32");
        int64_t n = 1;
        for (uint32_t d = 0; d < t.n_dims; ++d)
            n *= t.dims[d];
        if (n != d_model)
            return fail(err, "control vector: " + t.name + " has " + std::to_string(n) +
                                 " values, model d_model " + std::to_string(d_model));
        const size_t nbytes = sizeof(float) * static_cast<size_t>(d_model);
        if (t.offset > bytes.size() - data_start || nbytes > bytes.size() - data_start - t.offset)
            return fail(err, "control vector: " + t.name + " data past end of file");
        std::memcpy(out.dirs.data() + static_cast<size_t>(layer) * d_model,
                    bytes.data() + data_start + t.offset, nbytes);
        out.layer_mask[layer] = 1;
        ++found;
    }
    if (found == 0)
        return fail(err, "control vector: no direction.N tensor");
    return true;
}

bool load_control_vector_file(const std::string& path, int d_model, int n_layers, ControlVectorHost& out,
                              std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return fail(err, "control vector: cannot open " + path);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse_control_vector_gguf(bytes, d_model, n_layers, out, err);
}

}  // namespace imp
