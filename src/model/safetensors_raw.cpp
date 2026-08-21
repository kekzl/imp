#include "model/safetensors_raw.h"

#include "model/json_util.h"
#include "model/safetensors_loader.h"
#include "model/safetensors_writer.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <utility>

namespace imp {

RawSafeTensors::~RawSafeTensors() { close(); }

RawSafeTensors& RawSafeTensors::operator=(RawSafeTensors&& o) noexcept {
    if (this != &o) {
        close();
        mmap_base_ = o.mmap_base_;
        mmap_size_ = o.mmap_size_;
        tensors_ = std::move(o.tensors_);
        metadata_ = std::move(o.metadata_);
        o.mmap_base_ = nullptr;
        o.mmap_size_ = 0;
    }
    return *this;
}

void RawSafeTensors::close() {
    if (mmap_base_) {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
        mmap_size_ = 0;
    }
    tensors_.clear();
    metadata_.clear();
}

std::string RawSafeTensors::open(const std::string& path) {
    close();

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return "cannot open " + path;

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        return "cannot stat " + path;
    }
    const size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < 8) {
        ::close(fd);
        return path + " is too small to be a SafeTensors file";
    }

    void* base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED)
        return "mmap failed for " + path;
    mmap_base_ = base;
    mmap_size_ = file_size;

    const auto* bytes = static_cast<const uint8_t*>(base);
    uint64_t header_size = 0;
    for (int i = 7; i >= 0; i--)
        header_size = (header_size << 8) | bytes[i];

    std::string err;
    if (!safetensors_internal::validate_header_size(file_size, header_size, &err)) {
        close();
        return path + ": " + err;
    }

    JsonParser parser(
        std::string_view(reinterpret_cast<const char*>(bytes + 8), static_cast<size_t>(header_size)));
    JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT) {
        close();
        return path + ": header is not a JSON object";
    }

    const uint64_t data_offset = 8 + header_size;
    for (const auto& kv : root.obj) {
        if (kv.first == "__metadata__") {
            if (kv.second.type == JType::OBJECT)
                for (const auto& m : kv.second.obj)
                    if (m.second.type == JType::STRING)
                        metadata_.emplace_back(m.first, m.second.str_val);
            continue;
        }
        const JValue& meta = kv.second;
        if (meta.type != JType::OBJECT) {
            close();
            return path + ": entry '" + kv.first + "' is not an object";
        }
        const JValue* dtype = jobj_find(meta, "dtype");
        const JValue* shape = jobj_find(meta, "shape");
        const JValue* offsets = jobj_find(meta, "data_offsets");
        if (!dtype || dtype->type != JType::STRING || !shape || shape->type != JType::ARRAY || !offsets ||
            offsets->type != JType::ARRAY || offsets->arr.size() != 2) {
            close();
            return path + ": entry '" + kv.first + "' has a malformed descriptor";
        }

        RawTensor t;
        t.name = kv.first;
        t.dtype = dtype->str_val;
        for (const auto& d : shape->arr)
            t.shape.push_back(static_cast<int64_t>(d.num_val));

        const size_t width = safetensors_dtype_size(t.dtype);
        if (width == 0) {
            close();
            return path + ": tensor '" + t.name + "' has unsupported dtype '" + t.dtype + "'";
        }
        const uint64_t start = static_cast<uint64_t>(offsets->arr[0].num_val);
        const uint64_t end = static_cast<uint64_t>(offsets->arr[1].num_val);
        const uint64_t expected = static_cast<uint64_t>(t.numel()) * width;
        if (!safetensors_internal::validate_tensor_offsets(start, end, expected, data_offset, file_size,
                                                           &err)) {
            close();
            return path + ": tensor '" + t.name + "': " + err;
        }

        t.data = bytes + data_offset + start;
        t.nbytes = static_cast<size_t>(end - start);
        tensors_.push_back(std::move(t));
    }
    return "";
}

}  // namespace imp
