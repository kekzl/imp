#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// Raw, name-preserving SafeTensors reading — the counterpart of
// safetensors_writer, for tools that transform a checkpoint instead of loading
// one to run it.
//
// safetensors_loader.cpp deliberately does more than this: it translates tensor
// names, folds llm-compressor layouts, pulls MTP sidecars and assembles a
// Model. A quantizer must see the file exactly as written — original names,
// original dtypes — so it reads through here instead, and the production load
// path stays untouched.
//
// Tensor data points into a private read-only mmap owned by RawSafeTensors and
// stays valid until it is destroyed.

struct RawTensor {
    std::string name;
    std::string dtype;  // SafeTensors wire dtype, verbatim ("BF16", ...)
    std::vector<int64_t> shape;
    const void* data = nullptr;  // into the mmap; not owned
    size_t nbytes = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape)
            n *= d;
        return n;
    }
};

class RawSafeTensors {
public:
    RawSafeTensors() = default;
    ~RawSafeTensors();
    RawSafeTensors(const RawSafeTensors&) = delete;
    RawSafeTensors& operator=(const RawSafeTensors&) = delete;
    RawSafeTensors(RawSafeTensors&& o) noexcept { *this = std::move(o); }
    RawSafeTensors& operator=(RawSafeTensors&& o) noexcept;

    // Opens and validates one .safetensors file. Returns an empty string on
    // success, otherwise a one-line reason. Offsets and header size are checked
    // with the same rules the loader enforces (safetensors_internal::*), so a
    // truncated or hostile file is rejected here rather than read out of bounds.
    std::string open(const std::string& path);

    const std::vector<RawTensor>& tensors() const { return tensors_; }
    // "__metadata__" entries, in file order.
    const std::vector<std::pair<std::string, std::string>>& metadata() const { return metadata_; }

private:
    void close();

    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::vector<RawTensor> tensors_;
    std::vector<std::pair<std::string, std::string>> metadata_;
};

}  // namespace imp
