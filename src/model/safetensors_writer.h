#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// Minimal SafeTensors writer, the missing half of safetensors_loader: lets imp PRODUCE
// checkpoints instead of only consuming Modelopt/llm-compressor exports (docs/roadmap.md
// gap 1). Layout: [8-byte LE header length][header JSON][tensor data]; header maps
// name->{dtype,shape,data_offsets:[start,end)} relative to the data block start, padded to
// 8-byte alignment. dtype is a SafeTensors wire name the loader accepts; NVFP4 weights are
// U8 packed [N,K/2] + F8_E4M3 micro-scales + an F32 tensor scale.

struct SafeTensorsOut {
    std::string name;
    std::string dtype;           // SafeTensors wire dtype, e.g. "F16" / "U8"
    std::vector<int64_t> shape;  // as stored (packed dims for sub-byte data)
    const void* data = nullptr;  // host memory, nbytes readable
    size_t nbytes = 0;
};

// Writes one .safetensors file; returns "" on success, else a one-line reason. Nothing is
// left on failure: written to a temp path and renamed only once fully flushed, so a crash
// or full disk cannot leave a half-written checkpoint that later loads as garbage.
// `metadata` entries land under the reserved "__metadata__" key (string->string only).
std::string write_safetensors(const std::string& path, const std::vector<SafeTensorsOut>& tensors,
                              const std::vector<std::pair<std::string, std::string>>& metadata = {});

// Byte width of a SafeTensors wire dtype; 0 if unknown. Exposed for callers
// that want to size/verify a tensor before handing it over.
size_t safetensors_dtype_size(const std::string& dtype);

}  // namespace imp
