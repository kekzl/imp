#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// Minimal SafeTensors writer — the missing half of safetensors_loader.
//
// It exists so imp can PRODUCE checkpoints, not just consume them: the NVFP4
// path currently depends on somebody else publishing a Modelopt /
// llm-compressor export, which gates both model coverage and quality on a
// third party (see docs/roadmap.md, gap 1).
//
// Layout written (the format's whole spec):
//   [8 bytes little-endian header length][header JSON][tensor data]
// The header maps name -> {dtype, shape, data_offsets:[start,end)}, offsets
// being relative to the start of the data block. Tensor data is written in
// header order, and the header is padded with spaces so the data block starts
// 8-byte aligned (what every mainstream reader expects, mmap-friendly).
//
// dtype is passed through verbatim and must be a SafeTensors wire name the
// loader accepts: "F32", "F16", "BF16", "F8_E4M3", "U8", "I8", ... For NVFP4
// weights that means U8 with the ALREADY-PACKED shape [N, K/2] (two FP4
// nibbles per byte), F8_E4M3 micro-scales, and an F32 tensor scale.

struct SafeTensorsOut {
    std::string name;
    std::string dtype;           // SafeTensors wire dtype, e.g. "F16" / "U8"
    std::vector<int64_t> shape;  // as stored (packed dims for sub-byte data)
    const void* data = nullptr;  // host memory, nbytes readable
    size_t nbytes = 0;
};

// Write one .safetensors file. Returns an empty string on success, otherwise a
// one-line reason. Nothing is left behind on failure: the file is written to a
// temporary path and renamed only once fully flushed, so a crash or a full disk
// cannot leave a half-written checkpoint that later loads as garbage.
//
// `metadata` entries land under the reserved "__metadata__" key (string->string
// only, per the format).
std::string write_safetensors(const std::string& path, const std::vector<SafeTensorsOut>& tensors,
                              const std::vector<std::pair<std::string, std::string>>& metadata = {});

// Byte width of a SafeTensors wire dtype; 0 if unknown. Exposed for callers
// that want to size/verify a tensor before handing it over.
size_t safetensors_dtype_size(const std::string& dtype);

}  // namespace imp
