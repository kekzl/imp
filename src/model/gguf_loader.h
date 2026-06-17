#pragma once

#include "model/model.h"
#include <string>
#include <memory>
#include <cstdint>
#include <vector>

namespace imp {

// ---- GGUF format constants ----

static constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in little-endian
static constexpr size_t GGUF_DEFAULT_ALIGNMENT = 32;

// GGUF metadata value types
enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// GGML tensor quantization types
enum class GgufWireType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5 are deprecated
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    MXFP4 = 31,  // 32 elements: 16 bytes E2M1 + 1 byte UE8M0 scale (imp legacy)
    // 32..38 are unused / removed in modern ggml.
    MXFP4_V2 = 39,  // Modern llama.cpp GGML_TYPE_MXFP4 = 39 — identical 17-byte block layout to MXFP4=31.
};

// Block size (number of elements per quantization block)
int gguf_blck_size(GgufWireType type);

// Bytes per block for quantized types, bytes per element for unquantized
size_t gguf_type_size(GgufWireType type);

// Total bytes for a row of n_elements of given type
size_t gguf_row_size(GgufWireType type, int64_t n_elements);

// Convert GGML type to our QType (lossy for quantized types)
QType gguf_type_to_qtype(GgufWireType type);

const char* gguf_type_name(GgufWireType type);

// Tensor info parsed from GGUF file
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims{};
    int64_t dims[4]{};  // GGUF order: dims[0] = innermost (fastest-changing)
    GgufWireType type;
    uint64_t offset{};                   // relative to start of tensor data section
    const uint8_t* data_base = nullptr;  // shard data base pointer (for split GGUF)
    size_t data_limit = 0;               // bytes available from data_base (for bounds checks)
};

// Load a model from a GGUF file.
// Weights are mmap'd from the file (on_device=false).
// Tokenizer is populated from GGUF metadata if available.
std::unique_ptr<Model> load_gguf(const std::string& path);

}  // namespace imp
