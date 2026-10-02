#pragma once

// The LM head an export ships (#2479). Default: the per-row FP8 E4M3 head gemm.nvfp4_lm_head=auto
// builds at load, so the loader serves the file's bytes instead of converting. Host-only part
// (selection, FP16 staging, tensor layout); the codes come from the runtime kernel in main.cpp.

#include "checkpoint_out.h"
#include "model/safetensors_raw.h"
#include "model/safetensors_writer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace imp::quantize {

enum class LmHeadExport {
    Source,  // copied at source precision; the runtime converts at load
    Nvfp4,   // quantized like every Linear (`--lm-head` bare flag, the pre-#2479 opt-in)
    Fp8,     // per-row E4M3 codes + F32 row scales, the runtime auto head
};

// fp8 | nvfp4 | source. False on anything else.
bool parse_lm_head_export(const std::string& s, LmHeadExport& out);
const char* lm_head_export_name(LmHeadExport h);

// Fp8 for modelopt; Source for compressed-tensors, since vLLM's ParallelLMHead takes no scales.
LmHeadExport default_lm_head_export(OutputFormat fmt);

// Whether `t` can be written as the FP8 head: exactly lm_head.weight, 2-D, F16/BF16/F32, and
// cols % 256 == 0 (the FP8 head GEMV limit). False sets `why_not`.
bool fp8_head_writable(const RawTensor& t, std::string& why_not);

// The FP16 bits the loader uploads for this tensor (weight_upload_traits.h stage_bf16 /
// F32Fmt / F16 as-is), so the export quantizes the same values the runtime would.
std::vector<uint16_t> head_fp16_as_loaded(const RawTensor& t);

// On-disk bytes of an FP8 head: rows * cols codes + rows F32 scales.
size_t fp8_head_bytes(int64_t rows, int64_t cols);

// The two tensors the head becomes: lm_head.weight F8_E4M3 [rows, cols] and
// kLmHeadRowScaleTensor F32 [rows]. Pointers are borrowed until the shard is written.
std::array<SafeTensorsOut, 2> fp8_head_outputs(int64_t rows, int64_t cols, const uint8_t* codes,
                                               const float* row_scales);

struct Fp8Head {
    std::vector<uint8_t> codes;  // [rows, cols] E4M3
    std::vector<float> scales;   // [rows]
};

// The runtime's own kernel (quantize_fp8_rows_async) on the device: the codes and scales
// gemm.nvfp4_lm_head=auto builds at load from the same FP16 bits.
[[nodiscard]] std::expected<Fp8Head, std::string> quantize_head_fp8(const std::vector<uint16_t>& h_fp16,
                                                                    int64_t rows, int64_t cols);

// --lm-head fp8 for one source tensor: 0 = not the FP8 head (caller handles it), 1 = written to
// `out` (`store` owns the bytes until the shard is written; dry run counts bytes only), -1 = failed.
int emit_fp8_head(const RawTensor& t, bool dry_run, std::vector<Fp8Head>& store,
                  std::vector<SafeTensorsOut>& out, size_t& bytes_out);

}  // namespace imp::quantize
