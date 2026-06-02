#pragma once

#include "model/model.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace imp {

// Load model from SafeTensors format.
// path can be:
//   - A single .safetensors file
//   - A directory containing .safetensors files (+ config.json, etc.)
//
// load_mtp_head: when true, a `model_mtp.safetensors` sidecar (DeepSeek-V3
// family, e.g. Qwen3.6) is parsed and retained (~1.57 GiB BF16 on Qwen3.6,
// later uploaded to VRAM). Default false: the sidecar is skipped entirely, so
// the model behaves as if it had no MTP head (spec-decode unavailable). Only
// callers that actually intend to enable MTP spec-decode should pass true —
// the head is wasted VRAM otherwise (server + normal CLI never use it).
std::unique_ptr<Model> load_safetensors(const std::string& path, bool load_mtp_head = false);

// ---- Test-visible validation helpers ----
//
// These mirror the production validation rules in load_shard()
// (safetensors_loader.cpp). They are exposed so unit tests can drive them
// directly with synthetic blobs without constructing a full Model.
namespace safetensors_internal {

// Maximum legal SafeTensors JSON header size. Real models have headers below
// 1 MiB; 128 MiB is far above legitimate use and far below pathological
// inputs that would force the JSON parser to scan multi-GB regions. See
// docs/audit/decisions/0002-header-size-cap.md.
constexpr uint64_t kMaxHeaderBytes = 128ULL * 1024ULL * 1024ULL;

// Validate the SafeTensors 8-byte header_size prefix in isolation. Returns
// true if (8 + header_size <= file_size) and (header_size <= kMaxHeaderBytes).
// Overflow-safe: works for file_size >= 8 and any uint64_t header_size.
//
// `err` (when non-null) is populated with a one-line reason on failure.
bool validate_header_size(uint64_t file_size, uint64_t declared_header_size, std::string* err);

// Validate per-tensor data_offsets [start, end) against the file/header layout.
// `expected_nbytes` is the byte count implied by the tensor's shape × dtype
// width (for SafeTensors wire dtypes — none of which are block-quantised).
// `tensor_data_offset` is `8 + header_size` (start of the tensor data block).
//
// Three rules:
//   1. offset_start <= offset_end                                   (no swap)
//   2. tensor_data_offset + offset_end <= file_size                 (no OOB)
//   3. offset_end - offset_start == expected_nbytes                 (size match)
//
// Overflow-safe; uses subtractions in the safe ordering.
//
// `err` (when non-null) is populated with the first violated rule's reason.
bool validate_tensor_offsets(uint64_t offset_start, uint64_t offset_end,
                             uint64_t expected_nbytes, uint64_t tensor_data_offset,
                             uint64_t file_size, std::string* err);

}  // namespace safetensors_internal

}  // namespace imp
