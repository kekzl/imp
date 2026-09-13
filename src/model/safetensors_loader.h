#pragma once

#include "model/model.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace imp {

// path: a single .safetensors file, or a directory (+config.json etc). load_mtp_head=true
// parses+retains the model_mtp.safetensors sidecar (~1.57 GiB BF16 on Qwen3.6, later
// uploaded). Default false skips it entirely (spec-decode unavailable); only pass true when
// MTP is actually going to be enabled, the head is wasted VRAM otherwise.
std::unique_ptr<Model> load_safetensors(const std::string& path, bool load_mtp_head = false);

// True when a model.safetensors.index.json shard name is a bare filename next to the
// index. Names are file content; a separator escapes the model directory into an
// arbitrary open()/mmap() (#1612). Exposed for the test; the loader calls it first.
bool safetensors_shard_name_is_safe(const std::string& name);

// Mirrors the production validation rules in load_shard() (safetensors_loader.cpp),
// exposed so unit tests can drive them with synthetic blobs without a full Model.
namespace safetensors_internal {

// Maximum legal SafeTensors JSON header size. Real models stay below 1 MiB; 128 MiB is far
// above legitimate use and far below what would force the JSON parser to scan multi-GB
// regions. See docs/audit/decisions/0002-header-size-cap.md.
constexpr uint64_t kMaxHeaderBytes = 128ULL * 1024ULL * 1024ULL;

// Validates the 8-byte header_size prefix in isolation: true iff 8+header_size<=file_size
// and header_size<=kMaxHeaderBytes. Overflow-safe for file_size>=8 and any uint64_t
// header_size. `err`, when non-null, gets a one-line failure reason.
bool validate_header_size(uint64_t file_size, uint64_t declared_header_size, std::string* err);

// Validates per-tensor data_offsets [start,end) against the file/header layout.
// expected_nbytes = shape x dtype width (no SafeTensors wire dtype is block-quantised).
// tensor_data_offset = 8+header_size. Three rules:
//   1. offset_start <= offset_end (no swap)
//   2. tensor_data_offset + offset_end <= file_size (no OOB)
//   3. offset_end - offset_start == expected_nbytes (size match)
// Overflow-safe, subtractions in the safe ordering.
bool validate_tensor_offsets(uint64_t offset_start, uint64_t offset_end,
                             uint64_t expected_nbytes, uint64_t tensor_data_offset,
                             uint64_t file_size, std::string* err);

// One row of the SafeTensors wire-dtype table (#1604). wire_bytes is the on-disk element
// width, qtype what it's mapped to; for a servable row the two widths MUST agree, or the
// loader validates one width and the consumer reads another. Exposed so a test can assert
// the equality across the whole table.
struct DtypeTableRow {
    std::string name;
    size_t wire_bytes;
    QType qtype;
    bool servable;
};

size_t dtype_table_size();
DtypeTableRow dtype_table_row(size_t i);

}  // namespace safetensors_internal

}  // namespace imp
