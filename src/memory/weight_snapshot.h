#pragma once

// Suspend-to-RAM weight snapshot (see docs: /admin/suspend).
//
// Two pieces:
//
//   WeightUploadLog   — metadata-only log filled while Model::upload_weights_gpu
//                       runs: one record per keyed weight upload = the device
//                       allocation(s) backing it (pointer + byte size) plus the
//                       post-upload Tensor state. No weight bytes are copied at
//                       upload time; cost is a few thousand small host entries.
//
//   WeightSnapshot    — built at suspend by WeightSnapshot::capture(): D2H-copies
//                       every live logged allocation into pageable host memory.
//                       Survives full model/engine teardown (deliberately NOT
//                       cudaHostAlloc so it also survives cudaDeviceReset). At
//                       resume it is "armed" into a process-global slot; the next
//                       upload_weights_gpu consults it per key and restores buffer
//                       bytes instead of re-running mmap-read + host/GPU
//                       conversion. Any miss or mismatch falls back to the normal
//                       cold path per tensor — restored bytes are byte-identical
//                       to a cold upload, so warm and cold tensors mix freely.
//
// Keying is by canonical upload-site strings ("L{layer}.{slot}", "tok_emb",
// "mtp.{name}", "L{i}.expert_gate_exps", ...) — never by pointer or allocation
// order, both of which are unstable across a teardown.

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace imp {

class Model;

// Thrown by WeightSnapshot::capture so the API boundary can map to
// IMP_ERROR_UNSUPPORTED / IMP_ERROR_OUT_OF_MEMORY respectively.
struct SnapshotUnsupportedError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct SnapshotHostOomError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Formats the canonical key for an upload site into buf and returns buf.
//   layer >= 0            -> "L{layer}.{name}"
//   layer == kMtpKeyLayer -> "mtp.{name}"
//   otherwise             -> "{name}"
inline constexpr int kMtpKeyLayer = -2;
const char* make_weight_key(char* buf, size_t buf_len, const char* name, int layer);

struct WeightUploadRecord {
    std::string key;
    struct Alloc {
        void* ptr = nullptr;
        size_t bytes = 0;
    };
    std::vector<Alloc> allocs;  // device allocations, in gpu_allocations_ push order
    Tensor tensor;              // post-upload tensor state (device pointers inside)
    // Which alloc (and byte offset) backs tensor.data / tensor.scales.
    // scales_alloc == -1 means tensor.scales was not set by the upload handler.
    int data_alloc = -1;
    size_t data_off = 0;
    int scales_alloc = -1;
    size_t scales_off = 0;
    // Pre-upload source identity, checked at restore so a swapped model file
    // (or divergent load config) falls back to the cold path per tensor.
    QType src_qtype = QType::NONE;
    int64_t src_numel = 0;
    // Source dropped by the pipeline (release_gpu_allocation) — not capturable.
    bool dead = false;
};

class WeightUploadLog {
public:
    // Called from checked_cuda_malloc so record() can resolve byte sizes.
    void note_alloc(void* ptr, size_t bytes) { alloc_sizes_[ptr] = bytes; }

    // Associate the gpu_allocations_ delta of one upload with a key. Skips
    // silently (leaving the tensor cold-only) if any size is unknown or the
    // tensor data pointer is not backed by the listed allocations.
    void record(const char* key, const void* const* allocs, size_t n_allocs, const Tensor& post,
                QType src_qtype, int64_t src_numel);

    // Mark every record touching ptr dead (pipeline dropped the source).
    void evict_ptr(void* ptr);

    const std::vector<WeightUploadRecord>& records() const { return records_; }
    size_t live_bytes() const;  // sum of alloc bytes over non-dead records

private:
    std::unordered_map<void*, size_t> alloc_sizes_;
    std::vector<WeightUploadRecord> records_;
    std::unordered_map<std::string, size_t> by_key_;
};

// Restore-side CUDA hooks: routed back into weight_upload.cu's checked
// allocator (VRAM budget accounting) and staged H2D copy (PinnedStager).
struct WarmRestoreOps {
    cudaError_t (*alloc)(void** ptr, size_t bytes, cudaStream_t stream) = nullptr;
    cudaError_t (*copy_h2d)(void* dst, const void* src, size_t bytes, cudaStream_t stream) = nullptr;
};

class WeightSnapshot {
public:
    // D2H-copies all live logged allocations of `model` into pageable host
    // memory. Throws std::runtime_error with a clear message when the model is
    // unsupported (no upload log, device sources mutated in place) or host RAM
    // is insufficient (MemAvailable < snapshot bytes + headroom).
    static std::unique_ptr<WeightSnapshot> capture(const Model& model, size_t host_ram_headroom_bytes);

    // Restore one keyed upload: allocate device buffers, copy the blob back,
    // reproduce the post-upload tensor state, append the allocations to
    // gpu_allocs, and re-record into new_log (so a later suspend works).
    // Returns false (and leaves `weight` untouched) on miss/mismatch — the
    // caller proceeds with the normal cold upload.
    bool try_restore(const char* key, Tensor& weight, cudaStream_t stream,
                     std::vector<void*>& gpu_allocs, const WarmRestoreOps& ops,
                     WeightUploadLog* new_log);

    size_t total_bytes() const { return total_bytes_; }
    int hits() const { return hits_; }
    // Model-identity guard checked once by upload_weights_gpu before use.
    bool matches(const Model& model) const;
    // Test hook: drop one captured record so that key takes the cold path at
    // resume — proves warm and cold uploads mix byte-safely.
    bool drop_key(const std::string& key) { return blobs_.erase(key) > 0; }

private:
    struct Blob {
        WeightUploadRecord rec;                            // pointers inside are STALE (old device)
        std::vector<std::unique_ptr<uint8_t[]>> host_data;  // one buffer per rec.allocs entry
    };
    std::unordered_map<std::string, Blob> blobs_;
    size_t total_bytes_ = 0;
    int arch_id_ = -1;
    int n_layers_ = -1;
    int hits_ = 0;
};

// Process-global pending slot (same idiom as set_pending_runtime_config):
// the server arms a snapshot it owns, the next Model::upload_weights_gpu
// takes it for the duration of that one upload. Non-owning.
void weight_snapshot_arm(WeightSnapshot* snap);
WeightSnapshot* weight_snapshot_take_armed();
// Clear the armed slot iff it points at snap (called before destroying one).
void weight_snapshot_disarm(const WeightSnapshot* snap);

// /proc/meminfo MemAvailable in bytes (0 if unavailable). The parser is
// separated for unit testing.
size_t host_mem_available_bytes();
size_t parse_meminfo_available(std::string_view meminfo_text);

}  // namespace imp
