#pragma once
// diagnostics.vram_touch_audit: which device bytes were written between two points (after warmup, at
// teardown). A 64 KiB page whose hash did not change was not written in between: capacity the
// workload never used, or read-only data (weights). Covers VRAMAllocator allocations, Backend regions
// (committed chunks) and MemAccount::note_alloc rows.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

class VRAMAllocator;

class TouchAudit {
public:
    static constexpr size_t kPage = size_t{64} * 1024;
    TouchAudit() = default;
    TouchAudit(const TouchAudit&) = delete;
    TouchAudit& operator=(const TouchAudit&) = delete;
    ~TouchAudit();

    // The audit with its snapshot taken, or null: off, or a device error.
    [[nodiscard]] static std::unique_ptr<TouchAudit> start(bool on, const VRAMAllocator& va);
    // Hashes every live allocation of va. False = a device error, the audit stays off.
    [[nodiscard]] bool snapshot(const VRAMAllocator& va);
    struct Row {
        std::string tag;
        size_t bytes = 0;    // allocations alive at both points
        size_t written = 0;  // of those: bytes in pages whose hash changed
        size_t fresh = 0;    // allocated after snapshot()
        int allocs = 0;
    };
    // One row per tag, most unwritten bytes first. Empty without a snapshot.
    [[nodiscard]] std::vector<Row> compare(const VRAMAllocator& va);
    // compare() as log lines.
    void report(const VRAMAllocator& va, const char* label);

private:
    struct Entry {
        size_t bytes = 0;
        std::vector<uint64_t> pages;
    };
    [[nodiscard]] bool hash_(const void* p, size_t bytes, std::vector<uint64_t>& out);

    std::unordered_map<const void*, Entry> snap_;
    uint64_t* d_out_ = nullptr;
    size_t d_cap_ = 0;  // pages d_out_ holds
};

}  // namespace imp
