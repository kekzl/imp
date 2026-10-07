#pragma once

// Host expert tier: an LRU of (layer, projection, expert) units in one host arena (pinned and
// mapped in production), filled on demand from the checkpoint file with parallel pread.
// Serves host-resident NVFP4 experts when host RAM cannot pin all of them: the device expert
// cache gathers from a unit's device view; a unit missing here is read from disk first.
// Pure host code (no CUDA): the caller owns the arena and its device view.

#include "memory/host_task_pool.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace imp {

// File backing of a host pointer, from /proc/self/maps. fd < 0: not file-backed.
struct HostFileSource {
    int fd_direct = -1;    // O_DIRECT, -1 when the filesystem refuses it
    int fd_buffered = -1;  // plain O_RDONLY
    uint64_t offset = 0;   // file offset of the pointer
};

// Resolves host pointers to (file, offset) through the process's file mappings.
class HostFileMap {
public:
    HostFileMap() = default;
    ~HostFileMap();
    HostFileMap(const HostFileMap&) = delete;
    HostFileMap& operator=(const HostFileMap&) = delete;

    // Parses /proc/self/maps (or `maps_text` when given, for tests).
    void load(const std::string* maps_text = nullptr);
    [[nodiscard]] HostFileSource resolve(const void* p);

private:
    struct Region {
        uintptr_t start, end;
        uint64_t file_off;
        std::string path;
    };
    struct Fds {
        std::string path;
        int direct, buffered;
    };
    std::vector<Region> regions_;  // sorted by start
    std::vector<Fds> fds_;
};

class HostExpertTier {
public:
    struct Src {
        const void* packed = nullptr;
        size_t packed_bytes = 0;
        const void* ms = nullptr;
        size_t ms_bytes = 0;
    };
    struct UnitView {
        const char* packed;  // device views of the unit's bytes
        const char* ms;
    };
    struct Stats {
        uint64_t hits = 0, misses = 0, bytes_read = 0, direct_reads = 0, buffered_reads = 0, copies = 0;
        double read_ms = 0;
    };

    // Bytes one unit needs for a source of `packed_bytes` + `ms_bytes`, with O_DIRECT slack.
    static size_t unit_bytes(size_t packed_bytes, size_t ms_bytes);

    // `host`/`dev` are the same arena in two address spaces (equal pointers without a device),
    // carved into floor(bytes / unit) units; the last `scratch_units` serve transient acquires only.
    // `n_keys` dense keys, sources set per key below.
    HostExpertTier(char* host, char* dev, size_t bytes, size_t unit, int n_keys, int io_threads = 16,
                   int scratch_units = 0);
    HostExpertTier(const HostExpertTier&) = delete;
    HostExpertTier& operator=(const HostExpertTier&) = delete;

    void set_source(int key, const Src& src) { src_[key] = src; }
    int units() const { return n_lru_; }  // LRU units (scratch excluded)

    // Makes every key in keys[0..n) resident and writes its view to out[i]. Keys of one call are
    // never evicted by that call; duplicates are allowed. Missing units load in parallel.
    // False (out untouched) when more distinct keys than units are asked for. `transient`
    // (prefill staging): resident keys are used as they are, missing ones load into scratch units
    // without entering the LRU, so a scan over every expert does not evict the decode set.
    // Scratch views stay valid until the next transient acquire.
    [[nodiscard]] bool acquire(const int* keys, int n, UnitView* out, bool transient = false);

    // Resolves sources to file ranges (once; acquire() calls it lazily).
    void bind_files(const std::string* maps_text = nullptr);
    Stats stats() const;

private:
    void load_(int unit, int key);
    void touch_(int unit);
    void unlink_(int unit);

    char* host_;
    char* dev_;
    size_t unit_;
    int n_units_;
    int n_lru_;  // units [0, n_lru_) are the LRU, [n_lru_, n_units_) scratch
    std::vector<Src> src_;
    std::vector<HostFileSource> fsrc_packed_, fsrc_ms_;
    std::vector<int32_t> unit_of_;               // [n_keys], -1 = not resident
    std::vector<int32_t> key_of_;                // [n_units], -1 = free
    std::vector<int32_t> prev_, next_;           // LRU list, head_ = most recent
    std::vector<uint32_t> epoch_;                // [n_units], acquire() call that last used the unit
    std::vector<uint32_t> packed_off_, ms_off_;  // [n_units] data offsets inside the unit
    int head_ = -1, tail_ = -1;
    uint32_t epoch_now_ = 0;
    bool files_bound_ = false;
    HostFileMap files_;
    HostTaskPool pool_;
    mutable std::mutex mu_;        // acquire() is serialized: decode host nodes and prefill staging
    mutable std::mutex stats_mu_;  // load_() runs on pool threads
    Stats stats_;
};

}  // namespace imp
