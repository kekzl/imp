#include "memory/weight_snapshot.h"
#include "core/logging.h"
#include "model/model.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace imp {

const char* make_weight_key(std::span<char> buf, const char* name, int layer) {
    if (layer >= 0)
        snprintf(buf.data(), buf.size(), "L%d.%s", layer, name);
    else if (layer == kMtpKeyLayer)
        snprintf(buf.data(), buf.size(), "mtp.%s", name);
    else
        snprintf(buf.data(), buf.size(), "%s", name);
    return buf.data();
}

WeightSnapshot::~WeightSnapshot() {
#ifdef __linux__
    if (mmap_base_ && mmap_size_ > 0)
        munmap(mmap_base_, mmap_size_);
#endif
}

// ---------------------------------------------------------------------------
// WeightUploadLog
// ---------------------------------------------------------------------------

void WeightUploadLog::record(const char* key, const void* const* allocs, size_t n_allocs,
                             const Tensor& post, QType src_qtype, int64_t src_numel,
                             size_t src_nbytes) {
    if (!key || n_allocs == 0)
        return;

    WeightUploadRecord rec;
    rec.key = key;
    rec.allocs.reserve(n_allocs);
    for (size_t i = 0; i < n_allocs; ++i) {
        void* p = const_cast<void*>(allocs[i]);
        auto it = alloc_sizes_.find(p);
        if (it == alloc_sizes_.end())
            return;  // size unknown (raw cudaMallocAsync site) — leave cold-only
        rec.allocs.push_back({p, it->second});
    }

    // Resolve which alloc backs .data / .scales (interval check — .data may be
    // an interior pointer, e.g. a per-expert view into a flat upload).
    auto locate = [&rec](const void* p, int& idx, size_t& off) {
        if (!p)
            return true;  // nothing to locate
        for (size_t i = 0; i < rec.allocs.size(); ++i) {
            const char* base = static_cast<const char*>(rec.allocs[i].ptr);
            const char* q = static_cast<const char*>(p);
            if (q >= base && q < base + rec.allocs[i].bytes) {
                idx = static_cast<int>(i);
                off = static_cast<size_t>(q - base);
                return true;
            }
        }
        return false;
    };
    if (!locate(post.data, rec.data_alloc, rec.data_off))
        return;  // data lives outside this upload's allocations — leave cold-only
    if (!locate(post.scales, rec.scales_alloc, rec.scales_off))
        return;

    rec.tensor = post;
    rec.src_qtype = src_qtype;
    rec.src_numel = src_numel;
    rec.src_nbytes = src_nbytes;
    // See the field comment: heuristic only gates warm-cache persistence.
    rec.raw_from_source = rec.allocs.size() == 1 && post.qtype == src_qtype &&
                          src_qtype != QType::MXFP4 && post.scales == nullptr &&
                          rec.allocs[0].bytes == src_nbytes;

    auto it = by_key_.find(rec.key);
    if (it != by_key_.end()) {
        records_[it->second] = std::move(rec);  // re-upload of the same slot: replace
    } else {
        by_key_.emplace(rec.key, records_.size());
        records_.push_back(std::move(rec));
    }
}

void WeightUploadLog::evict_ptr(void* ptr) {
    if (!ptr)
        return;
    for (auto& rec : records_) {
        if (rec.dead)
            continue;
        for (const auto& a : rec.allocs) {
            if (a.ptr == ptr) {
                rec.dead = true;
                break;
            }
        }
    }
}

size_t WeightUploadLog::live_bytes() const {
    size_t total = 0;
    for (const auto& rec : records_) {
        if (rec.dead)
            continue;
        for (const auto& a : rec.allocs)
            total += a.bytes;
    }
    return total;
}

// ---------------------------------------------------------------------------
// /proc/meminfo
// ---------------------------------------------------------------------------

size_t parse_meminfo_available(std::string_view meminfo_text) {
    // Line format: "MemAvailable:   123456 kB"
    size_t pos = meminfo_text.find("MemAvailable:");
    if (pos == std::string_view::npos)
        return 0;
    const char* p = meminfo_text.data() + pos + strlen("MemAvailable:");
    const char* end = meminfo_text.data() + meminfo_text.size();
    while (p < end && (*p == ' ' || *p == '\t'))
        ++p;
    size_t kb = 0;
    bool any = false;
    while (p < end && *p >= '0' && *p <= '9') {
        kb = kb * 10 + static_cast<size_t>(*p - '0');
        ++p;
        any = true;
    }
    return any ? kb * 1024 : 0;
}

size_t parse_cgroup_headroom(std::string_view limit_text, std::string_view current_text,
                             std::string_view stat_text) {
    auto number = [](std::string_view s) -> size_t {
        size_t v = 0;
        bool any = false;
        for (char c : s) {
            if (c >= '0' && c <= '9') {
                v = v * 10 + static_cast<size_t>(c - '0');
                any = true;
            } else if (any) {
                break;
            }
        }
        return any ? v : 0;
    };
    // memory.current counts page cache, and a model load fills it with the mmap'd
    // checkpoint: a 70 GiB container reads 0.45 GiB headroom while 68.9 of its 70 GiB are
    // file pages the kernel drops on demand. Count them as free, the way MemAvailable does
    // on the host. The cost of being wrong here is re-reading the checkpoint, not an OOM,
    // because anonymous and pinned pages are what the limit actually binds.
    auto field = [&](std::string_view key) -> size_t {
        size_t pos = stat_text.find(key);
        while (pos != std::string_view::npos && pos != 0 && stat_text[pos - 1] != '\n')
            pos = stat_text.find(key, pos + 1);
        if (pos == std::string_view::npos)
            return 0;
        return number(stat_text.substr(pos + key.size()));
    };
    // cgroup v2 writes the literal "max" for "no limit"; v1 writes a number so large it
    // means the same. Either way there is no cgroup ceiling to report.
    if (limit_text.find("max") != std::string_view::npos)
        return 0;
    const size_t limit = number(limit_text);
    if (limit == 0 || limit >= (1ULL << 50))
        return 0;
    const size_t current = number(current_text);
    const size_t reclaimable = field("file ");  // cgroup v2; absent on v1, which then reads 0
    const size_t used = current > reclaimable ? current - reclaimable : 0;
    return limit > used ? limit - used : 1;  // 1, not 0: 0 means "no ceiling"
}

namespace {
std::string read_small_file(const char* path) {
    std::ifstream f(path);
    if (!f)
        return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
}  // namespace

// MemAvailable is the HOST's figure even inside a memory-capped container: a container
// limited to 8 GiB on a 78 GiB host reads 76 GiB free and pins until the OOM killer
// stops it. The cgroup ceiling is the real one, so report the smaller of the two.
size_t host_mem_available_bytes() {
    const std::string meminfo = read_small_file("/proc/meminfo");
    if (meminfo.empty())
        return 0;
    const size_t from_meminfo = parse_meminfo_available(meminfo);
    const size_t from_cgroup =
        parse_cgroup_headroom(read_small_file("/sys/fs/cgroup/memory.max"),
                              read_small_file("/sys/fs/cgroup/memory.current"),
                              read_small_file("/sys/fs/cgroup/memory.stat"));
    const size_t from_cgroup_v1 =
        from_cgroup ? 0
                    : parse_cgroup_headroom(
                          read_small_file("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                          read_small_file("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
                          read_small_file("/sys/fs/cgroup/memory/memory.stat"));
    const size_t capped = from_cgroup ? from_cgroup : from_cgroup_v1;
    if (capped == 0)
        return from_meminfo;
    return (from_meminfo && from_meminfo < capped) ? from_meminfo : capped;
}

// ---------------------------------------------------------------------------
// WeightSnapshot
// ---------------------------------------------------------------------------

std::unique_ptr<WeightSnapshot> WeightSnapshot::capture(const Model& model,
                                                        size_t host_ram_headroom_bytes) {
    const WeightUploadLog* log = model.upload_log();
    if (!log || !model.gpu_weights_ready())
        throw SnapshotUnsupportedError("weight snapshot: model has no completed GPU weight upload");
    if (model.device_sources_mutated())
        throw SnapshotUnsupportedError(
            "weight snapshot: unsupported model — device weight buffers were transformed "
            "in place after upload (e.g. native MXFP4 unpack); resume would corrupt them");

    const size_t need = log->live_bytes();
    const size_t avail = host_mem_available_bytes();
    if (avail > 0 && need + host_ram_headroom_bytes > avail) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "weight snapshot: insufficient host RAM (need %.2f GiB + %.2f GiB headroom, "
                 "MemAvailable %.2f GiB)",
                 need / (1024.0 * 1024.0 * 1024.0),
                 host_ram_headroom_bytes / (1024.0 * 1024.0 * 1024.0),
                 avail / (1024.0 * 1024.0 * 1024.0));
        throw SnapshotHostOomError(msg);
    }

    // Retire all outstanding device work before reading weight buffers.
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess)
        throw std::runtime_error(std::string("weight snapshot: device sync failed: ") +
                                 cudaGetErrorString(sync));

    auto snap = std::make_unique<WeightSnapshot>();
    snap->arch_id_ = static_cast<int>(model.config().arch);
    snap->n_layers_ = model.n_layers();

    int captured = 0, skipped_dead = 0;
    for (const auto& rec : log->records()) {
        if (rec.dead) {
            skipped_dead++;
            continue;
        }
        Blob blob;
        blob.rec = rec;
        blob.owned.reserve(rec.allocs.size());
        blob.views.reserve(rec.allocs.size());
        for (const auto& a : rec.allocs) {
            // Default-initialized (no memset) — filled entirely by the D2H copy.
            std::unique_ptr<uint8_t[]> buf(new uint8_t[a.bytes]);
            cudaError_t err = cudaMemcpy(buf.get(), a.ptr, a.bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                char msg[160];
                snprintf(msg, sizeof(msg), "weight snapshot: D2H copy failed for %s (%zu bytes): %s",
                         rec.key.c_str(), a.bytes, cudaGetErrorString(err));
                throw std::runtime_error(msg);
            }
            snap->total_bytes_ += a.bytes;
            blob.views.push_back(buf.get());
            blob.owned.push_back(std::move(buf));
        }
        snap->blobs_.emplace(rec.key, std::move(blob));
        captured++;
    }
    if (captured == 0)
        throw SnapshotUnsupportedError("weight snapshot: nothing capturable (all sources dropped)");

    IMP_LOG_INFO("Weight snapshot: captured %d uploads (%.2f GiB) to host RAM%s", captured,
                 snap->total_bytes_ / (1024.0 * 1024.0 * 1024.0),
                 skipped_dead ? " (dropped sources excluded — they re-upload cold at resume)" : "");
    return snap;
}

bool WeightSnapshot::matches(const Model& model) const {
    return arch_id_ == static_cast<int>(model.config().arch) && n_layers_ == model.n_layers();
}

bool WeightSnapshot::try_restore(const char* key, Tensor& weight, cudaStream_t stream,
                                 std::vector<void*>& gpu_allocs, const WarmRestoreOps& ops,
                                 WeightUploadLog* new_log) {
    if (!key || !ops.alloc || !ops.copy_h2d)
        return false;
    auto it = blobs_.find(key);
    if (it == blobs_.end())
        return false;
    const Blob& blob = it->second;
    const WeightUploadRecord& rec = blob.rec;
    if (rec.dead)
        return false;
    // Source identity must match what the record was made from.
    if (weight.qtype != rec.src_qtype || weight.numel() != rec.src_numel)
        return false;

    std::vector<void*> new_allocs;
    new_allocs.reserve(rec.allocs.size());
    for (size_t i = 0; i < rec.allocs.size(); ++i) {
        void* p = nullptr;
        if (ops.alloc(&p, rec.allocs[i].bytes, stream) != cudaSuccess || !p) {
            for (void* q : new_allocs)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(q, stream));
            return false;  // out of VRAM — cold path will make its own call
        }
        if (ops.copy_h2d(p, blob.views[i], rec.allocs[i].bytes, stream) != cudaSuccess) {
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(p, stream));
            for (void* q : new_allocs)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(q, stream));
            return false;
        }
        new_allocs.push_back(p);
    }

    for (void* p : new_allocs)
        gpu_allocs.push_back(p);

    weight = rec.tensor;
    weight.data = static_cast<char*>(new_allocs[rec.data_alloc]) + rec.data_off;
    if (rec.scales_alloc >= 0)
        weight.scales = static_cast<char*>(new_allocs[rec.scales_alloc]) + rec.scales_off;

    if (new_log) {
        // note_alloc happened inside ops.alloc (checked_cuda_malloc); re-record
        // so a subsequent suspend of the resumed model captures this slot again.
        new_log->record(key, const_cast<const void* const*>(new_allocs.data()), new_allocs.size(),
                        weight, rec.src_qtype, rec.src_numel, rec.src_nbytes);
    }
    hits_++;
    return true;
}

// ---------------------------------------------------------------------------
// Pending-arm slot
// ---------------------------------------------------------------------------

static WeightSnapshot* g_armed_snapshot = nullptr;

void weight_snapshot_arm(WeightSnapshot* snap) { g_armed_snapshot = snap; }

WeightSnapshot* weight_snapshot_take_armed() {
    WeightSnapshot* s = g_armed_snapshot;
    g_armed_snapshot = nullptr;
    return s;
}

void weight_snapshot_disarm(const WeightSnapshot* snap) {
    if (g_armed_snapshot == snap)
        g_armed_snapshot = nullptr;
}

}  // namespace imp
