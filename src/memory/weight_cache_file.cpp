#include "memory/weight_cache_file.h"
#include "core/logging.h"
#include "memory/weight_snapshot.h"
#include "model/model.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace imp {

namespace fs = std::filesystem;

namespace {

constexpr char kMagic[8] = {'I', 'M', 'P', 'W', 'C', 'A', 'C', 'H'};
constexpr uint32_t kVersion = 1;

struct FileHeader {
    char magic[8];
    uint32_t version;
    uint32_t tensor_pod_size;  // sizeof(Tensor) sanity guard across rebuilds
    int32_t arch_id;
    int32_t n_layers;
    uint64_t fp_total_bytes;
    int64_t fp_mtime_max;
    uint32_t record_count;
    uint32_t reserved = 0;
};

struct RecordHeader {
    uint32_t key_len;
    int32_t src_qtype;
    int64_t src_numel;
    uint64_t src_nbytes;
    int32_t data_alloc;
    uint64_t data_off;
    int32_t scales_alloc;
    uint64_t scales_off;
    uint32_t n_allocs;
};

int64_t mtime_seconds(const fs::directory_entry& e, std::error_code& ec) {
    auto t = e.last_write_time(ec);
    if (ec)
        return 0;
    return std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
}

}  // namespace

// Default cache directory: $XDG_CACHE_HOME/imp/warm, else $HOME/.cache/imp/warm,
// else /tmp/imp-warm-cache. Deliberately NOT next to the model: model mounts
// are often read-only for the serving user, and cache files should not clutter
// (or invalidate tooling checksums of) model directories.
static std::string default_warm_cache_dir() {
    if (const char* xdg = getenv("XDG_CACHE_HOME"); xdg && *xdg)
        return std::string(xdg) + "/imp/warm";
    if (const char* home = getenv("HOME"); home && *home)
        return std::string(home) + "/.cache/imp/warm";
    return "/tmp/imp-warm-cache";
}

std::string weight_cache_path_for(const std::string& model_path, const std::string& cache_dir) {
    std::error_code ec;
    std::string p = model_path;
    while (p.size() > 1 && p.back() == '/')
        p.pop_back();
    std::string dir = cache_dir.empty() ? default_warm_cache_dir() : cache_dir;
    while (dir.size() > 1 && dir.back() == '/')
        dir.pop_back();
    fs::create_directories(dir, ec);      // best-effort; write fails loudly later
    uint64_t h = 1469598103934665603ull;  // FNV-1a over the full model path
    for (unsigned char c : p) {
        h ^= c;
        h *= 1099511628211ull;
    }
    char hex[17];
    snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
    std::string base = fs::path(p).filename().string();
    return dir + "/" + base + "-" + hex + ".impwcache";
}

WeightCacheFingerprint weight_cache_fingerprint(const std::string& model_path) {
    WeightCacheFingerprint fp;
    std::error_code ec;
    if (fs::is_directory(model_path, ec)) {
        const std::string cache_name = ".imp_warm_cache";
        for (const auto& e : fs::recursive_directory_iterator(
                 model_path, fs::directory_options::skip_permission_denied, ec)) {
            if (ec)
                break;
            std::error_code fec;
            if (!e.is_regular_file(fec) || fec)
                continue;
            if (e.path().filename() == cache_name)
                continue;  // the cache must not fingerprint itself
            fp.total_bytes += e.file_size(fec);
            fp.mtime_max = std::max(fp.mtime_max, mtime_seconds(e, fec));
        }
    } else {
        fs::directory_entry e(model_path, ec);
        std::error_code fec;
        if (!ec && e.is_regular_file(fec) && !fec) {
            fp.total_bytes = e.file_size(fec);
            fp.mtime_max = mtime_seconds(e, fec);
        }
    }
    return fp;
}

bool weight_cache_write(const Model& model, const std::string& cache_path) {
    const WeightUploadLog* log = model.upload_log();
    if (!log || model.source_path().empty())
        return false;

    // Collect the transformed live records.
    std::vector<const WeightUploadRecord*> recs;
    size_t total = 0;
    for (const auto& rec : log->records()) {
        if (rec.dead || rec.raw_from_source)
            continue;
        recs.push_back(&rec);
        for (const auto& a : rec.allocs)
            total += a.bytes;
    }
    if (recs.empty()) {
        IMP_LOG_INFO("Warm cache: nothing to persist — every upload is raw-from-source");
        return false;
    }

    const WeightCacheFingerprint fp = weight_cache_fingerprint(model.source_path());

    // PID-unique tmp so concurrent processes loading the same model cannot
    // interleave writes; the rename below is atomic either way.
    const std::string tmp_path = cache_path + ".tmp." + std::to_string(getpid());
    std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        // Expected in containers where the model mount is read-only for the
        // serving user — point [warm_cache] dir at a writable volume to enable.
        IMP_LOG_INFO("Warm cache: %s not writable — skipping (set [warm_cache] dir to enable)",
                     tmp_path.c_str());
        return false;
    }

    FileHeader hdr{};
    memcpy(hdr.magic, kMagic, sizeof(kMagic));
    hdr.version = kVersion;
    hdr.tensor_pod_size = static_cast<uint32_t>(sizeof(Tensor));
    hdr.arch_id = static_cast<int32_t>(model.config().arch);
    hdr.n_layers = model.n_layers();
    hdr.fp_total_bytes = fp.total_bytes;
    hdr.fp_mtime_max = fp.mtime_max;
    hdr.record_count = static_cast<uint32_t>(recs.size());
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    // Retire outstanding device work once, then D2H each alloc via a reusable
    // host bounce buffer streamed straight to disk.
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        IMP_LOG_WARN("Warm cache: device sync failed (%s) — skipping write", cudaGetErrorString(sync));
        return false;
    }
    std::vector<uint8_t> bounce;
    for (const WeightUploadRecord* rec : recs) {
        RecordHeader rh{};
        rh.key_len = static_cast<uint32_t>(rec->key.size());
        rh.src_qtype = static_cast<int32_t>(rec->src_qtype);
        rh.src_numel = rec->src_numel;
        rh.src_nbytes = rec->src_nbytes;
        rh.data_alloc = rec->data_alloc;
        rh.data_off = rec->data_off;
        rh.scales_alloc = rec->scales_alloc;
        rh.scales_off = rec->scales_off;
        rh.n_allocs = static_cast<uint32_t>(rec->allocs.size());
        f.write(reinterpret_cast<const char*>(&rh), sizeof(rh));
        f.write(rec->key.data(), static_cast<std::streamsize>(rec->key.size()));
        // Post-upload tensor state verbatim; the two device pointers inside are
        // fixed up by try_restore at load. Guarded by tensor_pod_size + version.
        f.write(reinterpret_cast<const char*>(&rec->tensor), sizeof(Tensor));
        for (const auto& a : rec->allocs) {
            uint64_t bytes = a.bytes;
            f.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
            bounce.resize(a.bytes);
            if (cudaMemcpy(bounce.data(), a.ptr, a.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                IMP_LOG_WARN("Warm cache: D2H failed for %s — aborting write", rec->key.c_str());
                f.close();
                std::error_code ec;
                fs::remove(tmp_path, ec);
                return false;
            }
            f.write(reinterpret_cast<const char*>(bounce.data()), static_cast<std::streamsize>(a.bytes));
        }
    }
    f.flush();
    if (!f) {
        IMP_LOG_WARN("Warm cache: write to %s failed (disk full?) — skipping", tmp_path.c_str());
        f.close();
        std::error_code ec;
        fs::remove(tmp_path, ec);
        return false;
    }
    f.close();

    std::error_code ec;
    fs::rename(tmp_path, cache_path, ec);
    if (ec) {
        IMP_LOG_WARN("Warm cache: rename to %s failed: %s", cache_path.c_str(), ec.message().c_str());
        fs::remove(tmp_path, ec);
        return false;
    }
    IMP_LOG_INFO("Warm cache: persisted %zu transformed uploads (%.2f GiB) to %s", recs.size(),
                 total / (1024.0 * 1024.0 * 1024.0), cache_path.c_str());
    return true;
}

std::unique_ptr<WeightSnapshot> weight_cache_load(const std::string& cache_path, const Model& model) {
    int fd = open(cache_path.c_str(), O_RDONLY);
    if (fd < 0)
        return nullptr;  // no cache — the normal case on first load

    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(FileHeader))) {
        close(fd);
        return nullptr;
    }
    const size_t map_size = static_cast<size_t>(st.st_size);
    void* base = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // mapping keeps the file alive
    if (base == MAP_FAILED)
        return nullptr;
    // Kick off readahead: the blobs are consumed once, sequentially, during
    // the upload pass — overlap disk with the loader/H2D work.
    (void)madvise(base, map_size, MADV_WILLNEED);

    auto snap = std::make_unique<WeightSnapshot>();
    snap->builder_set_mmap(base, map_size);  // snapshot dtor munmaps on ALL exits below

    const uint8_t* cur = static_cast<const uint8_t*>(base);
    const uint8_t* end = cur + map_size;
    auto take = [&cur, end](void* dst, size_t n) {
        if (static_cast<size_t>(end - cur) < n)
            return false;
        memcpy(dst, cur, n);
        cur += n;
        return true;
    };

    FileHeader hdr{};
    if (!take(&hdr, sizeof(hdr)) || memcmp(hdr.magic, kMagic, sizeof(kMagic)) != 0 ||
        hdr.version != kVersion || hdr.tensor_pod_size != sizeof(Tensor)) {
        IMP_LOG_INFO("Warm cache: %s has an incompatible format — ignoring (cold load)",
                     cache_path.c_str());
        return nullptr;
    }
    if (hdr.arch_id != static_cast<int32_t>(model.config().arch) || hdr.n_layers != model.n_layers()) {
        IMP_LOG_INFO("Warm cache: %s does not match this model (arch/layers) — ignoring", cache_path.c_str());
        return nullptr;
    }
    const WeightCacheFingerprint now = weight_cache_fingerprint(model.source_path());
    if (now.total_bytes != hdr.fp_total_bytes || now.mtime_max != hdr.fp_mtime_max) {
        IMP_LOG_INFO("Warm cache: %s is stale (model file changed) — ignoring (cold load)",
                     cache_path.c_str());
        return nullptr;
    }

    for (uint32_t i = 0; i < hdr.record_count; ++i) {
        RecordHeader rh{};
        if (!take(&rh, sizeof(rh)) || rh.key_len == 0 || rh.key_len > 256 || rh.n_allocs == 0 ||
            rh.n_allocs > 16) {
            IMP_LOG_WARN("Warm cache: %s truncated/corrupt at record %u — ignoring (cold load)",
                         cache_path.c_str(), i);
            return nullptr;
        }
        WeightUploadRecord rec;
        rec.key.resize(rh.key_len);
        if (!take(rec.key.data(), rh.key_len) || !take(&rec.tensor, sizeof(Tensor))) {
            IMP_LOG_WARN("Warm cache: %s truncated in record %u — ignoring", cache_path.c_str(), i);
            return nullptr;
        }
        rec.src_qtype = static_cast<QType>(rh.src_qtype);
        rec.src_numel = rh.src_numel;
        rec.src_nbytes = rh.src_nbytes;
        rec.data_alloc = rh.data_alloc;
        rec.data_off = rh.data_off;
        rec.scales_alloc = rh.scales_alloc;
        rec.scales_off = rh.scales_off;

        std::vector<const uint8_t*> views;
        views.reserve(rh.n_allocs);
        for (uint32_t a = 0; a < rh.n_allocs; ++a) {
            uint64_t bytes = 0;
            if (!take(&bytes, sizeof(bytes)) || bytes == 0 ||
                bytes > static_cast<uint64_t>(end - cur)) {
                IMP_LOG_WARN("Warm cache: %s corrupt alloc size at record %u — ignoring",
                             cache_path.c_str(), i);
                return nullptr;
            }
            views.push_back(cur);  // zero-copy: blob stays in the mapping
            cur += bytes;
            rec.allocs.push_back({nullptr, static_cast<size_t>(bytes)});  // device ptr set at restore
        }
        snap->builder_add_views(std::move(rec), std::move(views));
    }

    IMP_LOG_INFO("Warm cache: mapped %zu transformed uploads (%.2f GiB) from %s", snap->record_count(),
                 snap->total_bytes() / (1024.0 * 1024.0 * 1024.0), cache_path.c_str());
    return snap;
}

}  // namespace imp
