#include "memory/host_expert_tier.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace imp {

namespace {

constexpr size_t kAlign = 4096;  // O_DIRECT offset/length/buffer alignment

size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Reads `bytes` at file offset `off` into the region at `dst` (region_bytes long, kAlign-aligned).
// Returns the data offset inside the region (16-aligned), or -1. O_DIRECT reads the aligned
// superset (a lead that is not 16-aligned is moved to the region start); the
// buffered fallback reads exactly to the region start.
long read_range(const HostFileSource& f, uint64_t off, size_t bytes, char* dst, size_t region_bytes,
                bool* direct) {
    *direct = false;
    if (f.fd_direct >= 0 && reinterpret_cast<uintptr_t>(dst) % kAlign == 0) {
        const uint64_t a0 = off / kAlign * kAlign;
        const size_t lead = static_cast<size_t>(off - a0);
        const size_t len = round_up(lead + bytes, kAlign);
        if (len <= region_bytes) {
            size_t got = 0;
            while (got < lead + bytes) {
                const ssize_t r = pread(f.fd_direct, dst + got, len - got, static_cast<off_t>(a0 + got));
                if (r <= 0)
                    break;
                got += static_cast<size_t>(r);
                if (got % kAlign != 0)
                    break;  // short read at EOF: whatever arrived is all there is
            }
            if (got >= lead + bytes) {
                *direct = true;
                if (lead % 16 == 0)
                    return static_cast<long>(lead);
                std::memmove(dst, dst + lead, bytes);  // the gather loads 16-byte vectors
                return 0;
            }
        }
    }
    if (f.fd_buffered < 0 || bytes > region_bytes)
        return -1;
    size_t got = 0;
    while (got < bytes) {
        const ssize_t r = pread(f.fd_buffered, dst + got, bytes - got, static_cast<off_t>(off + got));
        if (r <= 0)
            return -1;
        got += static_cast<size_t>(r);
    }
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// HostFileMap
// ---------------------------------------------------------------------------

HostFileMap::~HostFileMap() {
    for (const Fds& f : fds_) {
        if (f.direct >= 0)
            close(f.direct);
        if (f.buffered >= 0)
            close(f.buffered);
    }
}

void HostFileMap::load(const std::string* maps_text) {
    regions_.clear();
    std::string text;
    if (maps_text) {
        text = *maps_text;
    } else {
        std::ifstream in("/proc/self/maps");
        std::stringstream ss;
        ss << in.rdbuf();
        text = ss.str();
    }
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        unsigned long long s = 0, e = 0, off = 0;
        char perms[8] = {};
        int n = 0;
        if (std::sscanf(line.c_str(), "%llx-%llx %7s %llx %*s %*s %n", &s, &e, perms, &off, &n) < 4 || n <= 0)
            continue;
        std::string path = line.substr(static_cast<size_t>(n));
        while (!path.empty() && (path.back() == ' ' || path.back() == '\n'))
            path.pop_back();
        if (path.empty() || path[0] != '/' || path.find(" (deleted)") != std::string::npos)
            continue;
        regions_.push_back(
            Region{static_cast<uintptr_t>(s), static_cast<uintptr_t>(e), off, std::move(path)});
    }
    std::sort(regions_.begin(), regions_.end(),
              [](const Region& a, const Region& b) { return a.start < b.start; });
}

HostFileSource HostFileMap::resolve(const void* p) {
    const auto u = reinterpret_cast<uintptr_t>(p);
    auto it = std::upper_bound(regions_.begin(), regions_.end(), u,
                               [](uintptr_t v, const Region& r) { return v < r.start; });
    if (it == regions_.begin())
        return {};
    --it;
    if (u >= it->end)
        return {};
    auto f = std::find_if(fds_.begin(), fds_.end(), [&](const Fds& x) { return x.path == it->path; });
    if (f == fds_.end()) {
        const int b = open(it->path.c_str(), O_RDONLY | O_CLOEXEC);
        const int d = b >= 0 ? open(it->path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT) : -1;
        fds_.push_back(Fds{it->path, d, b});
        f = fds_.end() - 1;
    }
    if (f->buffered < 0)
        return {};
    return HostFileSource{f->direct, f->buffered, it->file_off + (u - it->start)};
}

// ---------------------------------------------------------------------------
// HostExpertTier
// ---------------------------------------------------------------------------

namespace {
size_t packed_region(size_t unit_bytes_total, size_t ms_region) { return unit_bytes_total - ms_region; }
size_t ms_region_bytes(size_t ms_bytes) { return round_up(ms_bytes, kAlign) + 2 * kAlign; }
}  // namespace

size_t HostExpertTier::unit_bytes(size_t packed_bytes, size_t ms_bytes) {
    return round_up(packed_bytes, kAlign) + 2 * kAlign + ms_region_bytes(ms_bytes);
}

HostExpertTier::HostExpertTier(char* host, char* dev, size_t bytes, size_t unit, int n_keys, int io_threads,
                               int scratch_units)
    : host_(host),
      dev_(dev),
      unit_(unit),
      n_units_(unit ? static_cast<int>(bytes / unit) : 0),
      n_lru_(std::max(0, n_units_ - std::max(0, scratch_units))),
      src_(n_keys),
      unit_of_(n_keys, -1),
      key_of_(n_units_, -1),
      prev_(n_units_, -1),
      next_(n_units_, -1),
      epoch_(n_units_, 0),
      packed_off_(n_units_, 0),
      ms_off_(n_units_, 0),
      pool_(io_threads) {
    // Every unit starts free, in the LRU list.
    for (int u = 0; u < n_lru_; ++u) {
        prev_[u] = u - 1;
        next_[u] = u + 1 < n_lru_ ? u + 1 : -1;
    }
    head_ = n_lru_ > 0 ? 0 : -1;
    tail_ = n_lru_ - 1;
}

void HostExpertTier::bind_files(const std::string* maps_text) {
    files_.load(maps_text);
    fsrc_packed_.assign(src_.size(), HostFileSource{});
    fsrc_ms_.assign(src_.size(), HostFileSource{});
    for (size_t k = 0; k < src_.size(); ++k) {
        if (src_[k].packed)
            fsrc_packed_[k] = files_.resolve(src_[k].packed);
        if (src_[k].ms)
            fsrc_ms_[k] = files_.resolve(src_[k].ms);
    }
    files_bound_ = true;
}

void HostExpertTier::unlink_(int u) {
    if (prev_[u] >= 0)
        next_[prev_[u]] = next_[u];
    else
        head_ = next_[u];
    if (next_[u] >= 0)
        prev_[next_[u]] = prev_[u];
    else
        tail_ = prev_[u];
    prev_[u] = next_[u] = -1;
}

void HostExpertTier::touch_(int u) {
    if (head_ == u)
        return;
    unlink_(u);
    next_[u] = head_;
    prev_[u] = -1;
    if (head_ >= 0)
        prev_[head_] = u;
    head_ = u;
    if (tail_ < 0)
        tail_ = u;
}

void HostExpertTier::load_(int u, int key) {
    const Src& s = src_[key];
    char* base = host_ + static_cast<size_t>(u) * unit_;
    const size_t ms_reg = ms_region_bytes(s.ms_bytes);
    const size_t pk_reg = packed_region(unit_, ms_reg);
    bool d0 = false, d1 = false;
    long po = read_range(fsrc_packed_[key], fsrc_packed_[key].offset, s.packed_bytes, base, pk_reg, &d0);
    long mo = read_range(fsrc_ms_[key], fsrc_ms_[key].offset, s.ms_bytes, base + pk_reg, ms_reg, &d1);
    const uint64_t buffered = static_cast<uint64_t>(po >= 0 && !d0) + static_cast<uint64_t>(mo >= 0 && !d1);
    uint64_t copies = 0;
    // Not file-backed, or the read failed: copy through the mapping (page faults, still correct).
    if (po < 0) {
        std::memcpy(base, s.packed, s.packed_bytes);
        po = 0;
        ++copies;
    }
    if (mo < 0) {
        std::memcpy(base + pk_reg, s.ms, s.ms_bytes);
        mo = 0;
        ++copies;
    }
    packed_off_[u] = static_cast<uint32_t>(po);
    ms_off_[u] = static_cast<uint32_t>(pk_reg + static_cast<size_t>(mo));
    std::lock_guard<std::mutex> lk(stats_mu_);
    stats_.bytes_read += s.packed_bytes + s.ms_bytes;
    stats_.direct_reads += static_cast<uint64_t>(d0) + static_cast<uint64_t>(d1);
    stats_.buffered_reads += buffered;
    stats_.copies += copies;
}

void HostExpertTier::load_all_(const std::vector<std::pair<int, int>>& loads) {
    if (loads.empty())
        return;
    const auto t0 = std::chrono::steady_clock::now();
    if (loads.size() == 1) {
        load_(loads[0].first, loads[0].second);
    } else {
        for (const auto& [u, k] : loads)
            pool_.submit([this, u = u, k = k] { load_(u, k); });
        pool_.wait();
    }
    stats_.read_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int HostExpertTier::victim_() {
    int u = tail_;
    while (u >= 0 && epoch_[u] == epoch_now_)
        u = prev_[u];
    return u;
}

// The GPU gathered these units into VRAM during the previous step, which ended before this call.
void HostExpertTier::release_leaving_() {
    for (const int u : leaving_) {
        if (key_of_[u] >= 0)
            unit_of_[key_of_[u]] = -1;
        key_of_[u] = -1;
        // Free units go to the cold end: the next victim_() takes them first.
        unlink_(u);
        prev_[u] = tail_;
        next_[u] = -1;
        if (tail_ >= 0)
            next_[tail_] = u;
        tail_ = u;
        if (head_ < 0)
            head_ = u;
    }
    leaving_.clear();
}

bool HostExpertTier::exchange(const int* in_keys, int n, UnitView* src, const int* out_keys, MutView* wb) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!files_bound_)
        bind_files();
    release_leaving_();
    ++epoch_now_;
    int n_victims = 0;
    for (int i = 0; i < n; ++i)
        n_victims += out_keys[i] >= 0 ? 1 : 0;
    if (n > n_units_ - n_lru_ || n + n_victims > n_lru_)
        return false;

    std::vector<int> unit_of_miss(static_cast<size_t>(n));
    std::vector<std::pair<int, int>> loads;
    uint64_t hits = 0;
    int next_scratch = n_lru_;
    for (int i = 0; i < n; ++i) {
        const int k = in_keys[i];
        int u = unit_of_[k];
        if (u >= 0) {
            ++hits;
            epoch_[u] = epoch_now_;  // no victim below may take it
            leaving_.push_back(u);
        } else {
            u = next_scratch++;
            loads.emplace_back(u, k);
        }
        unit_of_miss[static_cast<size_t>(i)] = u;
    }
    load_all_(loads);
    stats_.hits += hits;
    stats_.misses += loads.size();
    for (int i = 0; i < n; ++i) {
        const int u = unit_of_miss[static_cast<size_t>(i)];
        const size_t b = static_cast<size_t>(u) * unit_;
        src[i] = UnitView{dev_ + b + packed_off_[u], dev_ + b + ms_off_[u]};
    }
    for (int i = 0; i < n; ++i) {
        wb[i] = MutView{nullptr, nullptr};
        const int k = out_keys[i];
        if (k < 0)
            continue;
        if (unit_of_[k] >= 0) {  // still here (a prefill scan does not evict): no copy needed
            touch_(unit_of_[k]);
            continue;
        }
        const int u = victim_();
        if (key_of_[u] >= 0)
            unit_of_[key_of_[u]] = -1;
        key_of_[u] = k;
        unit_of_[k] = u;
        epoch_[u] = epoch_now_;
        touch_(u);
        const size_t pk_reg = packed_region(unit_, ms_region_bytes(src_[k].ms_bytes));
        packed_off_[u] = 0;
        ms_off_[u] = static_cast<uint32_t>(pk_reg);
        char* base = dev_ + static_cast<size_t>(u) * unit_;
        wb[i] = MutView{base, base + pk_reg};
        ++stats_.writebacks;
    }
    return true;
}

bool HostExpertTier::acquire(const int* keys, int n, UnitView* out, bool transient) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!files_bound_)
        bind_files();
    release_leaving_();
    ++epoch_now_;
    // Distinct keys must fit: units used by this call are never evicted by it. Transient calls
    // count only the keys that miss, against the scratch units. Passes run in input order.
    std::unordered_map<int, int> slot_of_key;  // key -> unit for this call
    slot_of_key.reserve(static_cast<size_t>(n) * 2);
    std::vector<int> order;  // distinct keys, first occurrence first
    order.reserve(static_cast<size_t>(n));
    int need = 0;
    for (int i = 0; i < n; ++i) {
        if (!slot_of_key.emplace(keys[i], unit_of_[keys[i]]).second)
            continue;
        order.push_back(keys[i]);
        need += (!transient || unit_of_[keys[i]] < 0) ? 1 : 0;
    }
    if (need > (transient ? n_units_ - n_lru_ : n_lru_))
        return false;

    std::vector<std::pair<int, int>> loads;  // (unit, key)
    uint64_t hits = 0;
    int next_scratch = n_lru_;
    // Pass 1: hits claim their units for this call before any miss picks a victim.
    for (const int k : order) {
        const int u = slot_of_key[k];
        if (u < 0)
            continue;
        ++hits;
        if (!transient) {
            epoch_[u] = epoch_now_;
            touch_(u);
        }
    }
    for (const int k : order) {
        int& u = slot_of_key[k];
        if (u >= 0)
            continue;
        if (transient) {
            u = next_scratch++;
        } else {
            u = victim_();  // the capacity check keeps it outside this call
            if (key_of_[u] >= 0)
                unit_of_[key_of_[u]] = -1;
            key_of_[u] = k;
            unit_of_[k] = u;
            epoch_[u] = epoch_now_;
            touch_(u);
        }
        loads.emplace_back(u, k);
    }
    load_all_(loads);
    stats_.hits += hits;
    stats_.misses += loads.size();
    for (int i = 0; i < n; ++i) {
        const int u = slot_of_key[keys[i]];
        const size_t b = static_cast<size_t>(u) * unit_;
        out[i] = UnitView{dev_ + b + packed_off_[u], dev_ + b + ms_off_[u]};
    }
    return true;
}

HostExpertTier::Stats HostExpertTier::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::lock_guard<std::mutex> lk2(stats_mu_);
    return stats_;
}

}  // namespace imp
