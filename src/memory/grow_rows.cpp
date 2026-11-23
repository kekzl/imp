#include "memory/grow_rows.h"

#include "core/logging.h"
#include "memory/mem_account.h"
#include "memory/vram_allocator.h"
#include "memory/vram_query.h"

#include <algorithm>

namespace imp {

GrowRows::~GrowRows() {
    for (void* p : fixed_)
        va_->free(p);
    for (Region& r : regions_) {
        MemAccount::instance().note(pool_, -static_cast<std::ptrdiff_t>(r.committed()));
        vram_reserved_uncommitted_add(-static_cast<std::ptrdiff_t>(r.reserved() - r.committed()));
    }
}

void GrowRows::init(VRAMAllocator& va, Backend* growable, const char* pool, size_t row_bytes, size_t max_rows,
                    size_t step_rows) {
    va_ = &va, backend_ = growable, pool_ = pool;
    row_bytes_ = row_bytes, max_rows_ = max_rows, step_rows_ = std::max<size_t>(step_rows, 1);
    rows_ = growable ? std::min(max_rows, step_rows_) : max_rows;
}

void* GrowRows::add() {
    const size_t full = max_rows_ * row_bytes_;
    if (backend_ && va_->can_allocate(full)) {
        AcquireResult r = backend_->acquire_growable(full, rows_ * row_bytes_, 256, RegionTag::SsmState);
        if (r) {
            void* base = r.region.base();
            MemAccount::instance().note(pool_, static_cast<std::ptrdiff_t>(r.region.committed()));
            vram_reserved_uncommitted_add(
                static_cast<std::ptrdiff_t>(r.region.reserved() - r.region.committed()));
            regions_.push_back(std::move(r.region));
            return base;
        }
        IMP_LOG_WARN("%s: growable acquisition of %.1f MiB failed (%s)", pool_,
                     static_cast<double>(full) / 1048576.0, mem_error_name(r.error));
        return nullptr;
    }
    if (backend_)
        return nullptr;
    void* p = va_->allocate(full, pool_);
    if (p)
        fixed_.push_back(p);
    return p;
}

bool GrowRows::ensure(size_t rows) {
    if (rows <= rows_)
        return true;
    if (!backend_ || rows > max_rows_)
        return false;
    const size_t want = std::min(max_rows_, (rows + step_rows_ - 1) / step_rows_ * step_rows_);
    for (Region& r : regions_) {
        const size_t before = r.committed();
        const MemError e = backend_->commit(r, want * row_bytes_);
        const std::ptrdiff_t added = static_cast<std::ptrdiff_t>(r.committed()) -
                                     static_cast<std::ptrdiff_t>(before);
        MemAccount::instance().note(pool_, added);
        vram_reserved_uncommitted_add(-added);
        if (e != MemError::Ok) {
            IMP_LOG_ERROR("%s: commit to %zu rows failed (%s)", pool_, want, mem_error_name(e));
            return false;
        }
    }
    rows_ = want;
    return true;
}

}  // namespace imp
