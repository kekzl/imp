#pragma once
// Row arrays that commit their memory as the row count grows (a sequence's context K/V): one growable
// Region per array, all grown together in steps. Without a growable backend the arrays are allocated
// in full. What is reserved but not committed is booked with VRAMAllocator::plan().

#include "memory/backend.h"

#include <cstddef>
#include <vector>

namespace imp {

class VRAMAllocator;

class GrowRows {
public:
    GrowRows() = default;
    GrowRows(const GrowRows&) = delete;
    GrowRows& operator=(const GrowRows&) = delete;
    ~GrowRows();

    // Arrays of max_rows rows of row_bytes; growable = null: fixed. pool: the VRAM audit name.
    void init(VRAMAllocator& va, Backend* growable, const char* pool, size_t row_bytes, size_t max_rows,
              size_t step_rows);
    // One array, its first step committed. Null: no room (the full array must fit the plan).
    [[nodiscard]] void* add();
    // Every array holds at least rows rows (capped at max_rows). False: the device refused the commit.
    [[nodiscard]] bool ensure(size_t rows);
    [[nodiscard]] size_t rows() const { return rows_; }
    [[nodiscard]] bool growable() const { return backend_ != nullptr; }

private:
    VRAMAllocator* va_ = nullptr;
    Backend* backend_ = nullptr;
    const char* pool_ = "grow_rows";
    size_t row_bytes_ = 0, max_rows_ = 0, step_rows_ = 0, rows_ = 0;
    std::vector<Region> regions_;
    std::vector<void*> fixed_;
};

}  // namespace imp
