#pragma once
// One source of a gathered recurrent snapshot (RecurrentSnapshotStore::save_gather): bytes from src, device
// or pinned host memory, land at byte off of the entry.

#include <cstddef>

namespace imp {

struct SnapshotPiece {
    const void* src = nullptr;
    size_t off = 0, bytes = 0;
};

}  // namespace imp
