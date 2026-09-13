#pragma once

// Persisted library-reserve measurements (MEMORY.md A1.5, AUDIT B41/B42/B49).
// kMeasuredLibraryReserveBytes (what cuBLAS/CUTLASS claim on the first forward) was a
// single constant, wrong in both directions across models: it either wastes GiB of KV
// pool or under-reserves and spills the card.
// The plan needs the figure BEFORE the forward that produces it, so a single run cannot
// both measure and use it; the value IS stable per (model, quant path, library stack)
// and invariant to batch/context (A1.5 M5), so the first start charges the constant and
// records what happened, and every start after charges the measured value. That is
// "capacity planned, not discovered" (I4) for a quantity only the device can tell you.
// Format: one `key<TAB>bytes` line per entry, rewritten whole. A cache: a missing,
// unreadable or corrupt file just means "charge the constant and measure again".

#include <cstddef>
#include <cstdint>
#include <string>

namespace imp {

// Identity of a measurement. Everything the charge was observed to vary with;
// deliberately NOT batch or context, which it does not vary with.
struct LibraryReserveKey {
    uint64_t model_fingerprint = 0;
    int nvfp4_decode_mode = 0;  // use_nvfp4_decode: selects the execution path
    bool fp8_prefill = false;
    int cuda_runtime_version = 0;  // the libraries are the thing being measured

    std::string str() const;
};

// Default cache location. `$XDG_CACHE_HOME/imp/library_reserve` when set,
// otherwise `$HOME/.cache/imp/library_reserve`. Empty when neither is set —
// callers then skip the cache rather than guessing a path.
std::string library_reserve_cache_default_path();

// Recorded bytes for `key`. `found` distinguishes a recorded ZERO from no entry at all:
// models whose first forward claims nothing record 0, and a `> 0` test on the return
// value silently threw that measurement away and charged the 3900 MiB constant instead
// (B43 fixed this shape in the reporter; the loader kept the bug, AUDIT B70). Never
// throws; an absent or malformed file reads as "no entry".
size_t library_reserve_cache_load(const std::string& path, const LibraryReserveKey& key,
                                  bool* found = nullptr);

// Record `bytes` for `key`, replacing any previous entry. Returns false when the
// file could not be written — the caller should warn once and carry on, because
// failing a model load over a cache write would be absurd.
bool library_reserve_cache_store(const std::string& path, const LibraryReserveKey& key,
                                 size_t bytes);

}  // namespace imp
