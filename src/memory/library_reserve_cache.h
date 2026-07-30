#pragma once

// Persisted library-reserve measurements (docs/MEMORY_ARCHITECTURE.md A1.5,
// AUDIT B41/B42/B49).
//
// The planner's largest fixed charge is what cuBLAS/CUTLASS claim on the FIRST
// forward pass. `kMeasuredLibraryReserveBytes` was a single constant, and it is
// wrong in both directions: measured 0 MiB on Qwen3-4B-IQ4_NL, 4182 on
// Qwen3-4B-Q8_0 and 7460 on Qwen3-8B-Q8_0. Charging 3900 for all of them sets
// aside 3.9 GiB of KV pool for nothing on the first, and under-reserves the
// third by 3.5 GiB — which is most of the residual acceptance criterion 5 was
// missing, and the difference between 82.5 % and 98.3 % accounted for 6.
//
// The plan needs the figure BEFORE the forward that produces it, so a single run
// cannot both measure and use it. It can, however, remember: the value is
// stable per (model, quant path, library stack) and invariant to batch and
// context (A1.5 M5). So the first start on a given model charges the constant
// and records what actually happened; every start after that charges the
// measured value.
//
// That is what "capacity planned, not discovered" (I4) looks like for a quantity
// only the device can tell you — planned from a recorded measurement rather than
// re-derived from a live query on every boot.
//
// Format is one `key<TAB>bytes` line per entry, rewritten whole. It is a cache:
// a missing, unreadable or corrupt file is not an error, it just means "charge
// the constant and measure again".

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

// Recorded bytes for `key`. `found` distinguishes a recorded ZERO from no entry
// at all — and that distinction is the whole point of the out-param: models whose
// first forward claims nothing (measured: Qwen3-4B-IQ4_NL, Qwen3.6-35B-A3B-NVFP4)
// record 0, and a `> 0` test on the return value silently threw their
// measurement away and charged the 3900 MiB constant instead. B43 found exactly
// this shape in the reporter and fixed it there; the loader kept it (AUDIT B70).
// Never throws; an absent or malformed file reads as "no entry".
size_t library_reserve_cache_load(const std::string& path, const LibraryReserveKey& key,
                                  bool* found = nullptr);

// Record `bytes` for `key`, replacing any previous entry. Returns false when the
// file could not be written — the caller should warn once and carry on, because
// failing a model load over a cache write would be absurd.
bool library_reserve_cache_store(const std::string& path, const LibraryReserveKey& key,
                                 size_t bytes);

}  // namespace imp
