#pragma once

// On-disk warm weight cache — cold-boot companion to the suspend-to-RAM
// snapshot (weight_snapshot.h).
//
// At the end of a fully-cold weight upload, the TRANSFORMED upload records
// (host BF16→FP16 conversions, GPU dequants, split layouts — everything whose
// device bytes differ from the model file) are copied D2H once and persisted
// next to the model. Raw-from-source records are deliberately NOT stored: a
// cold re-upload from the model file mmap is byte-equivalent and costs the
// same, so the cache stays small (near-zero for raw-served GGUF quants and
// NVFP4-prequant SafeTensors; ~model-size only for BF16-dense checkpoints,
// which is exactly where it saves the most startup time).
//
// At the next boot the cache is loaded into a WeightSnapshot and armed for
// the upload pass — the same per-key restore machinery as suspend/resume,
// with the same safety property: any miss or mismatch falls back to the
// normal cold path per tensor.
//
// Staleness guards: format version, model arch + layer count, and a content
// fingerprint of the model path (total regular-file bytes + newest mtime).
// The file is written atomically (tmp + rename) and is best-effort in both
// directions — any failure just means a normal cold load.

#include <memory>
#include <string>

namespace imp {

class Model;
class WeightSnapshot;

// "<file>.impwcache" for a regular file, "<dir>/.imp_warm_cache" for a
// SafeTensors directory.
std::string weight_cache_path_for(const std::string& model_path);

struct WeightCacheFingerprint {
    uint64_t total_bytes = 0;
    int64_t mtime_max = 0;  // seconds since epoch of the newest regular file
    bool operator==(const WeightCacheFingerprint&) const = default;
};

// Content fingerprint of a model file or SafeTensors directory (recursive,
// regular files only; the cache file itself is excluded). Zero-initialized
// result if the path does not exist.
WeightCacheFingerprint weight_cache_fingerprint(const std::string& model_path);

// Persist the model's transformed live upload records (device bytes D2H'd
// here). Returns false and logs on any failure; never throws. Writes nothing
// when there are no transformed records to store.
bool weight_cache_write(const Model& model, const std::string& cache_path);

// Load + validate a cache file against the model's identity (arch, n_layers)
// and the current fingerprint of its source path. Returns nullptr on any
// mismatch or parse problem (logged at INFO — stale caches are expected
// after a model re-download).
std::unique_ptr<WeightSnapshot> weight_cache_load(const std::string& cache_path, const Model& model);

}  // namespace imp
