#pragma once

// kv_cache.fp8_scale wiring outside Engine (engine_kv_fp8_scale.cpp). Selection math: core/kv_fp8_scale.h.

#include "core/kv_fp8_scale.h"
#include "core/qtype.h"

#include <cstdint>
#include <vector>

namespace imp {

class Engine;
class GraphExecutor;
class Model;
struct RuntimeConfig;

// kv_cache.fp8_scale parsed; throws std::runtime_error on an unknown value (any KV dtype).
[[nodiscard]] Fp8KvScaleMode resolve_fp8_kv_scale_mode(const RuntimeConfig& rc);

// checkpoint mode on FP8 KV: static per-layer scales from the model's k_scale / v_scale (1.0 where
// absent). No-op otherwise. Call after GraphExecutor::set_kv_layer_map.
void apply_checkpoint_fp8_kv_scales(GraphExecutor& ex, const Model& model,
                                    const std::vector<int>& kv_layer_map, QType kv_dtype,
                                    const RuntimeConfig& rc);

// Persisted prefix-cache key: a static mode is folded in so blocks never restore under another
// scale; legacy keeps the model fingerprint (old cache files stay valid).
[[nodiscard]] uint64_t prefix_cache_fingerprint(uint64_t model_fingerprint, QType kv_dtype,
                                                const RuntimeConfig& rc);

// calibrate mode knows its scale only after warmup: no prefix-cache restore at init.
[[nodiscard]] bool prefix_cache_restore_allowed(QType kv_dtype, const RuntimeConfig& rc);

// calibrate mode on FP8 KV: one fixed-text prefill through the public request path, then static
// per-layer scales. Call before any decode graph capture. run_forward false only warns when no
// calibration ran (the MXFP4 warmup skips its forward).
void calibrate_fp8_kv_scales(Engine& engine, QType kv_dtype, int prefill_chunk_size, bool run_forward);

}  // namespace imp
