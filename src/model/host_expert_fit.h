#pragma once
// Host-resident expert bytes and whether host RAM can pin them: weight_upload.cpp decides
// placement with these, the auto max_batch_size resolver asks the same before the upload.
#include <cstddef>

namespace imp {

class Model;
struct TransformerLayer;

// Expert tensor bytes of one layer still on the host: packed 3-D (GPU-dequantizable qtypes)
// plus per-expert 2-D tensors (NVFP4 llm-compressor). 0 for a dense layer.
size_t layer_host_expert_bytes(const TransformerLayer& L);
// SfAtom scale bytes phase 3 builds on the device for this layer's NVFP4-prequant experts
// (cutlass_nvfp4_sf_size per expert, ~1/8 of the packed bytes; Kimi-Linear: 2.88 GB).
size_t layer_nvfp4_expert_sf_bytes(const TransformerLayer& L);
// Sum over the model's layers.
size_t model_host_expert_bytes(const Model& model);
// Host RAM kept free when the experts are pinned: 6 GiB flat. 10 % of RAM (7.85 GiB on 78 GiB) left
// Flash-Next 0.4-3.1 GiB short and unpinned at 63.6 tok/s instead of 103 (PERF_LOG 2026-11-18).
inline constexpr size_t kHostExpertPinHeadroomBytes = 6ull << 30;
// Host RAM holds `bytes` pinned with that headroom kept (true when the host figure is unknown).
[[nodiscard]] bool host_ram_holds(size_t bytes);
// moe.host_expert_pool_mib caps the host expert tier below `bytes`.
[[nodiscard]] bool host_expert_pool_caps(size_t bytes, int pool_mib);

}  // namespace imp
