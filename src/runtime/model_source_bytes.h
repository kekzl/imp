#pragma once

// Bytes of a model on disk and what the auto resolvers reserve for NVFP4 weights from them.

#include <cstddef>
#include <string>

namespace imp {

// A checkpoint directory sums its weight shards (config / tokenizer JSON is noise); 0 = could not tell.
size_t model_source_bytes(const std::string& path);

// Weight bytes the max_seq_len auto resolver holds back on a native-NVFP4 checkpoint: the shards on disk when
// every expert stays on the card and they are smaller than the shape estimate (Flint: 23.3 GB on disk, 26.8
// GB estimated, which left 1.0 GB and a 26690-token context on a 32 GB card).
[[nodiscard]] constexpr size_t nvfp4_weight_reserve_bytes(size_t shape_estimate, int host_expert_layers,
                                                          size_t on_disk) {
    return host_expert_layers == 0 && on_disk > 0 && on_disk < shape_estimate ? on_disk : shape_estimate;
}

}  // namespace imp
