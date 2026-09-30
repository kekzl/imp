#pragma once

// Decode graph pool: when the captured forward must be re-derived for the step's context.
// Decode attention bakes its launch topology (split-K num_splits, attention_paged.cu) from
// max_context_len at capture. Default: pow2 high-water mark, growth only (#948).
// deterministic: also on a new request set, so no request replays another's capture (#2182).

#include <cstdint>
#include <span>

namespace imp {

constexpr int decode_graph_bucket_pow2(int x) {
    int b = 1;
    while (b < x)
        b <<= 1;
    return b;
}

// FNV-1a over the request ids of one decode batch, in batch order.
constexpr uint64_t decode_graph_batch_key(std::span<const int> req_ids) {
    uint64_t h = 1469598103934665603ULL;
    for (const int id : req_ids) {
        h ^= static_cast<uint32_t>(id);
        h *= 1099511628211ULL;
    }
    return h;
}

// Key of a batch of request pointers (anything with `->id`).
template <class Reqs>
uint64_t decode_graph_batch_key_of(const Reqs& reqs) {
    uint64_t h = decode_graph_batch_key({});
    for (const auto& r : reqs) {
        const int id = r->id;
        h = (h ^ static_cast<uint32_t>(id)) * 1099511628211ULL;
    }
    return h;
}

// True: re-capture this step. Updates the high-water mark and the batch key on true.
constexpr bool decode_graph_ctx_recapture(int& high_water, uint64_t& last_key, int max_ctx, uint64_t batch_key,
                                          bool deterministic) {
    const int bucket = decode_graph_bucket_pow2(max_ctx);
    const bool new_batch = deterministic && batch_key != last_key;
    if (bucket <= high_water && !new_batch)
        return false;
    high_water = bucket;
    last_key = batch_key;
    return true;
}

}  // namespace imp
