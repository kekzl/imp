// Prefill pacing under decode: how many prompt rows one engine step may
// prefill while other sequences decode.
//
// runtime.prefill_chunk_decode_cap bounds one chunk so a decoder's
// inter-token gap stays bounded during concurrent ingest (#1643). It counts
// only whether anyone decodes, not how many wait, so a waiting burst can
// starve behind its own first finishers.
// runtime.prefill_cap_fairness = W scales the cap by W x waiting / decoding
// once that ratio exceeds 1, clamped to the full chunk. W = 1 is symmetric,
// W = 4 (default) relaxes once under a quarter of the wave still waits,
// W = 0 disables scaling.
//
// CUDA-free and RuntimeConfig-free so the CPU lane can pin the table.
#pragma once

#include <algorithm>

namespace imp {

// Row cap for this step's prefill, 0 = uncapped. `decode_cap` is the
// configured cap, `full_chunk` the resolved chunk size (executor max,
// block-rounded), `n_waiting` the requests in prefill this step, `n_decoding`
// the requests in the decode batch, `weight` runtime.prefill_cap_fairness.
inline int paced_prefill_cap(int decode_cap, int full_chunk, int n_waiting, int n_decoding, int weight) {
    if (decode_cap <= 0 || n_decoding <= 0)
        return 0;
    const long long weighted = static_cast<long long>(std::max(weight, 0)) * n_waiting;
    if (weighted <= n_decoding)
        return decode_cap;
    const long long scaled = static_cast<long long>(decode_cap) * weighted / n_decoding;
    const long long ceiling = std::max(full_chunk, decode_cap);
    return static_cast<int>(std::min(scaled, ceiling));
}

}  // namespace imp
