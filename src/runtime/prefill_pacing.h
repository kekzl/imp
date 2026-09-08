// Prefill pacing under decode: how many prompt rows one engine step may
// prefill while other sequences decode.
//
// runtime.prefill_chunk_decode_cap bounds ONE chunk forward so a concurrent
// DECODER's inter-token gap stays bounded during another session's ingest
// (#1643). That cap looks only at whether anyone decodes, not at how many
// wait: on a 32-stream burst the second step already has a decoder, and from
// there every step prefills one capped chunk while 30 clients wait for their
// first token; the decoders being protected are the burst's own first
// finishers. With runtime.prefill_cap_fairness = W the cap scales by
// W x waiting / decoding once W x waiting exceeds decoding, clamped to the
// full chunk: W is how many waiters one decoder's smoothness is worth. W = 1
// is the symmetric rule (31 waiting behind 1 decoder run the full chunk, 16
// behind 16 keep the cap); W = 4 (the default) keeps a burst at the full
// chunk until fewer than a quarter of the wave still waits. The protection
// scenario (31 decoders, one ingest) is unchanged for every W below 31, and
// W = 0 turns the scaling off.
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
