// Prefill pacing under decode: how many prompt rows one engine step may
// prefill while other sequences decode.
//
// runtime.prefill_chunk_decode_cap bounds ONE chunk forward so a concurrent
// DECODER's inter-token gap stays bounded during another session's ingest
// (#1643). That cap looks only at whether anyone decodes, not at how many
// wait: on a 32-stream burst the second step already has a decoder, and from
// there every step prefills one capped chunk while 30 clients wait for their
// first token; the decoders being protected are the burst's own first
// finishers. With runtime.prefill_cap_fairness the cap scales by
// waiting / decoding, clamped to the full chunk: 31 waiting behind 1 decoder
// run the full chunk, 16 behind 16 keep the cap, and the protection scenario
// (31 decoders, one ingest) is unchanged by construction.
//
// CUDA-free and RuntimeConfig-free so the CPU lane can pin the table.
#pragma once

#include <algorithm>

namespace imp {

// Row cap for this step's prefill, 0 = uncapped. `decode_cap` is the
// configured cap, `full_chunk` the resolved chunk size (executor max,
// block-rounded), `n_waiting` the requests in prefill this step, `n_decoding`
// the requests in the decode batch.
inline int paced_prefill_cap(int decode_cap, int full_chunk, int n_waiting, int n_decoding, bool fair) {
    if (decode_cap <= 0 || n_decoding <= 0)
        return 0;
    if (!fair || n_waiting <= n_decoding)
        return decode_cap;
    const long long scaled = static_cast<long long>(decode_cap) * n_waiting / n_decoding;
    const long long ceiling = std::max(full_chunk, decode_cap);
    return static_cast<int>(std::min(scaled, ceiling));
}

}  // namespace imp
