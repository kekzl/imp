#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Instrumentation-only probe for contextual FFN sparsity (Vector 1 in the
// 2026-05-17 "Break the Memory Wall" research note). Counts, for each of
// the hard-coded thresholds {0.005, 0.01, 0.02, 0.05, 0.1}, the number of
// intermediate-dim rows i with |silu(gate[i]) * up[i]| < t — i.e. how many
// columns of w_down a per-token-aware kernel could legally skip.
//
// Per-layer counters accumulate across every dense-FFN decode step of the
// process. A call to flush_ffn_sparsity_probe_log() drains them to stderr
// (one line per (layer, threshold) pair + one model-aggregate line) and
// resets the counters.
//
// Off unless ffn.sparsity_probe = true in imp.conf (or
// IMP_FFN_SPARSITY_PROBE=1 in the env). When off, the public functions
// short-circuit before any device work.
void probe_ffn_silu_sparsity(int layer, const __half* gate, const __half* up, int K,
                             cudaStream_t stream);

void flush_ffn_sparsity_probe_log();

}  // namespace imp
