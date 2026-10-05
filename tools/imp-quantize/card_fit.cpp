#include "card_fit.h"

#include "memory/plan.h"

#include <cuda_runtime.h>

#include <cstdio>

namespace imp::quantize {

namespace {

// What is gone from the card before imp allocates a single weight: kMeasuredLibraryReserveBytes
// (src/memory/plan.h, re-measure after a CUDA/driver bump) and the CUDA primary context size
// for this WSL2/WDDM box. Both are measurements, not headroom guesses.
constexpr size_t kContextBytes = 1680ull * 1024 * 1024;

}  // namespace

void report_card_fit(size_t bytes_out, size_t mx_widen) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b == 0)
        return;  // no device visible: the size line above still stands on its own
    const double mib = 1024.0 * 1024.0;
    const size_t overhead = kContextBytes + kMeasuredLibraryReserveBytes;
    if (total_b <= overhead)
        return;
    const double budget = double(total_b - overhead) / mib;
    const double ckpt = double(bytes_out + mx_widen) / mib;
    const char* where = mx_widen > 0 ? "once widened" : "on disk";
    if (mx_widen > 0)
        printf("\nMXFP8 projections are widened to BF16 at load: +%.0f MiB counted below",
               double(mx_widen) / mib);
    printf("\ncard: %.0f MiB total, less %.0f context and %.0f library reserve leaves %.0f MiB",
           double(total_b) / mib, kContextBytes / mib, kMeasuredLibraryReserveBytes / mib, budget);
    if (ckpt >= budget)
        printf(
            "\n      this checkpoint is %.0f MiB %s and does NOT fit: %.0f MiB short before"
            "\n      any KV cache or workspace",
            ckpt, where, ckpt - budget);
    else
        printf(
            "\n      this checkpoint is %.0f MiB %s, leaving %.0f MiB for the KV cache and"
            "\n      workspaces. On-disk size is not the resident figure: a scale-factor cache is"
            "\n      built on top, and a vision tower or MTP sidecar may load separately or not at all",
            ckpt, where, budget - ckpt);
}

}  // namespace imp::quantize
