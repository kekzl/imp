// Inkling2-Loop conv state of a row inside a logged call (inkling_ops.h conv_state_at).

#include "compute/inkling_ops.h"
#include "core/cuda_raii.h"

namespace imp::inkling {

namespace {

constexpr int kThreads = 256;

// dst = the last three of [pre state (3 rows), logged inputs 0..j] for every conv site.
// Per application: k [., KD], v [., KD], attn [., H], mlp [., H] with 3 rows (state) or rows rows (log).
__global__ void conv_state_at_k(const float* pre, const float* log, float* dst, int j, size_t n, int KD,
                                int H, int rows) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const size_t kd = KD, h = H, R = rows;
    const size_t S = 3 * (2 * kd + 2 * h), L = R * (2 * kd + 2 * h);
    const size_t app = i / S, o = i % S;
    // Site: 0 k, 1 v, 2 attn, 3 mlp; its channel count, state offset and log offset.
    const int site = o < 3 * kd ? 0 : (o < 6 * kd ? 1 : (o < 6 * kd + 3 * h ? 2 : 3));
    const size_t C = site < 2 ? kd : h;
    const size_t base = site == 0 ? 0 : (site == 1 ? 3 * kd : (site == 2 ? 6 * kd : 6 * kd + 3 * h));
    const size_t lbase = site == 0 ? 0 : (site == 1 ? R * kd : (site == 2 ? 2 * R * kd : 2 * R * kd + R * h));
    const size_t r = (o - base) / C, c = (o - base) % C, idx = (size_t)j + 1 + r;
    dst[i] = idx < 3 ? pre[app * S + base + idx * C + c] : log[app * L + lbase + (idx - 3) * C + c];
}

}  // namespace

void conv_state_at(const float* pre, const float* log, float* dst, int j, int n_apps, int KD, int H, int rows,
                   cudaStream_t s) {
    const size_t n = static_cast<size_t>(n_apps) * 3 *
                     (2 * static_cast<size_t>(KD) + 2 * static_cast<size_t>(H));
    conv_state_at_k<<<static_cast<unsigned>((n + kThreads - 1) / kThreads), kThreads, 0, s>>>(pre, log, dst,
                                                                                              j, n, KD, H,
                                                                                              rows);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp::inkling
