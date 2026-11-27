#pragma once
// Shared by tests/test_gemm_f16_narrow_smallm.cu and tests/bench/test_gemm_f16_narrow_smallm_bench.cu.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gemm.h"
#include "compute/gemm_f16_narrow_smallm.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DevBuf() {
        if (p)
            cudaFree(p);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

std::vector<__half> to_half(const std::vector<float>& v) {
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        h[i] = __float2half(v[i]);
    return h;
}

struct Case {
    int M, N0, N1, K;
    std::vector<__half> A, W0, W1;
    DevBuf dA, dW0, dW1, dC0, dC1, ws;
    size_t ws_bytes;

    Case(int m, int n0, int n1, int k, uint32_t seed)
        : M(m),
          N0(n0),
          N1(n1),
          K(k),
          dA(static_cast<size_t>(m) * k * 2),
          dW0(static_cast<size_t>(n0) * k * 2),
          dW1(static_cast<size_t>(n1 > 0 ? n1 : 1) * k * 2),
          dC0(static_cast<size_t>(m) * n0 * 2),
          dC1(static_cast<size_t>(m) * (n1 > 0 ? n1 : 1) * 2),
          ws(gemm_f16_narrow_smallm_workspace_bytes(n0 + n1)),
          ws_bytes(gemm_f16_narrow_smallm_workspace_bytes(n0 + n1)) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> na(0.0f, 1.0f), nw(0.0f, 0.05f);
        std::vector<float> a(static_cast<size_t>(m) * k), w0(static_cast<size_t>(n0) * k),
            w1(static_cast<size_t>(n1) * k);
        for (auto& x : a)
            x = na(rng);
        for (auto& x : w0)
            x = nw(rng);
        for (auto& x : w1)
            x = nw(rng);
        A = to_half(a);
        W0 = to_half(w0);
        W1 = to_half(w1);
        cudaMemcpy(dA.p, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dW0.p, W0.data(), W0.size() * 2, cudaMemcpyHostToDevice);
        if (n1 > 0)
            cudaMemcpy(dW1.p, W1.data(), W1.size() * 2, cudaMemcpyHostToDevice);
        cudaMemset(ws.p, 0, ws_bytes);
    }

    bool run(cudaStream_t s = nullptr) {
        return gemm_f16_narrow_smallm(static_cast<const half*>(dA.p), M, K, static_cast<const half*>(dW0.p),
                                      static_cast<half*>(dC0.p), N0,
                                      N1 > 0 ? static_cast<const half*>(dW1.p) : nullptr,
                                      N1 > 0 ? static_cast<half*>(dC1.p) : nullptr, N1, ws.p, ws_bytes, s);
    }

    std::vector<__half> out(int pair) const {
        const int n = pair == 0 ? N0 : N1;
        std::vector<__half> h(static_cast<size_t>(M) * n);
        cudaMemcpy(h.data(), pair == 0 ? dC0.p : dC1.p, h.size() * 2, cudaMemcpyDeviceToHost);
        return h;
    }
};

}  // namespace
