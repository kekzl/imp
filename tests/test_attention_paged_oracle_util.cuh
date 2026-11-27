#pragma once
// Shared by tests/test_attention_paged_oracle.cu and tests/bench/test_attention_paged_oracle_bench.cu.

#include "compute/attention_paged.h"
#include "core/tensor.h"
#include "core/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace imp {
namespace {

static constexpr int BLOCK_SIZE = 16;  // kKVBlockSize

// LCG fill bit-exact mirror of tests/refs/gen_attention_crosspath_golden.py::lcg_fill (f32
// multiply-only, no libm): heavy-tailed cubed-uniform, amp scales envelope, 1/256 outliers at 2x.
void lcg_fill(std::vector<half>& out, uint32_t seed, float amp) {
    uint32_t x = seed;
    for (auto& h : out) {
        x = x * 1664525u + 1013904223u;
        int32_t v = (int32_t)((x >> 8) & 0x3FFFu) - 8192;
        float f = (float)v * (1.0f / 8192.0f);
        float val = f * f * f * amp;
        if ((x & 0xFFu) == 0)
            val *= 2.0f;
        h = __float2half(val);
    }
}

// ---------------------------------------------------------------------------
// Device tensor helpers
// ---------------------------------------------------------------------------
void* up(const void* host, size_t bytes) {
    void* d = nullptr;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice);
    return d;
}

Tensor f16_tensor(void* d, std::initializer_list<int64_t> shape) {
    Tensor t;
    t.qtype = QType::F16;
    t.ndim = (int)shape.size();
    int i = 0;
    for (auto s : shape)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    t.data = d;
    return t;
}

Tensor raw_tensor(void* d, QType qt, std::initializer_list<int64_t> shape) {
    Tensor t;
    t.qtype = qt;
    t.ndim = (int)shape.size();
    int i = 0;
    for (auto s : shape)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    t.data = d;
    return t;
}

std::vector<float> read_o(void* d_o, size_t elems) {
    std::vector<half> h(elems);
    cudaMemcpy(h.data(), d_o, elems * sizeof(half), cudaMemcpyDeviceToHost);
    std::vector<float> o(elems);
    for (size_t i = 0; i < elems; i++)
        o[i] = __half2float(h[i]);
    return o;
}

}  // namespace
}  // namespace imp
