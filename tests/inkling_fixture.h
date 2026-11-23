#pragma once
// Shared by the Inkling GPU tests of test-compute: one weight load per process (the checkpoint takes
// 16 GiB), the id reader and the two ways to drive InklingForward.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "compute/inkling_ops.h"
#include "exec/inkling_forward.h"
#include "memory/vram_allocator.h"
#include "model/inkling_weights.h"

namespace imp::inkling_test {
inline std::vector<int32_t> read_ids(const std::string& path) {
    std::ifstream f(path);
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int32_t> ids;
    const size_t a = s.find('[', s.find("\"ids\"")), b = s.find(']', a);
    if (a == std::string::npos || b == std::string::npos)
        return ids;
    std::stringstream ss(s.substr(a + 1, b - a - 1));
    std::string tok;
    while (std::getline(ss, tok, ','))
        ids.push_back(std::stoi(tok));
    return ids;
}

struct Fixture {
    VRAMAllocator va;  // declared first: the weights free into it
    InklingWeights w;
    bool loaded = false;
    bool load(const char* dir) { return loaded = va.init() && load_inkling_weights(dir, w, va); }
};
inline Fixture& fixture() {
    static Fixture f;
    return f;
}

// Final state of every row after run() over ids[from..) at position from.
inline std::vector<float> run_rows(InklingForward& fwd, const InklingWeights& w,
                                   const std::vector<int32_t>& ids, int from, const std::vector<int>& cuts,
                                   cudaStream_t rs = nullptr) {
    const int T = static_cast<int>(ids.size()) - from, H = w.cfg.hidden;
    const size_t n = static_cast<size_t>(T) * H;
    int32_t* d_ids = nullptr;
    float* d_f = nullptr;
    half* d_h = nullptr;
    std::vector<float> last(n);
    if (cudaMalloc(&d_ids, static_cast<size_t>(T) * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_f, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_h, n * sizeof(half)) != cudaSuccess)
        return {};
    cudaMemcpy(d_ids, ids.data() + from, static_cast<size_t>(T) * sizeof(int32_t), cudaMemcpyHostToDevice);
    bool ok = true;
    // One call per stretch between the cuts (absolute positions).
    for (int at = 0, m = 0; at < T && ok; at += m) {
        m = T - at;
        for (int c : cuts)
            if (c > from + at && c - from - at < m)
                m = c - from - at;
        const size_t o = static_cast<size_t>(at) * H, mn = static_cast<size_t>(m) * H;
        inkling::embed_rows(w.embed, d_ids + at, d_f + o, m, H, nullptr);
        inkling::f32_to_f16(d_f + o, d_h + o, mn, nullptr);
        ok = fwd.run(d_h + o, m, from + at, rs, ids.data() + from + at);
        if (rs)
            cudaStreamSynchronize(rs);
    }
    inkling::f16_to_f32(d_h, d_f, n, nullptr);
    cudaMemcpy(last.data(), d_f, last.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ids);
    cudaFree(d_f);
    cudaFree(d_h);
    return ok ? last : std::vector<float>();
}

// One decode step through run(): the token's embedding in, the final state of the row out.
struct Stepper {
    const InklingWeights& w;
    cudaStream_t s = nullptr;
    int32_t* d_id = nullptr;
    float* d_f = nullptr;
    half* d_h = nullptr;
    explicit Stepper(const InklingWeights& w_) : w(w_) {
        const size_t H = static_cast<size_t>(w.cfg.hidden);
        cudaStreamCreate(&s);
        cudaMalloc(&d_id, sizeof(int32_t));
        cudaMalloc(&d_f, H * sizeof(float));
        cudaMalloc(&d_h, H * sizeof(half));
    }
    ~Stepper() {
        cudaFree(d_id);
        cudaFree(d_f);
        cudaFree(d_h);
        cudaStreamDestroy(s);
    }
    std::vector<float> step(InklingForward& f, int32_t tok, int pos) {
        const int H = w.cfg.hidden;
        std::vector<float> out(static_cast<size_t>(H));
        cudaMemcpyAsync(d_id, &tok, sizeof tok, cudaMemcpyHostToDevice, s);
        inkling::embed_rows(w.embed, d_id, d_f, 1, H, s);
        inkling::f32_to_f16(d_f, d_h, out.size(), s);
        if (!f.run(d_h, 1, pos, s, &tok))
            return {};
        inkling::f16_to_f32(d_h, d_f, out.size(), s);
        cudaMemcpyAsync(out.data(), d_f, out.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        return out;
    }
};

}  // namespace imp::inkling_test
