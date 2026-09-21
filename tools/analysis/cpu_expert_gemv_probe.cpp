// Can the CPU compute a host-resident NVFP4 expert faster than PCIe can fetch it?
//
// Decode on Qwen3.8-Flash-Next leaves ~470 MiB of expert misses per token crossing PCIe at
// ~52 GB/s (tools/analysis/h2d_gather_probe.cu, and the in-situ gather kernel). The same
// bytes already sit in host RAM. This probe does the work the GPU would do, on the CPU,
// over the same access pattern: pick random experts out of a multi-GiB slab, dequant NVFP4
// (FP4 pairs + per-16 FP8 E4M3 micro-scales) and GEMV them against one token's activations.
//
// Reports ms per simulated token and the effective read bandwidth, so the number can be put
// next to the PCIe one. Nothing here is an engine path; it sizes a decision.
//
//   g++ -O3 -march=native -fopenmp -o /tmp/probe tools/analysis/cpu_expert_gemv_probe.cpp
//   /tmp/probe [misses_per_token] [tokens] [threads]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

namespace {

// Qwen3.8-Flash-Next expert shapes: gate/up [640, 2560], down [2560, 640]. Either way one
// expert-projection is 640 * 2560 weights: 819200 packed bytes + 102400 micro-scale bytes.
constexpr int kN = 640;
constexpr int kK = 2560;
constexpr size_t kPacked = size_t(kN) * kK / 2;
constexpr size_t kScales = size_t(kN) * kK / 16;
constexpr size_t kExpertBytes = kPacked + kScales;

// FP4 E2M1 code -> value, the same 16 values the kernels use.
constexpr float kFp4[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

float fp8_e4m3_slow(uint8_t b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    float v;
    if (e == 0)
        v = float(m) / 8.0f * 0.015625f;  // 2^-6 subnormal
    else
        v = (1.0f + float(m) / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}

// 256-entry table: the per-micro-block scale decode must not be a call in the inner loop.
struct Fp8Table {
    float v[256];
    Fp8Table() {
        for (int i = 0; i < 256; ++i)
            v[i] = fp8_e4m3_slow(uint8_t(i));
    }
};
const Fp8Table kFp8;

// Memory-bound ceiling for the same access pattern: read the block, nothing else.
uint64_t read_only(const uint8_t* p, size_t n) {
    uint64_t a = 0, b = 0, c = 0, d = 0;
    const uint64_t* q = reinterpret_cast<const uint64_t*>(p);
    for (size_t i = 0; i + 4 <= n / 8; i += 4) {
        a += q[i]; b += q[i + 1]; c += q[i + 2]; d += q[i + 3];
    }
    return a + b + c + d;
}

// One expert-projection GEMV: y[n] = sum_k dequant(W[n][k]) * x[k], rows split over threads.
void gemv_expert(const uint8_t* packed, const uint8_t* scales, const float* x, float* y,
                 int row_begin, int row_end) {
    for (int n = row_begin; n < row_end; ++n) {
        const uint8_t* w = packed + size_t(n) * (kK / 2);
        const uint8_t* s = scales + size_t(n) * (kK / 16);
        float acc = 0.0f;
        for (int kb = 0; kb < kK / 16; ++kb) {
            const float sc = kFp8.v[s[kb]];
            float blk = 0.0f;
            const uint8_t* wb = w + kb * 8;
            for (int j = 0; j < 8; ++j) {
                const uint8_t byte = wb[j];
                blk += kFp4[byte & 0xF] * x[kb * 16 + j * 2];
                blk += kFp4[byte >> 4] * x[kb * 16 + j * 2 + 1];
            }
            acc += blk * sc;
        }
        y[n] = acc;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int misses = argc > 1 ? std::atoi(argv[1]) : 537;   // measured: 62.7 % hits of 1440
    const int tokens = argc > 2 ? std::atoi(argv[2]) : 3;
    const int threads = argc > 3 ? std::atoi(argv[3]) : int(std::thread::hardware_concurrency());

    // A slab big enough that nothing stays in cache: 4 GiB of "experts".
    const size_t slab_experts = (4ull << 30) / kExpertBytes;
    printf("slab %zu experts (%.2f GiB), expert %.2f MiB, misses/token %d, threads %d\n",
           slab_experts, double(slab_experts * kExpertBytes) / (1 << 30),
           double(kExpertBytes) / (1 << 20), misses, threads);

    std::vector<uint8_t> slab(slab_experts * kExpertBytes);
    std::mt19937 rng(7);
    for (size_t i = 0; i < slab.size(); i += 4096)
        slab[i] = uint8_t(rng());  // fault every page in
    std::memset(slab.data() + 1, 0x42, 4096);

    std::vector<float> x(kK, 0.01f);
    std::vector<float> y(size_t(kN) * threads, 0.0f);
    std::uniform_int_distribution<size_t> pick(0, slab_experts - 1);

    double best_ms = 1e18;
    for (int t = 0; t < tokens; ++t) {
        std::vector<size_t> idx(misses);
        for (int i = 0; i < misses; ++i)
            idx[i] = pick(rng);

        auto run = [&](bool compute) {
            std::vector<std::thread> pool;
            for (int th = 0; th < threads; ++th) {
                pool.emplace_back([&, th] {
                    uint64_t sink = 0;
                    for (int i = th; i < misses; i += threads) {
                        const uint8_t* base = slab.data() + idx[i] * kExpertBytes;
                        if (compute)
                            gemv_expert(base, base + kPacked, x.data(),
                                        y.data() + size_t(kN) * th, 0, kN);
                        else
                            sink += read_only(base, kExpertBytes);
                    }
                    if (!compute)
                        y[size_t(kN) * th] += float(sink & 1);
                });
            }
            for (auto& p : pool)
                p.join();
        };

        auto r0 = std::chrono::steady_clock::now();
        run(false);
        auto r1 = std::chrono::steady_clock::now();
        const double rms = std::chrono::duration<double, std::milli>(r1 - r0).count();
        printf("  token %d: read-only %6.2f ms  %6.1f GB/s |", t, rms,
               double(misses) * kExpertBytes / 1e9 / (rms / 1e3));

        auto t0 = std::chrono::steady_clock::now();
        run(true);
        auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double gb = double(misses) * kExpertBytes / 1e9;
        printf(" dequant+gemv %7.2f ms  %6.1f GB/s\n", ms, gb / (ms / 1e3));
        if (ms < best_ms)
            best_ms = ms;
    }

    const double gb = double(misses) * kExpertBytes / 1e9;
    printf("best %.2f ms/token, %.1f GB/s. PCIe for the same bytes at 52 GB/s: %.2f ms\n",
           best_ms, gb / (best_ms / 1e3), gb / 52.0 * 1e3);
    printf("checksum %.3f\n", double(y[0] + y[kN / 2]));
    return 0;
}
