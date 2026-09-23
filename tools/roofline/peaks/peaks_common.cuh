// Shared helpers for the sm_120a peak microbenchmarks (tools/roofline/peaks/).
// Every result carries the SM/mem clock and power sampled by NVML right after its timed window.
#pragma once

#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define PK_CHECK(x)                                                                                  \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, \
                         cudaGetErrorString(e_));                                                    \
            std::exit(2);                                                                            \
        }                                                                                            \
    } while (0)

namespace peaks {

struct Clocks {
    unsigned sm_mhz = 0, mem_mhz = 0, power_mw = 0;
};

inline nvmlDevice_t nvml_dev() {
    static nvmlDevice_t dev = [] {
        nvmlDevice_t d{};
        if (nvmlInit_v2() != NVML_SUCCESS || nvmlDeviceGetHandleByIndex_v2(0, &d) != NVML_SUCCESS)
            std::fprintf(stderr, "NVML unavailable: clocks will read 0\n");
        return d;
    }();
    return dev;
}

inline Clocks sample_clocks() {
    Clocks c;
    nvmlDevice_t d = nvml_dev();
    nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c.sm_mhz);
    nvmlDeviceGetClockInfo(d, NVML_CLOCK_MEM, &c.mem_mhz);
    nvmlDeviceGetPowerUsage(d, &c.power_mw);
    return c;
}

struct Result {
    std::string group, name, unit;
    double value = 0.0;   // median over reps
    double spread = 0.0;  // (max - min) / median
    Clocks clk;
    std::string note;
};

inline std::vector<Result>& results() {
    static std::vector<Result> r;
    return r;
}

// Runs `launch` until >= warm_s of busy time has passed (idle downclock ramps ~1 s), then
// times `reps` windows of `inner` launches each. Returns per-launch milliseconds per window.
template <class F>
std::vector<double> time_launches(F&& launch, int inner, int reps = 7, double warm_s = 1.5) {
    cudaEvent_t s, e;
    PK_CHECK(cudaEventCreate(&s));
    PK_CHECK(cudaEventCreate(&e));
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
        for (int i = 0; i < inner; ++i)
            launch();
        PK_CHECK(cudaDeviceSynchronize());
    }
    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        PK_CHECK(cudaEventRecord(s));
        for (int i = 0; i < inner; ++i)
            launch();
        PK_CHECK(cudaEventRecord(e));
        PK_CHECK(cudaEventSynchronize(e));
        float t = 0.0f;
        PK_CHECK(cudaEventElapsedTime(&t, s, e));
        ms.push_back(static_cast<double>(t) / inner);
    }
    PK_CHECK(cudaGetLastError());
    PK_CHECK(cudaEventDestroy(s));
    PK_CHECK(cudaEventDestroy(e));
    return ms;
}

// Converts per-launch times to a rate (work / time) and records median + spread.
inline void record(const std::string& group, const std::string& name, const std::string& unit,
                   double work_per_launch, std::vector<double> ms, const std::string& note = "") {
    Clocks clk = sample_clocks();
    std::vector<double> v;
    for (double t : ms)
        v.push_back(work_per_launch / (t * 1e-3));
    std::sort(v.begin(), v.end());
    Result r{group, name, unit, v[v.size() / 2], (v.back() - v.front()) / v[v.size() / 2], clk, note};
    std::printf("%-8s %-34s %12.2f %-8s spread %5.2f%%  sm %4u MHz  mem %5u MHz  %6.1f W %s\n", group.c_str(),
                name.c_str(), r.value, unit.c_str(), 100.0 * r.spread, clk.sm_mhz, clk.mem_mhz,
                clk.power_mw / 1000.0, note.c_str());
    std::fflush(stdout);
    results().push_back(r);
}

// Latency variant: records the per-launch time itself in microseconds.
inline void record_us(const std::string& group, const std::string& name, std::vector<double> ms,
                      const std::string& note = "") {
    for (double& t : ms)
        t = 1.0 / (t * 1e3);  // record() inverts: work 1 / (1 / us) = us
    record(group, name, "us", 1e-3, ms, note);
}

inline int sm_count() {
    int n = 0;
    PK_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0));
    return n;
}

// Suites, one translation unit each.
void run_mem();
void run_tc();
void run_simt();
void run_launch();

}  // namespace peaks
