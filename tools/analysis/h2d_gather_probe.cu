// Probe: three ways to move N scattered host-pinned expert slots (~0.9 MiB each) into
// scattered device slots per MoE layer. Measures API cost and effective H2D bandwidth.
//   A: one cudaMemcpyAsync per slot (the expert-cache miss path before 2026-09-20)
//   B: one cudaMemcpyBatchAsync per layer (the miss path since)
//   C: one gather kernel per layer reading mapped pinned memory (zero-copy)
// Args: n_copy n_bufs buf_mb pool_mb mapped(0/1). The in-situ shape of Qwen3.8-Flash-Next
// is 305 pinned buffers of ~180 MiB and a 7.8 GiB pool.
// Build: nvcc -O2 -arch=sm_120a -o /tmp/h2d_gather_probe tools/analysis/h2d_gather_probe.cu
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,                 \
                   cudaGetErrorString(e_));                                                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

constexpr int kMaxCopies = 128;
struct GatherArgs {
    const char* src[kMaxCopies];
    char* dst[kMaxCopies];
};

__global__ void gather_kernel(GatherArgs a, size_t bytes) {
    const int4* s = reinterpret_cast<const int4*>(a.src[blockIdx.y]);
    int4* d = reinterpret_cast<int4*>(a.dst[blockIdx.y]);
    const size_t n = bytes / 16;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        d[i] = s[i];
}

int main(int argc, char** argv) {
    const int n_copy = argc > 1 ? atoi(argv[1]) : 60;
    const int n_bufs = argc > 2 ? atoi(argv[2]) : 1;
    const size_t buf_mb = argc > 3 ? atoi(argv[3]) : 460;
    const size_t pool_mb = argc > 4 ? atoi(argv[4]) : 58;
    const bool mapped = argc > 5 ? atoi(argv[5]) != 0 : true;
    if (n_copy > kMaxCopies) {
        printf("n_copy <= %d\n", kMaxCopies);
        return 1;
    }
    const size_t bytes = 922 * 1024;  // 0.9 MiB, 16-byte multiple
    const int per_buf = static_cast<int>(buf_mb * 1024 * 1024 / bytes);
    const int n_slots = static_cast<int>(pool_mb * 1024 * 1024 / bytes);
    std::vector<char*> hosts(n_bufs), hosts_dev(n_bufs);
    auto ta = std::chrono::steady_clock::now();
    for (int b = 0; b < n_bufs; ++b) {
        CK(cudaHostAlloc(&hosts[b], per_buf * bytes, mapped ? cudaHostAllocMapped : 0));
        for (int e = 0; e < per_buf; ++e)
            memset(hosts[b] + e * bytes, (b * 7 + e) & 0xff, bytes);
        if (mapped)
            CK(cudaHostGetDevicePointer(&hosts_dev[b], hosts[b], 0));
    }
    auto tb = std::chrono::steady_clock::now();
    printf("pinned %d x %zu MiB in %.1f s, pool %zu MiB (%d slots)\n", n_bufs, buf_mb,
           std::chrono::duration<double>(tb - ta).count(), pool_mb, n_slots);
    char* pool = nullptr;
    CK(cudaMalloc(&pool, static_cast<size_t>(n_slots) * bytes));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    srand(7);

    struct Pick {
        int buf, expert, slot;
    };
    auto pick = [&](std::vector<Pick>& p) {
        p.resize(n_copy);
        for (int i = 0; i < n_copy; ++i)
            p[i] = {rand() % n_bufs, rand() % per_buf, rand() % n_slots};
    };
    auto verify = [&](const std::vector<Pick>& p) {
        std::vector<char> buf(bytes);
        for (int i = 0; i < n_copy; ++i) {
            bool dup = false;  // a later copy into the same slot wins
            for (int j = i + 1; j < n_copy; ++j)
                dup |= p[j].slot == p[i].slot;
            if (dup)
                continue;
            CK(cudaMemcpy(buf.data(), pool + static_cast<size_t>(p[i].slot) * bytes, bytes,
                          cudaMemcpyDeviceToHost));
            const char want = (p[i].buf * 7 + p[i].expert) & 0xff;
            if (buf[0] != want || buf[bytes - 1] != want) {
                printf("  VERIFY FAILED at copy %d\n", i);
                return;
            }
        }
    };
    const int reps = 20;
    std::vector<Pick> p;
    using clk = std::chrono::steady_clock;
    auto report = [&](const char* name, double api_ms, double tot_ms) {
        printf("%s x%d: api %.2f ms, total %.2f ms, %.1f GB/s\n", name, n_copy, api_ms / (reps - 2),
               tot_ms / (reps - 2), n_copy * bytes / (tot_ms / (reps - 2)) / 1e6);
    };

    // A: per-slot cudaMemcpyAsync
    {
        double api_ms = 0, tot_ms = 0;
        for (int r = 0; r < reps; ++r) {
            pick(p);
            CK(cudaStreamSynchronize(st));
            auto t0 = clk::now();
            for (int i = 0; i < n_copy; ++i)
                CK(cudaMemcpyAsync(pool + static_cast<size_t>(p[i].slot) * bytes,
                                   hosts[p[i].buf] + static_cast<size_t>(p[i].expert) * bytes, bytes,
                                   cudaMemcpyHostToDevice, st));
            auto t1 = clk::now();
            CK(cudaStreamSynchronize(st));
            auto t2 = clk::now();
            if (r > 1) {
                api_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                tot_ms += std::chrono::duration<double, std::milli>(t2 - t0).count();
            }
        }
        verify(p);
        report("A memcpyAsync", api_ms, tot_ms);
    }
    // B: cudaMemcpyBatchAsync
    {
        double api_ms = 0, tot_ms = 0;
        std::vector<void*> dsts(n_copy);
        std::vector<const void*> srcs(n_copy);
        std::vector<size_t> sizes(n_copy, bytes);
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attr_idx = 0;
        bool ok = true;
        for (int r = 0; r < reps && ok; ++r) {
            pick(p);
            for (int i = 0; i < n_copy; ++i) {
                dsts[i] = pool + static_cast<size_t>(p[i].slot) * bytes;
                srcs[i] = hosts[p[i].buf] + static_cast<size_t>(p[i].expert) * bytes;
            }
            CK(cudaStreamSynchronize(st));
            auto t0 = clk::now();
            cudaError_t e = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), n_copy,
                                                 &attr, &attr_idx, 1, st);
            if (e != cudaSuccess) {
                printf("B cudaMemcpyBatchAsync: %s\n", cudaGetErrorString(e));
                (void)cudaGetLastError();
                ok = false;
                break;
            }
            auto t1 = clk::now();
            CK(cudaStreamSynchronize(st));
            auto t2 = clk::now();
            if (r > 1) {
                api_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                tot_ms += std::chrono::duration<double, std::milli>(t2 - t0).count();
            }
        }
        if (ok) {
            verify(p);
            report("B memcpyBatchAsync", api_ms, tot_ms);
        }
    }
    // C: gather kernel over mapped pinned memory
    if (mapped) {
        for (int blocks_per_copy : {8, 32}) {
            double api_ms = 0, tot_ms = 0;
            GatherArgs a{};
            for (int r = 0; r < reps; ++r) {
                pick(p);
                for (int i = 0; i < n_copy; ++i) {
                    a.src[i] = hosts_dev[p[i].buf] + static_cast<size_t>(p[i].expert) * bytes;
                    a.dst[i] = pool + static_cast<size_t>(p[i].slot) * bytes;
                }
                CK(cudaStreamSynchronize(st));
                auto t0 = clk::now();
                gather_kernel<<<dim3(blocks_per_copy, n_copy), 256, 0, st>>>(a, bytes);
                CK(cudaGetLastError());
                auto t1 = clk::now();
                CK(cudaStreamSynchronize(st));
                auto t2 = clk::now();
                if (r > 1) {
                    api_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    tot_ms += std::chrono::duration<double, std::milli>(t2 - t0).count();
                }
            }
            verify(p);
            char name[64];
            snprintf(name, sizeof name, "C gather kernel (%d blocks/copy)", blocks_per_copy);
            report(name, api_ms, tot_ms);
        }
    }
    return 0;
}
