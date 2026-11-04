// WSL2/WDDM probe: does cudaMemGetInfo track nvidia-smi as a process allocates? No imp dependencies.
// Build+run: docker run --rm --gpus all -v $PWD:/w -w /w imp:toolchain bash -lc
// 'nvcc -O2 -arch=sm_120a -o /tmp/mgi tools/analysis/wddm_memgetinfo_probe.cu && /tmp/mgi 120'.
// cudaMalloc + memset 256 MiB per step; per step: cudaMemGetInfo used, nvidia-smi memory.used, offset vs
// start. 2026-11-04 RTX 5090: offset 0 up to smi 24514 MiB, +585 MiB from 24770 MiB
// (docs/internals/MEMORY.md).

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static long smi_used_mib() {
    FILE* p = popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", "r");
    long v = -1;
    if (p) {
        if (std::fscanf(p, "%ld", &v) != 1)
            v = -1;
        pclose(p);
    }
    return v;
}

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 100;
    const size_t chunk = 256ull << 20;
    cudaFree(nullptr);
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    const long base_gap = static_cast<long>((t - f) >> 20) - smi_used_mib();
    std::printf("total %zu MiB, start getinfo_used %zu, smi_used %ld, gap %ld\n", t >> 20, (t - f) >> 20,
                smi_used_mib(), base_gap);
    for (int i = 1; i <= steps; ++i) {
        void* p = nullptr;
        if (cudaMalloc(&p, chunk) != cudaSuccess) {
            std::printf("step %d: cudaMalloc failed\n", i);
            break;
        }
        cudaMemset(p, 1, chunk);
        cudaDeviceSynchronize();
        cudaMemGetInfo(&f, &t);
        const long used = static_cast<long>((t - f) >> 20), smi = smi_used_mib();
        std::printf(
            "step %3d alloc %6d MiB getinfo_used %6ld smi_used %6ld getinfo-smi %+5ld (vs start %+ld)\n", i,
            i * 256, used, smi, used - smi, used - smi - base_gap);
        if (f < (1ull << 30))
            break;
    }
    return 0;
}
