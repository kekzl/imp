// Launch overhead: back-to-back empty kernels per launch, as plain stream launches, as one CUDA
// graph replay of the same chain, and as a PDL chain (programmatic stream serialization).
#include "peaks_common.cuh"

#include <cstdint>

namespace peaks {
namespace {

constexpr int kChain = 256;

__global__ void empty_kernel(uint32_t* p) {
    if (p != nullptr && threadIdx.x == 0 && blockIdx.x == 0 && p[1] == 0x9e3779b9u)
        p[0] = 1;
}

__global__ void pdl_kernel(uint32_t* p) {
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n");
    if (threadIdx.x == 0 && blockIdx.x == 0 && p[1] == 0x9e3779b9u)
        p[0] = 1;
}

void launch_pdl(cudaStream_t s, int grid, uint32_t* p) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(128);
    cfg.stream = s;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    PK_CHECK(cudaLaunchKernelEx(&cfg, pdl_kernel, p));
}

}  // namespace

void run_launch() {
    uint32_t* p = nullptr;
    PK_CHECK(cudaMalloc(&p, 64));
    PK_CHECK(cudaMemset(p, 0, 64));
    cudaStream_t s;
    PK_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    for (int grid : {1, 170}) {
        const std::string g = " grid=" + std::to_string(grid);
        // time_launches uses the legacy stream; run each variant on it for identical timing.
        auto plain = time_launches([&] { empty_kernel<<<grid, 128>>>(p); }, kChain);
        record_us("launch", "plain" + g, plain, "per kernel, stream order");

        auto pdl = time_launches([&] { launch_pdl(0, grid, p); }, kChain);
        record_us("launch", "pdl chain" + g, pdl, "per kernel");

        cudaGraph_t graph;
        cudaGraphExec_t exec;
        PK_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < kChain; ++i)
            empty_kernel<<<grid, 128, 0, s>>>(p);
        PK_CHECK(cudaStreamEndCapture(s, &graph));
        PK_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        auto gr = time_launches([&] { PK_CHECK(cudaGraphLaunch(exec, 0)); }, 8);
        for (double& t : gr)
            t /= kChain;
        record_us("launch", "graph node" + g, gr, "per kernel inside a 256-node graph");

        PK_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < kChain; ++i)
            launch_pdl(s, grid, p);
        PK_CHECK(cudaStreamEndCapture(s, &graph));
        cudaGraphExec_t exec_pdl;
        PK_CHECK(cudaGraphInstantiate(&exec_pdl, graph, 0));
        auto gp = time_launches([&] { PK_CHECK(cudaGraphLaunch(exec_pdl, 0)); }, 8);
        for (double& t : gp)
            t /= kChain;
        record_us("launch", "graph+pdl node" + g, gp, "per kernel inside a 256-node graph");
        PK_CHECK(cudaGraphExecDestroy(exec));
        PK_CHECK(cudaGraphExecDestroy(exec_pdl));
    }
    PK_CHECK(cudaStreamDestroy(s));
    PK_CHECK(cudaFree(p));
}

}  // namespace peaks
