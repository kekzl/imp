// tools/analysis/host_transfer_latency.cu — what one MoE-layer host round trip
// actually costs on this box (docs/roadmap.md, "CPU-resident cold experts").
//
// Build + run (needs a free GPU):
//   docker run --rm --gpus all -v $PWD/tools/analysis:/w -w /w imp:toolchain \
//     bash -c "nvcc -O3 -arch=sm_120 host_transfer_latency.cu -o hxl && ./hxl 300"
//
// Split the round trip: how much of it is the two transfers, and how much is
// the kernel launch sitting between them? The cold-expert path has NO GPU
// kernel there — the CPU is what computes — so launch overhead must not be
// billed to the transfer.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t) {
    return std::chrono::duration<double, std::micro>(clk::now() - t).count();
}
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("err %s L%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
__global__ void touch(float* p, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]+=1.0f;}
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}

int main(int argc,char**argv){
    int reps = argc>1?atoi(argv[1]):300;
    size_t bytes = 8*1024;                       // one token's activations
    void* d; CK(cudaMalloc(&d,bytes));
    void* h; CK(cudaHostAlloc(&h,bytes,cudaHostAllocDefault));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    for(int i=0;i<50;i++){CK(cudaMemcpyAsync(d,h,bytes,cudaMemcpyHostToDevice,st));CK(cudaStreamSynchronize(st));}

    std::vector<double> d2h, rt_nokernel, rt_kernel, launch;
    for(int i=0;i<reps;i++){
        auto t=clk::now();
        CK(cudaMemcpyAsync(h,d,bytes,cudaMemcpyDeviceToHost,st)); CK(cudaStreamSynchronize(st));
        d2h.push_back(us_since(t));

        // the cold-expert shape: pull activations out, push the result back.
        // No GPU kernel in between — the host is what computes.
        t=clk::now();
        CK(cudaMemcpyAsync(h,d,bytes,cudaMemcpyDeviceToHost,st)); CK(cudaStreamSynchronize(st));
        CK(cudaMemcpyAsync(d,h,bytes,cudaMemcpyHostToDevice,st)); CK(cudaStreamSynchronize(st));
        rt_nokernel.push_back(us_since(t));

        // same, with a kernel between — separates launch cost out
        t=clk::now();
        CK(cudaMemcpyAsync(h,d,bytes,cudaMemcpyDeviceToHost,st)); CK(cudaStreamSynchronize(st));
        touch<<<8,256,0,st>>>((float*)d,(int)(bytes/4));
        CK(cudaMemcpyAsync(d,h,bytes,cudaMemcpyHostToDevice,st)); CK(cudaStreamSynchronize(st));
        rt_kernel.push_back(us_since(t));

        t=clk::now(); touch<<<8,256,0,st>>>((float*)d,(int)(bytes/4)); CK(cudaStreamSynchronize(st));
        launch.push_back(us_since(t));
    }
    printf("8 KiB, medians over %d reps:\n", reps);
    printf("  single D2H + sync                      %7.1f us\n", med(d2h));
    printf("  D2H + H2D, no kernel (cold-expert)     %7.1f us   <- one MoE layer\n", med(rt_nokernel));
    printf("  D2H + kernel + H2D                     %7.1f us\n", med(rt_kernel));
    printf("  bare kernel launch + sync              %7.1f us\n", med(launch));
    printf("  => 48 layers, cold-expert shape:       %7.2f ms/token\n", med(rt_nokernel)*48/1000.0);
    return 0;
}
