// Can an expert promotion be hidden behind compute, or does it stall the layer?
//
// docs/roadmap.md prices promotions as if serialised on the critical path,
// because routing for layer N is known only at layer N. That is the pessimistic
// reading. The optimistic one is a prefetch issued a layer early on a copy
// stream. Which one holds depends on whether copy/compute overlap actually works
// on this WSL2/WDDM box — so measure all three arms, don't assume.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("err %s L%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){return std::chrono::duration<double,std::milli>(clk::now()-t).count();}
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}

// Busy-wait kernel standing in for a layer's compute.
__global__ void spin(long long cycles, float* sink) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (threadIdx.x == 1023456) sink[0] = 1.0f;   // never taken; defeats DCE
}
__global__ void consume(const float* w, float* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i % 256] += w[i];
}

int main(int argc, char** argv) {
    const int LAYERS = 48;
    const size_t EXPERT = 6u << 20;              // ~one expert of a 120B at 4 bit
    const int reps = (argc>1)?atoi(argv[1]):20;
    // clocks for a target per-layer compute time, calibrated below
    float* d_sink; CK(cudaMalloc(&d_sink, 1024*sizeof(float)));
    float* d_w;    CK(cudaMalloc(&d_w, EXPERT));
    float* d_out;  CK(cudaMalloc(&d_out, 256*sizeof(float)));
    void*  h_w;    CK(cudaHostAlloc(&h_w, EXPERT, cudaHostAllocDefault));

    cudaStream_t comp, copy; CK(cudaStreamCreate(&comp)); CK(cudaStreamCreate(&copy));
    std::vector<cudaEvent_t> ev(LAYERS);
    for (auto& evt : ev) CK(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));

    // calibrate: cycles for ~100 us
    // Calibrate over MANY launches: one launch is mostly launch overhead, so a
    // single-shot calibration set the spin an order of magnitude too short and
    // the baseline came out 5x faster than the target.
    long long cyc = 100000;
    for (int i=0;i<4;i++){
        const int K=200;
        auto t=clk::now();
        for(int k=0;k<K;k++) spin<<<1,32,0,comp>>>(cyc,d_sink);
        CK(cudaStreamSynchronize(comp));
        double per = ms(t)/K;
        cyc = (long long)(cyc * (0.100/(per>1e-6?per:0.1)));
    }
    printf("calibrated: %lld cycles ~= 100 us/layer of compute\n\n", cyc);

    for (double dur_us : {50.0, 100.0, 200.0, 400.0}) {
        long long c = (long long)(cyc * dur_us / 100.0);
        std::vector<double> base, naive, pref;
        for (int r=0;r<reps;r++){
            // (C) baseline: compute only, no promotion
            auto t=clk::now();
            for(int l=0;l<LAYERS;l++) spin<<<1,32,0,comp>>>(c,d_sink);
            CK(cudaStreamSynchronize(comp)); base.push_back(ms(t));

            // (A) naive: promotion issued on the compute stream, in front of the
            //     consumer — the pessimistic reading, fully serialised
            t=clk::now();
            for(int l=0;l<LAYERS;l++){
                spin<<<1,32,0,comp>>>(c,d_sink);
                CK(cudaMemcpyAsync(d_w,h_w,EXPERT,cudaMemcpyHostToDevice,comp));
                consume<<<64,256,0,comp>>>(d_w,d_out,64*256);
            }
            CK(cudaStreamSynchronize(comp)); naive.push_back(ms(t));

            // (B) prefetch: promotion for layer l+1 issued on a copy stream at
            //     the start of layer l, consumer waits on the event
            t=clk::now();
            CK(cudaMemcpyAsync(d_w,h_w,EXPERT,cudaMemcpyHostToDevice,copy));
            CK(cudaEventRecord(ev[0],copy));
            for(int l=0;l<LAYERS;l++){
                spin<<<1,32,0,comp>>>(c,d_sink);
                if(l+1<LAYERS){
                    CK(cudaMemcpyAsync(d_w,h_w,EXPERT,cudaMemcpyHostToDevice,copy));
                    CK(cudaEventRecord(ev[l+1],copy));
                }
                CK(cudaStreamWaitEvent(comp,ev[l],0));
                consume<<<64,256,0,comp>>>(d_w,d_out,64*256);
            }
            CK(cudaStreamSynchronize(comp)); pref.push_back(ms(t));
        }
        // realistic: only ~42% of layers promote anything (measured LRU miss rate)
        std::vector<double> real;
        for(int r=0;r<reps;r++){
            auto t=clk::now();
            int pending=0;
            for(int l=0;l<LAYERS;l++){
                bool promote = ((l*42)/100) != (((l+1)*42)/100);
                if(promote){ CK(cudaMemcpyAsync(d_w,h_w,EXPERT,cudaMemcpyHostToDevice,copy));
                             CK(cudaEventRecord(ev[l],copy)); pending=l; }
                spin<<<1,32,0,comp>>>(c,d_sink);
                if(promote){ CK(cudaStreamWaitEvent(comp,ev[pending],0));
                             consume<<<64,256,0,comp>>>(d_w,d_out,64*256); }
            }
            CK(cudaStreamSynchronize(comp)); real.push_back(ms(t));
        }
        double b=med(base), a=med(naive), p=med(pref), rr=med(real);
        printf("  compute %5.0f us/layer: base %6.2f | naive %6.2f (+%5.2f) | prefetch-all %6.2f (+%5.2f) | prefetch@42%% %6.2f (+%5.2f)\n",
               dur_us, b, a, a-b, p, p-b, rr, rr-b);
    }
    return 0;
}
