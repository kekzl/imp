// GPU power and energy for /metrics (roadmap row 121). NVML is loaded with dlopen: no link-time
// dependency, and a host without libnvidia-ml.so.1 emits no power lines instead of failing.

#include "handlers.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

namespace {

using nvml_ret = int;
using nvml_dev = void*;
constexpr nvml_ret kNvmlSuccess = 0;
constexpr auto kSamplePeriod = std::chrono::milliseconds(100);

// One reader per process. Energy comes from the driver counter (mJ since driver load) where
// the GPU reports it, else from integrating power.draw every 100 ms since the first scrape.
class GpuPower {
public:
    static GpuPower& get() {
        static GpuPower p;
        return p;
    }

    // false: NVML or the device is unavailable, emit nothing.
    bool read(double& watts, double& joules) {
        if (!ok_)
            return false;
        unsigned int mw = 0;
        if (power_mw_(dev_, &mw) != kNvmlSuccess)
            return false;
        watts = mw / 1000.0;
        if (energy_mj_) {
            unsigned long long mj = 0;
            if (energy_mj_(dev_, &mj) != kNvmlSuccess)
                return false;
            joules = static_cast<double>(mj) / 1000.0;
        } else {
            std::lock_guard<std::mutex> lk(mu_);
            joules = integrated_j_;
        }
        return true;
    }

    ~GpuPower() {
        stop_.store(true);
        if (sampler_.joinable())
            sampler_.join();
    }

private:
    GpuPower() {
        void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib)
            return;
        auto init = reinterpret_cast<nvml_ret (*)()>(dlsym(lib, "nvmlInit_v2"));
        auto by_pci = reinterpret_cast<nvml_ret (*)(const char*, nvml_dev*)>(
            dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2"));
        power_mw_ = reinterpret_cast<nvml_ret (*)(nvml_dev, unsigned int*)>(
            dlsym(lib, "nvmlDeviceGetPowerUsage"));
        auto energy = reinterpret_cast<nvml_ret (*)(nvml_dev, unsigned long long*)>(
            dlsym(lib, "nvmlDeviceGetTotalEnergyConsumption"));
        if (!init || !by_pci || !power_mw_ || init() != kNvmlSuccess)
            return;
        // The engine runs on CUDA device 0; the PCI bus id maps it to NVML under any
        // CUDA_VISIBLE_DEVICES remapping.
        char bus[32] = {};
        if (cudaDeviceGetPCIBusId(bus, sizeof(bus), 0) != cudaSuccess || by_pci(bus, &dev_) != kNvmlSuccess)
            return;
        unsigned int mw = 0;
        if (power_mw_(dev_, &mw) != kNvmlSuccess)
            return;
        unsigned long long mj = 0;
        if (energy && energy(dev_, &mj) == kNvmlSuccess)
            energy_mj_ = energy;
        else
            sampler_ = std::thread([this] { sample_loop(); });
        ok_ = true;
    }

    void sample_loop() {
        auto prev_t = std::chrono::steady_clock::now();
        unsigned int prev_mw = 0;
        power_mw_(dev_, &prev_mw);
        while (!stop_.load()) {
            std::this_thread::sleep_for(kSamplePeriod);
            unsigned int mw = prev_mw;
            if (power_mw_(dev_, &mw) != kNvmlSuccess)
                mw = prev_mw;
            const auto t = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(t - prev_t).count();
            {
                std::lock_guard<std::mutex> lk(mu_);
                integrated_j_ += 0.5 * (prev_mw + mw) / 1000.0 * dt;  // trapezoid, W x s
            }
            prev_t = t;
            prev_mw = mw;
        }
    }

    bool ok_ = false;
    nvml_dev dev_ = nullptr;
    nvml_ret (*power_mw_)(nvml_dev, unsigned int*) = nullptr;
    nvml_ret (*energy_mj_)(nvml_dev, unsigned long long*) = nullptr;
    std::mutex mu_;
    double integrated_j_ = 0.0;
    std::atomic<bool> stop_{false};
    std::thread sampler_;
};

}  // namespace

void append_power_metrics(std::string& out) {
    double watts = 0.0, joules = 0.0;
    if (!GpuPower::get().read(watts, joules))
        return;
    char buf[64];
    out += "# HELP imp_gpu_power_watts Board power draw of the serving GPU (NVML)\n";
    out += "# TYPE imp_gpu_power_watts gauge\n";
    std::snprintf(buf, sizeof(buf), "imp_gpu_power_watts %.2f\n", watts);
    out += buf;
    out += "# HELP imp_gpu_energy_joules_total Energy drawn by the serving GPU (NVML)\n";
    out += "# TYPE imp_gpu_energy_joules_total counter\n";
    std::snprintf(buf, sizeof(buf), "imp_gpu_energy_joules_total %.3f\n", joules);
    out += buf;
}
