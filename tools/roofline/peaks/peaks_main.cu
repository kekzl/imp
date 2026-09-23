// imp-peaks: measured sm_120a limits. Usage: imp-peaks [--json out.json] [mem|tc|simt|launch]...
#include "peaks_common.cuh"

#include <cstring>
#include <fstream>

namespace {

void write_json(const char* path) {
    std::ofstream o(path);
    cudaDeviceProp p{};
    PK_CHECK(cudaGetDeviceProperties(&p, 0));
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    o << "{\n  \"device\": \"" << p.name << "\", \"sms\": " << p.multiProcessorCount
      << ", \"l2_bytes\": " << p.l2CacheSize << ", \"cuda_driver\": " << drv << ", \"cuda_runtime\": " << rt
      << ",\n  \"results\": [\n";
    const auto& r = peaks::results();
    for (size_t i = 0; i < r.size(); ++i) {
        o << "    {\"group\": \"" << r[i].group << "\", \"name\": \"" << r[i].name << "\", \"unit\": \""
          << r[i].unit << "\", \"value\": " << r[i].value << ", \"spread\": " << r[i].spread
          << ", \"sm_mhz\": " << r[i].clk.sm_mhz << ", \"mem_mhz\": " << r[i].clk.mem_mhz
          << ", \"power_w\": " << r[i].clk.power_mw / 1000.0 << ", \"note\": \"" << r[i].note << "\"}"
          << (i + 1 < r.size() ? ",\n" : "\n");
    }
    o << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    const char* json = nullptr;
    bool mem = false, tc = false, simt = false, launch = false, any = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--json") && i + 1 < argc) {
            json = argv[++i];
            continue;
        }
        any = true;
        mem |= !std::strcmp(argv[i], "mem");
        tc |= !std::strcmp(argv[i], "tc");
        simt |= !std::strcmp(argv[i], "simt");
        launch |= !std::strcmp(argv[i], "launch");
    }
    if (!any)
        mem = tc = simt = launch = true;
    peaks::sample_clocks();
    if (mem)
        peaks::run_mem();
    if (tc)
        peaks::run_tc();
    if (simt)
        peaks::run_simt();
    if (launch)
        peaks::run_launch();
    if (json != nullptr)
        write_json(json);
    return 0;
}
