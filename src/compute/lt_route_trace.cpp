#include "compute/lt_route_trace.h"

#include "core/logging.h"
#include "core/process_diag.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace imp {

namespace {

using RouteKey = std::tuple<std::string, std::string, int, int, int, int>;  // route, site, m, n, k, batch

std::mutex s_mutex;
std::map<RouteKey, uint64_t> s_counts;

// "src/compute/gemm.cpp:12" from an absolute or build-relative path (last src/tools/tests component).
std::string site_of(const std::source_location& loc) {
    const std::string f = loc.file_name();
    size_t cut = std::string::npos;
    for (const char* root : {"/src/", "/tools/", "/tests/"}) {
        const size_t p = f.rfind(root);
        if (p != std::string::npos && (cut == std::string::npos || p > cut))
            cut = p;
    }
    return (cut == std::string::npos ? f : f.substr(cut + 1)) + ":" + std::to_string(loc.line());
}

}  // namespace

const char* lt_route_gemm_label(bool fp16_acc, bool a_f16, bool int8) {
    if (int8)
        return "gemm:int8";
    if (fp16_acc)
        return "gemm:f16_acc16";
    return a_f16 ? "gemm:f16_acc32" : "gemm:other";
}

const char* lt_route_fp8_label(bool a_fp8, bool b_fp8) {
    if (!b_fp8)
        return "gemm_cublaslt:other";
    return a_fp8 ? "gemm_cublaslt:fp8xfp8" : "gemm_cublaslt:f16xfp8";
}

void lt_route_note(const char* route, int m, int n, int k, int batch, const std::source_location& loc) {
    if (!process_diag_log_gemm_algo())
        return;
    std::lock_guard<std::mutex> lock(s_mutex);
    s_counts[RouteKey{route, site_of(loc), m, n, k, batch}]++;
}

void lt_route_dump() {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_counts.empty())
        return;
    IMP_LOG_INFO("[lt-route] %zu distinct cuBLASLt calls (route site m n k batch: count)", s_counts.size());
    for (const auto& [key, count] : s_counts) {
        const auto& [route, site, m, n, k, batch] = key;
        IMP_LOG_INFO("[lt-route] %s %s m=%d n=%d k=%d batch=%d: %llu", route.c_str(), site.c_str(), m, n, k,
                     batch, static_cast<unsigned long long>(count));
    }
    s_counts.clear();
}

}  // namespace imp
