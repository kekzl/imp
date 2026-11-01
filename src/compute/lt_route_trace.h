#pragma once

// cuBLASLt route counter (ROADMAP row 123): under diagnostics.log_gemm_algo every cublasLtMatmul
// entry counts (route, caller file:line, m, n, k, batch); gemm_cleanup() logs the table. Host calls:
// a graph-replayed launch counts once at capture. Off: one flag read per Lt call.

#include <source_location>

namespace imp {

void lt_route_note(const char* route, int m, int n, int k, int batch, const std::source_location& loc);

// Logs and clears the table (no-op when empty).
void lt_route_dump();

}  // namespace imp
