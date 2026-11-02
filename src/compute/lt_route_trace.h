#pragma once

// cuBLASLt route counter (ROADMAP row 123): under diagnostics.log_gemm_algo every cublasLtMatmul
// entry counts (route, caller file:line, m, n, k, batch); gemm_cleanup() logs the table. Host calls:
// a graph-replayed launch counts once at capture. Off: one flag read per Lt call.

#include <source_location>

namespace imp {

// Route labels: gemm() cached path ("gemm:int8" > "gemm:f16_acc16" > "gemm:f16_acc32" > "gemm:other"),
// gemm_cublaslt ("gemm_cublaslt:fp8xfp8" / "f16xfp8" / "other").
[[nodiscard]] const char* lt_route_gemm_label(bool fp16_acc, bool a_f16, bool int8);
[[nodiscard]] const char* lt_route_fp8_label(bool a_fp8, bool b_fp8);

void lt_route_note(const char* route, int m, int n, int k, int batch, const std::source_location& loc);

// Logs and clears the table (no-op when empty).
void lt_route_dump();

}  // namespace imp
