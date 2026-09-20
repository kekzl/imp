// Does cuBLASLt offer grouped-batch algos on sm_120, and does it offer them for NVFP4?
//
// docs/internals/SM120.md records "zero grouped algos returned on sm_120" and asks for a
// re-probe on every toolkit release. This is that probe: it asks cublasLtMatmulAlgoGetHeuristic
// how many algos it returns for four configurations, so a zero is readable against a non-zero
// control instead of standing alone.
//
//   FP16   plain      control: the heuristic works at all
//   FP16   grouped    control: grouped batch mode works at all on this card
//   NVFP4  plain      control: block-scaled NVFP4 works (CUTLASS path exists, cuBLASLt may too)
//   NVFP4  grouped    the question
//
// Build and run (needs a GPU):
//   docker run --rm --gpus all -v $PWD:/src -w /src nvidia/cuda:13.4.1-devel-ubuntu26.04 bash -c \
//     'nvcc -O2 -arch=sm_120a tools/analysis/cublaslt_grouped_probe.cu -lcublasLt -o /tmp/probe && /tmp/probe'
//
// Exit code is 0 whatever the outcome: the printed table is the result, a zero row is a finding,
// not an error.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

struct Probe {
    const char* name;
    cudaDataType_t ab_type;
    bool grouped;
    bool block_scaled;
};

// One heuristic query. Returns the algo count, or -1 when a setup call itself failed.
int algo_count(const Probe& p) {
    cublasLtHandle_t lt = nullptr;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) return -1;

    const int64_t M = 1024, N = 1024, K = 1024;
    cublasLtMatmulDesc_t desc = nullptr;
    // FP32 accumulate for both; NVFP4 has no other compute type on this path.
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        cublasLtDestroy(lt);
        return -1;
    }

    const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    if (p.block_scaled) {
        // NVFP4: one UE4M3 scale per 16 elements, the layout the CUTLASS path also produces.
        const int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    }

    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatrixLayoutCreate(&la, p.ab_type, K, M, K);
    cublasLtMatrixLayoutCreate(&lb, p.ab_type, K, N, K);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, M, N, M);

    // Device-side shape arrays: grouped mode reads rows/cols/ld from them, one entry per group.
    const int kGroups = 4;
    int64_t *d_rows = nullptr, *d_cols = nullptr, *d_ld = nullptr;
    if (p.grouped) {
        const int64_t rows[kGroups] = {M, M, M, M};
        const int64_t cols[kGroups] = {N, N, N, N};
        const int64_t ld[kGroups] = {K, K, K, K};
        cudaMalloc(&d_rows, sizeof(rows));
        cudaMalloc(&d_cols, sizeof(cols));
        cudaMalloc(&d_ld, sizeof(ld));
        cudaMemcpy(d_rows, rows, sizeof(rows), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cols, cols, sizeof(cols), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ld, ld, sizeof(ld), cudaMemcpyHostToDevice);

        const int32_t mode = CUBLASLT_BATCH_MODE_GROUPED;
        const int32_t width = CUBLASLT_INTEGER_WIDTH_64;
        const int32_t count = kGroups;
        for (cublasLtMatrixLayout_t l : {la, lb, lc}) {
            cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &mode, sizeof(mode));
            cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count));
            cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY, &d_rows, sizeof(d_rows));
            cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY, &d_cols, sizeof(d_cols));
            cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY, &d_ld, sizeof(d_ld));
            cublasLtMatrixLayoutSetAttribute(
                l, CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH, &width, sizeof(width));
            cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH, &width,
                                             sizeof(width));
        }
    }

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    const size_t ws = 32u << 20;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    cublasLtMatmulHeuristicResult_t results[16] = {};
    int returned = 0;
    const cublasStatus_t st =
        cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 16, results, &returned);
    if (st != CUBLAS_STATUS_SUCCESS) returned = 0;

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(lt);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_ld);
    return returned;
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    size_t ver = 0;
    ver = cublasLtGetVersion();
    printf("device: %s (sm_%d%d), cuBLASLt %zu\n\n", prop.name, prop.major, prop.minor, ver);

    const Probe probes[] = {
        {"FP16  plain  ", CUDA_R_16F, false, false},
        {"FP16  grouped", CUDA_R_16F, true, false},
        {"NVFP4 plain  ", CUDA_R_4F_E2M1, false, true},
        {"NVFP4 grouped", CUDA_R_4F_E2M1, true, true},
    };
    printf("%-14s  algos\n", "configuration");
    for (const Probe& p : probes) printf("%-14s  %5d\n", p.name, algo_count(p));
    return 0;
}
