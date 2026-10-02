// Does cuBLASLt offer grouped-batch algos on sm_120, and does it offer them for NVFP4?
//
// docs/internals/SM120.md records the grouped result and asks for a re-probe on every toolkit
// release. This probe asks cublasLtMatmulAlgoGetHeuristic how many algos it returns per
// configuration and prints the heuristic status, so a zero is readable against non-zero controls.
//
// NVFP4 setup per the cuBLAS 13.x docs ("To use block-scaled FP4 kernels"): A transposed, B not
// (TN), CUBLAS_COMPUTE_32F, scale type CUDA_R_32F, CUBLASLT_MATMUL_DESC_{A,B}_SCALE_MODE =
// CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, 16-byte aligned dims; C/D 16F, 16BF or 32F (Table 4).
// Rows marked "+ptr" also set the A/B scale pointers to device buffers of the tiled-layout size.
//
// Build and run (needs a GPU):
//   docker run --rm --gpus all -v $PWD:/src -w /src imp:toolchain bash -c \
//     'nvcc -O2 -arch=sm_120a tools/analysis/cublaslt_grouped_probe.cu -lcublasLt -o /tmp/probe \
//      && /tmp/probe'
//
// Exit code is 0 whatever the outcome: the printed table is the result, a zero row is a finding.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr int32_t kNoScale = -1;

struct Probe {
    const char* name;
    cudaDataType_t ab_type;
    cudaDataType_t cd_type;
    int32_t scale_mode;  // CUBLASLT_MATMUL_MATRIX_SCALE_*, kNoScale = library default
    bool grouped;
    bool scale_pointers;
};

struct Result {
    int algos = 0;
    cublasStatus_t heuristic = CUBLAS_STATUS_SUCCESS;
    cublasStatus_t first_setup_error = CUBLAS_STATUS_SUCCESS;  // first failed create/set call
};

Result probe(const Probe& p) {
    Result r;
    auto check = [&r](cublasStatus_t s) {
        if (s != CUBLAS_STATUS_SUCCESS && r.first_setup_error == CUBLAS_STATUS_SUCCESS)
            r.first_setup_error = s;
    };
    cublasLtHandle_t lt = nullptr;
    check(cublasLtCreate(&lt));

    const int64_t M = 1024, N = 1024, K = 1024;
    cublasLtMatmulDesc_t desc = nullptr;
    check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

    void *d_sa = nullptr, *d_sb = nullptr;
    if (p.scale_mode != kNoScale) {
        check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &p.scale_mode,
                                             sizeof(p.scale_mode)));
        check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &p.scale_mode,
                                             sizeof(p.scale_mode)));
    }
    if (p.scale_pointers) {
        // One byte per 16 (NVFP4) or 32 (MXFP8) elements, rounded up to whole 128x4 tiles; 1 MiB covers both.
        cudaMalloc(&d_sa, 1 << 20);
        cudaMalloc(&d_sb, 1 << 20);
        check(
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sa, sizeof(d_sa)));
        check(
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_sb, sizeof(d_sb)));
    }

    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    check(cublasLtMatrixLayoutCreate(&la, p.ab_type, K, M, K));
    check(cublasLtMatrixLayoutCreate(&lb, p.ab_type, K, N, K));
    check(cublasLtMatrixLayoutCreate(&lc, p.cd_type, M, N, M));

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
            check(
                cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &mode, sizeof(mode)));
            check(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count,
                                                   sizeof(count)));
            check(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY, &d_rows,
                                                   sizeof(d_rows)));
            check(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY, &d_cols,
                                                   sizeof(d_cols)));
            check(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY, &d_ld,
                                                   sizeof(d_ld)));
            check(cublasLtMatrixLayoutSetAttribute(
                l, CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH, &width, sizeof(width)));
            check(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
                                                   &width, sizeof(width)));
        }
    }

    cublasLtMatmulPreference_t pref = nullptr;
    check(cublasLtMatmulPreferenceCreate(&pref));
    const size_t ws = 32u << 20;
    check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws,
                                               sizeof(ws)));

    cublasLtMatmulHeuristicResult_t results[16] = {};
    r.heuristic = cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 16, results, &r.algos);
    if (r.heuristic != CUBLAS_STATUS_SUCCESS)
        r.algos = 0;

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(lt);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_ld);
    cudaFree(d_sa);
    cudaFree(d_sb);
    return r;
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d), cuBLASLt %zu\n\n", prop.name, prop.major, prop.minor, cublasLtGetVersion());

    const int32_t nvfp4 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    const int32_t mxfp8 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    const int32_t scalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    const Probe probes[] = {
        {"FP16  plain           ", CUDA_R_16F, CUDA_R_16F, kNoScale, false, false},
        {"FP16  grouped         ", CUDA_R_16F, CUDA_R_16F, kNoScale, true, false},
        {"FP8   plain  scalar   ", CUDA_R_8F_E4M3, CUDA_R_16BF, scalar, false, false},
        {"MXFP8 plain  D16BF    ", CUDA_R_8F_E4M3, CUDA_R_16BF, mxfp8, false, false},
        {"MXFP8 plain  D16BF+ptr", CUDA_R_8F_E4M3, CUDA_R_16BF, mxfp8, false, true},
        {"NVFP4 plain  D16F     ", CUDA_R_4F_E2M1, CUDA_R_16F, nvfp4, false, false},
        {"NVFP4 plain  D16F+ptr ", CUDA_R_4F_E2M1, CUDA_R_16F, nvfp4, false, true},
        {"NVFP4 plain  D16BF+ptr", CUDA_R_4F_E2M1, CUDA_R_16BF, nvfp4, false, true},
        {"NVFP4 plain  D32F+ptr ", CUDA_R_4F_E2M1, CUDA_R_32F, nvfp4, false, true},
        {"NVFP4 grouped D16F+ptr", CUDA_R_4F_E2M1, CUDA_R_16F, nvfp4, true, true},
    };
    printf("%-22s  %5s  %-28s  %s\n", "configuration", "algos", "heuristic status", "first setup error");
    for (const Probe& p : probes) {
        const Result r = probe(p);
        printf("%-22s  %5d  %-28s  %s\n", p.name, r.algos, cublasLtGetStatusName(r.heuristic),
               r.first_setup_error == CUBLAS_STATUS_SUCCESS ? "-"
                                                            : cublasLtGetStatusName(r.first_setup_error));
    }
    return 0;
}
