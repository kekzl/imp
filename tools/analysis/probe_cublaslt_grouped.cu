// Probe: does cublasLtMatmul (BATCH_MODE_GROUPED) have any algorithms on
// sm_120 / Consumer Blackwell with cuBLAS 13.4? Re-test of the dead-end recorded
// in MEMORY: "cuBLASLt grouped layout (sm_120): ZERO algorithms for ALL dtypes."
//
// Tests FP16, BF16, FP8 E4M3, NVFP4 (CUDA_R_4F_E2M1) compute paths.
// Output: per-dtype algorithm count via cublasLtMatmulAlgoGetHeuristic.
//
// Build (host):
//   /usr/local/cuda/bin/nvcc -O2 -arch=sm_120a probe_cublaslt_grouped.cu \
//     -lcublasLt -lcublas -o probe_grouped
// Run:
//   ./probe_grouped

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define CK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

#define CK_NORET(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
    } \
} while (0)

static int probe(const char* label, cudaDataType ab_type, cudaDataType c_type,
                 cublasComputeType_t compute_type, cudaDataType scale_type) {
    cublasLtHandle_t lt;
    CK(cublasLtCreate(&lt));

    // Typical MoE prefill shape — 8 active experts, varying token counts.
    constexpr int kGroups = 8;
    int rows_a[kGroups] = {2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};   // K (row of A^T)
    int cols_a[kGroups] = {32, 64, 128, 16, 48, 96, 24, 80};                   // M
    int ld_a[kGroups]   = {2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};

    int rows_b[kGroups] = {2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};   // K
    int cols_b[kGroups] = {1408, 1408, 1408, 1408, 1408, 1408, 1408, 1408};   // N (intermediate)
    int ld_b[kGroups]   = {2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};

    int rows_c[kGroups] = {1408, 1408, 1408, 1408, 1408, 1408, 1408, 1408};
    int cols_c[kGroups];
    for (int i = 0; i < kGroups; i++) cols_c[i] = cols_a[i];
    int ld_c[kGroups]   = {1408, 1408, 1408, 1408, 1408, 1408, 1408, 1408};

    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    CK_NORET(cublasLtGroupedMatrixLayoutCreate(&Adesc, ab_type, kGroups, rows_a, cols_a, ld_a));
    CK_NORET(cublasLtGroupedMatrixLayoutCreate(&Bdesc, ab_type, kGroups, rows_b, cols_b, ld_b));
    CK_NORET(cublasLtGroupedMatrixLayoutCreate(&Cdesc, c_type,  kGroups, rows_c, cols_c, ld_c));
    if (!Adesc || !Bdesc || !Cdesc) {
        printf("%-12s : layout-create failed\n", label);
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtDestroy(lt);
        return 0;
    }

    cublasLtMatmulDesc_t opDesc;
    CK(cublasLtMatmulDescCreate(&opDesc, compute_type, scale_type));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    CK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    cublasLtMatmulPreference_t pref;
    CK(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws_max = 64ull << 20;
    CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &ws_max, sizeof(ws_max)));

    constexpr int kReq = 16;
    cublasLtMatmulHeuristicResult_t results[kReq] = {};
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Adesc, Bdesc, Cdesc, Cdesc, pref,
                                                        kReq, results, &returned);
    printf("%-12s : status=%d  algos=%d\n", label, (int)st, returned);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtDestroy(lt);
    return returned;
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int cublas_ver = 0;
    cublasLtHandle_t tmp;
    cublasLtCreate(&tmp);
    cublasLtDestroy(tmp);
    printf("Device  : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("cuBLAS  : %d (header)\n", CUBLAS_VERSION);
    printf("Probe   : MoE-style 8 groups, varying M ∈ {16..128}, K=2048, N=1408\n\n");

    int n_fp16 = probe("FP16",  CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F,        CUDA_R_16F);
    int n_bf16 = probe("BF16",  CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F,      CUDA_R_32F);
    int n_fp8  = probe("FP8E4M3", CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F,  CUDA_R_32F);
    int n_fp4  = probe("NVFP4",  CUDA_R_4F_E2M1, CUDA_R_16F, CUBLAS_COMPUTE_32F,   CUDA_R_32F);

    printf("\nSummary: FP16=%d BF16=%d FP8=%d NVFP4=%d\n", n_fp16, n_bf16, n_fp8, n_fp4);
    if (n_fp16 + n_bf16 + n_fp8 + n_fp4 == 0) {
        printf("→ Dead end CONFIRMED: cuBLAS 13.4 still has zero grouped algos on sm_120.\n");
        return 1;
    }
    printf("→ Some grouped algos available — integration may be feasible.\n");
    return 0;
}
