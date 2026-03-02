#include <chrono>
#include <cstdio>
#include <cstring>

namespace imp {
void bench_gemm();
void bench_attention();
void bench_paged_attention();
void bench_e2e();
} // namespace imp

static void print_usage(const char* prog) {
    printf("Usage: %s [benchmark] [--help]\n\n", prog);
    printf("Available benchmarks:\n");
    printf("  gemm        GEMM micro-benchmark\n");
    printf("  attention   Flash Attention prefill benchmark\n");
    printf("  decode-attn Paged Attention decode benchmark\n");
    printf("  e2e         End-to-end tok/s benchmark\n");
    printf("  all         Run all benchmarks (default)\n");
    printf("\nOptions:\n");
    printf("  --help, -h  Show this help message\n");
}

int main(int argc, char** argv) {
    printf("IMP Benchmark Tool\n");
    printf("==================\n\n");

    // Parse arguments
    bool run_gemm = false;
    bool run_attention = false;
    bool run_decode_attn = false;
    bool run_e2e = false;

    if (argc <= 1) {
        // No arguments: run all benchmarks
        run_gemm = true;
        run_attention = true;
        run_decode_attn = true;
        run_e2e = true;
    } else {
        const char* arg = argv[1];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(arg, "gemm") == 0) {
            run_gemm = true;
        } else if (strcmp(arg, "attention") == 0) {
            run_attention = true;
        } else if (strcmp(arg, "decode-attn") == 0) {
            run_decode_attn = true;
        } else if (strcmp(arg, "e2e") == 0) {
            run_e2e = true;
        } else if (strcmp(arg, "all") == 0) {
            run_gemm = true;
            run_attention = true;
            run_decode_attn = true;
            run_e2e = true;
        } else {
            printf("Unknown benchmark: '%s'\n\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    int benchmarks_run = 0;

    if (run_gemm) {
        imp::bench_gemm();
        ++benchmarks_run;
    }
    if (run_attention) {
        imp::bench_attention();
        ++benchmarks_run;
    }
    if (run_decode_attn) {
        imp::bench_paged_attention();
        ++benchmarks_run;
    }
    if (run_e2e) {
        imp::bench_e2e();
        ++benchmarks_run;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(wall_end - wall_start).count();

    printf("--------------------------------------------------\n");
    printf("Benchmarks run: %d    Total wall time: %.2f s\n", benchmarks_run, total_s);

    return 0;
}
