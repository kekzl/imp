#include "runtime/config.h"
#include "runtime/process_diag.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace imp {
void bench_gemm();
void bench_gemm_nvfp4_cutlass();
void bench_attention();
void bench_paged_attention();
void bench_e2e();
}  // namespace imp

static void print_usage(const char* prog) {
    printf("Usage: %s [benchmark] [--help]\n\n", prog);
    printf("Available benchmarks:\n");
    printf("  gemm        GEMM micro-benchmark\n");
    printf("  nvfp4       Production CUTLASS sm_120 NVFP4 dense GEMM (isolated, ncu target)\n");
    printf("  attention   Flash Attention prefill benchmark\n");
    printf("  decode-attn Paged Attention decode benchmark\n");
    printf("  e2e         End-to-end tok/s benchmark\n");
    printf("  all         Run all benchmarks\n");
    printf("\nDefault (no argument) runs gemm, attention, decode-attn and e2e;\n");
    printf("nvfp4 is excluded (isolated ncu target) — run it via 'nvfp4' or 'all'.\n");
    printf("\nOptions:\n");
    printf("  --config <path>       imp.conf path (default: ./imp.conf, ~/.config/imp/imp.conf)\n");
    printf("  --set <sec.key=val>   Override one imp.conf key (repeatable)\n");
    printf("  --help, -h            Show this help message\n");
}

int main(int argc, char** argv) {
    printf("IMP Benchmark Tool\n");
    printf("==================\n\n");

    bool run_gemm = false;
    bool run_gemm_nvfp4 = false;
    bool run_attention = false;
    bool run_decode_attn = false;
    bool run_e2e = false;

    // Every argument is examined. This used to read argv[1] and nothing else,
    // so `imp-bench gemm --set runtime.deterministic_gemm=true` measured the
    // default configuration while looking like it measured the flag — the same
    // trap imp-cli's main already warns about ("this is how a benchmark ends up
    // measuring a configuration nobody asked for"). A benchmark that silently
    // drops its knobs is worse than one that has none.
    std::string config_path;
    std::vector<std::string> config_overrides;
    const char* benchmark = nullptr;

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(arg, "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(arg, "--set") == 0 && i + 1 < argc) {
            config_overrides.emplace_back(argv[++i]);
        } else if (arg[0] == '-') {
            printf("Unknown option: '%s'\n\n", arg);
            print_usage(argv[0]);
            return 1;
        } else if (benchmark == nullptr) {
            benchmark = arg;
        } else {
            printf("Unexpected argument: '%s' (benchmark already set to '%s')\n\n", arg, benchmark);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Install the config BEFORE any benchmark runs: gemm.cu reads
    // deterministic_gemm through process_diag, which is only populated here.
    {
        std::vector<std::string> rejected_overrides;
        imp::RuntimeConfig cfg = imp::RuntimeConfig::load(config_path, config_overrides, &rejected_overrides);
        if (!rejected_overrides.empty()) {
            for (const auto& bad : rejected_overrides)
                fprintf(stderr, "Error: --set %s\n", bad.c_str());
            fprintf(stderr, "See imp.conf.example for the key names.\n");
            return 1;
        }
        imp::process_diag_install(cfg);
    }

    if (benchmark == nullptr) {
        run_gemm = true;
        run_attention = true;
        run_decode_attn = true;
        run_e2e = true;
    } else if (strcmp(benchmark, "gemm") == 0) {
        run_gemm = true;
    } else if (strcmp(benchmark, "nvfp4") == 0) {
        run_gemm_nvfp4 = true;
    } else if (strcmp(benchmark, "attention") == 0) {
        run_attention = true;
    } else if (strcmp(benchmark, "decode-attn") == 0) {
        run_decode_attn = true;
    } else if (strcmp(benchmark, "e2e") == 0) {
        run_e2e = true;
    } else if (strcmp(benchmark, "all") == 0) {
        run_gemm = true;
        run_gemm_nvfp4 = true;
        run_attention = true;
        run_decode_attn = true;
        run_e2e = true;
    } else {
        printf("Unknown benchmark: '%s'\n\n", benchmark);
        print_usage(argv[0]);
        return 1;
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    int benchmarks_run = 0;

    if (run_gemm) {
        imp::bench_gemm();
        ++benchmarks_run;
    }
    if (run_gemm_nvfp4) {
        imp::bench_gemm_nvfp4_cutlass();
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
