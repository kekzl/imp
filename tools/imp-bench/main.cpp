#include "common/exit_codes.h"
#include "common/json_out.h"
#include "runtime/config.h"
#include "runtime/process_diag.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace imp {
// bool, not void (#1584): each of these returns early when there is no CUDA
// device, and every one of them used to do that silently into an exit 0. A
// benchmark that measured nothing is not a successful benchmark run - it is
// the one result a CI job or a shell script must be able to see.
bool bench_gemm();
bool bench_gemm_nvfp4_cutlass();
bool bench_attention();
bool bench_paged_attention();
bool bench_e2e();
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
    printf("  --json                One JSON document on stdout, tables on stderr\n");
    printf("  --help, -h            Show this help message\n");
}

int main(int argc, char** argv) {
    // Pre-scan for --json: the banner below is the first thing on stdout, and
    // with --json stdout belongs to the single JSON document (#1583).
    bool json_out = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) {
            json_out = true;
            imp_tools::json_stdout_reserve();
            break;
        }
    }

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
        } else if (strcmp(arg, "--json") == 0) {
            // Consumed by the pre-scan above; accepted here so it is not an
            // "unknown option".
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

    // Requested against measured. The old counter incremented per INVOCATION,
    // so "Benchmarks run: 4" was printed by a host with no GPU that ran none.
    int benchmarks_run = 0;
    int benchmarks_requested = 0;

    std::string entries;  // rendered JSON array elements, in run order

    auto run_one = [&](bool wanted, const char* name, bool (*fn)()) {
        if (!wanted)
            return;
        ++benchmarks_requested;
        const auto t0 = std::chrono::high_resolution_clock::now();
        const bool measured = fn();
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (measured)
            ++benchmarks_run;
        if (!json_out)
            return;
        imp_tools::JsonOut e;
        e.str("name", name)
            .boolean("measured", measured)
            .num("seconds", std::chrono::duration<double>(t1 - t0).count(), 3);
        if (!entries.empty())
            entries += ',';
        entries += e.str();
    };

    run_one(run_gemm, "gemm", imp::bench_gemm);
    run_one(run_gemm_nvfp4, "nvfp4", imp::bench_gemm_nvfp4_cutlass);
    run_one(run_attention, "attention", imp::bench_attention);
    run_one(run_decode_attn, "decode-attn", imp::bench_paged_attention);
    run_one(run_e2e, "e2e", imp::bench_e2e);

    auto wall_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(wall_end - wall_start).count();

    printf("--------------------------------------------------\n");
    printf("Benchmarks run: %d of %d requested    Total wall time: %.2f s\n", benchmarks_run,
           benchmarks_requested, total_s);

    if (json_out) {
        // Per-benchmark timings and the measured flag, not the tables: the five
        // bench entry points return bool, and the numbers inside them (GFLOPS,
        // tok/s, per-shape rows) have no shared shape to serialise. The
        // consumer that needed machine-readable throughput is
        // scripts/gen_perf_baseline.sh, and it reads `imp-cli --bench --json`.
        imp_tools::JsonOut j;
        j.str("mode", "bench-suite")
            .intg("requested", benchmarks_requested)
            .intg("run", benchmarks_run)
            .num("wall_s", total_s, 2)
            .key("benchmarks");
        std::string doc = j.str();
        doc.pop_back();  // the trailing '}' - the array closes the document
        doc += "[" + entries + "]}";
        imp_tools::json_emit(doc);
    }

    if (benchmarks_run < benchmarks_requested) {
        fprintf(stderr, "imp-bench: %d of %d benchmark(s) measured nothing\n",
                benchmarks_requested - benchmarks_run, benchmarks_requested);
        return imp::tools::exit_code_for(IMP_ERROR_INTERNAL);
    }
    return 0;
}
