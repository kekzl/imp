#!/usr/bin/env bash
# FA2 hd=64 (gpt-oss) tensor share under ncu, clocks locked by ncu (--clock-control base): HMMA flops of the
# pp4096 phase's 2-CTA FA2 calls over the FP16 f32-acc peak scaled to the measured clock (246 TFLOPS at
# 2842 MHz x 170 SMs = 509.2 flops/clk/SM, tools/roofline/PERF_LOG.md:175). #2442.
# Usage: tools/analysis/fa2_hd64_tensor_share.sh BINDIR [OUT]. BINDIR holds an imp-cli built from the tree.
set -euo pipefail
dir="$1"
out="${2:-$HOME/imp-ab/ncu}"
NCU="${NCU:-/opt/nvidia/nsight-compute/2026.2.1/ncu}"
mkdir -p "$out" && chmod 777 "$out"
"$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" > /dev/null || { echo "GPU busy" >&2; exit 1; }
m="gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second"
m="$m,sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum,sm__ops_path_tensor_op_hmma_src_fp16_dst_fp16.sum"
m="$m,sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active"
docker run --rm --gpus all --user root -w /tmp -v "$HOME/models:/models:ro" -v "$out:/out" \
    -v /opt/nvidia:/opt/nvidia:ro -v "$dir":/arm:ro --entrypoint "$NCU" imp:test \
    --clock-control base --nvtx --nvtx-include "bench:pp/" --kernel-name "regex:fmha_sm120_fa2_kernel_2cta" \
    --launch-count 24 --metrics "$m" --csv --page raw \
    /arm/imp-cli --model /models/gpt-oss-20b-mxfp4.gguf --bench --bench-pp 4096 --max-tokens 1 --bench-reps 1 \
    --set runtime.cuda_graphs=never > "$out/fa2_hd64_share.csv" 2> "$out/fa2_hd64_share.err"
docker run --rm -v "$out":/d:ro python:3.13-slim python -I -c "
import csv
rows = list(csv.reader(open('/d/fa2_hd64_share.csv')))
h = next(i for i, r in enumerate(rows) if 'Kernel Name' in r)
c = {n: i for i, n in enumerate(rows[h])}
d = [r for r in rows[h + 2:] if len(r) == len(rows[h])]
g = lambda r, n: float(r[c[n]].replace(',', ''))
fl = [g(r, 'sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum') + g(r, 'sm__ops_path_tensor_op_hmma_src_fp16_dst_fp16.sum') for r in d]
t = [g(r, 'gpu__time_duration.sum') * 1e-9 for r in d]
hz = [g(r, 'sm__cycles_elapsed.avg.per_second') for r in d]
per = 246e12 / (170 * 2.842e9)
share = [f / (ti * z * 170 * per) * 100 for f, ti, z in zip(fl, t, hz)]
print(f'{d[0][c[\"Kernel Name\"]][:72]}')
print(f'calls {len(d)}, kernel sum {sum(t)*1e6:.1f} us, clock {min(hz)/1e6:.0f}..{max(hz)/1e6:.0f} MHz')
print(f'TFLOPS {sum(fl)/sum(t)/1e12:.1f}, share {sum(fl)/sum(ti*z for ti, z in zip(t, hz))/170/per*100:.1f} % (per call {min(share):.1f}..{max(share):.1f} %)')
"
