#!/usr/bin/env bash
# ncu counters for the kernels of one inventory cell, restricted to the measured NVTX phase.
# Usage: tools/roofline/inventory/ncu_cell.sh <model_key> <workload> <kernel-regex> [launch_count=40] [mode]
#   mode full   : SpeedOfLight + WarpStateStats + Occupancy + raw bytes/tensor counters (multi-pass)
#   mode single : one pass, raw counters only (TMA kernels: multi-pass replay dies on this WSL2 driver)
# Clocks: --clock-control none, the card must run locked (tools/roofline/peaks/PEAKS.md).
# Output: out/ncu/<model>__<workload>__<tag>.csv (ncu --page raw).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
key="$1" wl="$2" rx="$3" count="${4:-40}" mode="${5:-full}"
out="$here/out/ncu"
mkdir -p "$out"
chmod 777 "$out"
row="$(awk -F'\t' -v k="$key" '$1 == k' "$here/matrix.tsv")"
[[ -z "$row" ]] && { echo "no matrix row: $key" >&2; exit 1; }
path="$(cut -f2 <<<"$row")"
extra="$(cut -f3 <<<"$row")"
case "$wl" in
    pp512) args="--bench-pp 512 --max-tokens 1 --bench-reps 1"; phase="bench:pp/" ;;
    pp4096) args="--bench-pp 4096 --max-tokens 1 --bench-reps 1"; phase="bench:pp/" ;;
    tg128) args="--bench-pp 512 --max-tokens 128 --bench-reps 1 --set speculative.ngram=false"; phase="bench:tg/" ;;
    tg128_ctx8k) args="--bench-pp 8192 --max-tokens 128 --bench-reps 1 --set speculative.ngram=false"; phase="bench:tg/" ;;
    *) echo "unknown workload $wl" >&2; exit 1 ;;
esac
raw="dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,smsp__inst_executed_pipe_tensor.sum,gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second"
if [[ "$mode" == full ]]; then
    sel=(--section SpeedOfLight --section WarpStateStats --section Occupancy --section LaunchStats
         --metrics "$raw,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")
else
    sel=(--metrics "$raw")
fi
tag="$(tr -c 'A-Za-z0-9_' '_' <<<"$rx" | cut -c1-40)_$mode"
if ! "$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" >/dev/null; then
    echo "GPU busy" >&2
    exit 3
fi
# shellcheck disable=SC2086
timeout 1800 docker run --rm --gpus all --user root -w /tmp \
    -v "$HOME/models:/models:ro" -v "$out:/out" -v /opt/nvidia:/opt/nvidia:ro \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 --entrypoint /opt/nvidia/nsight-compute/2026.2.1/ncu "imp:test" \
    --clock-control none --nvtx --nvtx-include "$phase" --kernel-name "regex:$rx" \
    --launch-count "$count" "${sel[@]}" --csv --page raw \
    imp-cli --model "/models/$path" --bench $args --set runtime.cuda_graphs=never $extra \
    >"$out/${key}__${wl}__${tag}.csv" 2>"$out/${key}__${wl}__${tag}.err"
rc=$?
echo "rc=$rc rows=$(($(wc -l <"$out/${key}__${wl}__${tag}.csv") - 2)) -> $out/${key}__${wl}__${tag}.csv"
