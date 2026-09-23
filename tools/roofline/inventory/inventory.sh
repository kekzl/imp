#!/usr/bin/env bash
# Kernel inventory: nsys (graphs ON, --cuda-graph-trace=node) over model x workload cells of
# `imp-cli --bench`, per-phase kernel sums via inventory_extract.py (NVTX bench:* ranges).
# Usage: tools/roofline/inventory/inventory.sh [model_key...]   (default: every row of matrix.tsv)
# Env: WORKLOADS="pp512 pp4096 tg128 tg128_ctx8k" IMAGE=imp:test
#      BIN_DIR=<dir> BIN=<file>: profile that imp-cli copy (inside IMAGE) instead of the image's
#      OUT=<dir>: output directory; an existing <cell>.json is skipped
# Output: tools/roofline/inventory/out/<model>__<workload>.{json,log}; .nsys-rep/.sqlite deleted.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${OUT:-$here/out}"
mkdir -p "$out"
chmod 777 "$out"
image="${IMAGE:-imp:test}"
nsys_dir=/opt/nvidia/nsight-systems/2026.1.3
cli=imp-cli
bin_mount=()
if [[ -n "${BIN_DIR:-}" ]]; then
    cli="/bins/${BIN:?BIN_DIR needs BIN}"
    bin_mount=(-v "$BIN_DIR:/bins:ro")
fi
workloads="${WORKLOADS:-pp512 pp4096 tg128 tg128_ctx8k}"

# workload -> imp-cli args (decode arms: spec off, the gate's regime)
wl_args() {
    case "$1" in
        pp512) echo "--bench-pp 512 --max-tokens 1 --bench-reps 3" ;;
        pp4096) echo "--bench-pp 4096 --max-tokens 1 --bench-reps 2" ;;
        tg128) echo "--bench-pp 512 --max-tokens 128 --bench-reps 2 --set speculative.ngram=false" ;;
        tg128_ctx8k) echo "--bench-pp 8192 --max-tokens 128 --bench-reps 1 --set speculative.ngram=false" ;;
        *) echo "unknown workload $1" >&2; return 1 ;;
    esac
}

keys=("$@")
if [[ ${#keys[@]} -eq 0 ]]; then
    mapfile -t keys < <(awk -F'\t' '!/^#/ && NF >= 2 {print $1}' "$here/matrix.tsv")
fi

for key in "${keys[@]}"; do
    row="$(awk -F'\t' -v k="$key" '$1 == k' "$here/matrix.tsv")"
    [[ -z "$row" ]] && { echo "no matrix row: $key" >&2; continue; }
    path="$(cut -f2 <<<"$row")"
    extra="$(cut -f3 <<<"$row")"
    for wl in $workloads; do
        cell="${key}__${wl}"
        [[ -s "$out/$cell.json" ]] && { echo "skip $cell (done)"; continue; }
        if ! "$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" >/dev/null; then
            echo "GPU busy before $cell, stopping" >&2
            exit 3
        fi
        args="$(wl_args "$wl")" || exit 1
        echo "== $cell"
        # shellcheck disable=SC2086
        timeout 1800 docker run --rm --gpus all -u "$(id -u):$(id -g)" -w /tmp \
            -v "$HOME/models:/models:ro" -v "$out:/out" -v /opt/nvidia:/opt/nvidia:ro "${bin_mount[@]}" \
            -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            --entrypoint "$nsys_dir/target-linux-x64/nsys" "$image" profile \
            --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
            --backtrace=none --stats=false --force-overwrite=true -o "/out/$cell" \
            "$cli" --model "/models/$path" --bench $args $extra >"$out/$cell.log" 2>&1
        rc=$?
        grep -E '^(pp|tg) +[0-9]+ tokens' "$out/$cell.log" || echo "rc=$rc, no bench line: $(tail -3 "$out/$cell.log")"
        [[ -f "$out/$cell.nsys-rep" ]] || continue
        docker run --rm -u "$(id -u):$(id -g)" -v "$out:/out" -v /opt/nvidia:/opt/nvidia:ro \
            --entrypoint "$nsys_dir/target-linux-x64/nsys" "$image" export --type sqlite \
            --force-overwrite=true -o "/out/$cell.sqlite" "/out/$cell.nsys-rep" >/dev/null 2>&1
        docker run --rm -u "$(id -u):$(id -g)" -v "$out:/out" -v "$here:/src:ro" python:3.13-slim \
            python /src/inventory_extract.py "/out/$cell.sqlite" "/out/$cell.json" "$cell"
        rm -f "$out/$cell.sqlite" "$out/$cell.nsys-rep"
    done
done
