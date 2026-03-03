#!/usr/bin/env bash
# benchmark.sh — Run imp vs llama.cpp throughput benchmarks across models.
# Produces formatted table (stdout), JSON, and Markdown report.
#
# Usage: ./tools/benchmark.sh [options]
#
# Options:
#   --scenario SCENARIO  short|standard|long|all (default: standard)
#   --reps N             Benchmark repetitions (default: 3)
#   --pp N               Override prefill tokens (overrides scenario)
#   --tg N               Override decode tokens (overrides scenario)
#   --model NAME         Run only models matching NAME (substring)
#   --no-llama           Skip llama-bench even if available
#   --nvfp4              Include NVFP4 decode mode comparison
#   --fp8-kv             Include FP8 KV cache mode comparison
#   --quick              Quick mode: 1 rep, scenario=short
#   --output-dir DIR     Output directory (default: benchmarks/results)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$ROOT_DIR/build/imp-cli"
LLAMA_BENCH="${LLAMA_BENCH:-$(command -v llama-bench 2>/dev/null || echo "${HOME}/llama.cpp/build/bin/llama-bench")}"
MODELS_DIR="$ROOT_DIR/models"

# ── Defaults ──────────────────────────────────────────────────────────────
REPS=3
SCENARIO="standard"
PP_OVERRIDE=""
TG_OVERRIDE=""
FILTER=""
NO_LLAMA=0
WITH_NVFP4=0
WITH_FP8KV=0
OUTPUT_DIR="$ROOT_DIR/benchmarks/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)   SCENARIO="$2"; shift 2 ;;
        --reps)       REPS="$2"; shift 2 ;;
        --pp)         PP_OVERRIDE="$2"; shift 2 ;;
        --tg)         TG_OVERRIDE="$2"; shift 2 ;;
        --model)      FILTER="$2"; shift 2 ;;
        --no-llama)   NO_LLAMA=1; shift ;;
        --nvfp4)      WITH_NVFP4=1; shift ;;
        --fp8-kv)     WITH_FP8KV=1; shift ;;
        --quick)      REPS=1; SCENARIO="short"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Scenario definitions ──────────────────────────────────────────────────
# Each scenario: pp,tg
declare -A SCENARIOS=(
    [short]="128,128"
    [standard]="512,256"
    [long]="2048,512"
)

build_scenario_list() {
    if [[ "$SCENARIO" == "all" ]]; then
        echo "short standard long"
    else
        if [[ -z "${SCENARIOS[$SCENARIO]+x}" ]]; then
            echo "ERROR: Unknown scenario '$SCENARIO'. Use: short|standard|long|all" >&2
            exit 1
        fi
        echo "$SCENARIO"
    fi
}

SCENARIO_LIST=$(build_scenario_list)

# ── Validate binaries ─────────────────────────────────────────────────────
if [[ ! -x "$CLI" ]]; then
    echo "ERROR: imp-cli not found at $CLI — run cmake --build build first"
    exit 1
fi

HAS_LLAMA=0
if [[ $NO_LLAMA -eq 0 && -x "$LLAMA_BENCH" ]]; then
    HAS_LLAMA=1
fi

# ── System info ───────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "?")
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' || echo "?")
OS_INFO=$(uname -sr)
TIMESTAMP=$(date '+%Y-%m-%dT%H:%M:%S')
TIMESTAMP_SHORT=$(date '+%Y-%m-%d_%H-%M')

IMP_COMMIT=$(cd "$ROOT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "?")
LLAMA_COMMIT="—"
if [[ $HAS_LLAMA -eq 1 ]]; then
    llama_dir=$(dirname "$(dirname "$LLAMA_BENCH")")
    LLAMA_COMMIT=$(cd "$llama_dir" && git rev-parse --short HEAD 2>/dev/null || echo "?")
fi

# ── Model registry ────────────────────────────────────────────────────────
# Format: short_name|dir_glob|extra_flags
MODELS=(
    "Phi-4-mini Q8_0|models--unsloth--Phi-4-mini-instruct-GGUF|"
    "Qwen3-4B Q8_0|models--unsloth--Qwen3-4B-Instruct-2507-GGUF|"
    "DS-R1-7B Q8_0|models--unsloth--DeepSeek-R1-Distill-Qwen-7B-GGUF|"
    "Gemma-3-12B Q8_0|models--unsloth--gemma-3-12b-it-GGUF|--chat-template gemma"
    "DS-R1-14B Q6_K|models--unsloth--DeepSeek-R1-Distill-Qwen-14B-GGUF|"
    "Qwen3-Coder-30B-A3B Q6_K|models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF|"
    "Nemotron-30B-A3B Q6_K|models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF|"
)

# ── Helpers ───────────────────────────────────────────────────────────────
find_gguf() {
    find "$MODELS_DIR/$1" -name "*.gguf" -not -path "*/.no_exist/*" 2>/dev/null | head -1
}

file_size_gib() {
    local f="$1"
    python3 -c "import os; print(f'{os.path.getsize(\"$f\") / (1024**3):.1f}')" 2>/dev/null || echo "?"
}

fmt_delta() {
    local base="$1" new="$2"
    [[ "$base" == "—" || "$new" == "—" || -z "$base" || -z "$new" ]] && return
    python3 -c "
b, n = float('$base'), float('$new')
if b > 0:
    d = (n - b) / b * 100
    print(f'+{d:.1f}%' if d >= 0 else f'{d:.1f}%')
" 2>/dev/null
}

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# ── run_imp: Run imp-cli --bench and parse results ────────────────────────
# Sets: R_PP R_TG R_VRAM R_ERR
run_imp() {
    local gguf="$1" pp="$2" tg="$3"
    shift 3
    local extra_flags="$1"; shift
    local mode_flags=("$@")
    local out="$TMPDIR/run.out"

    R_PP="—" R_TG="—" R_VRAM="—" R_ERR=""

    if ! "$CLI" --model "$gguf" --bench --bench-pp "$pp" --max-tokens "$tg" \
         --bench-reps "$REPS" --seed 42 $extra_flags "${mode_flags[@]}" \
         >"$out" 2>&1; then
        R_ERR=$(grep -iE "error|failed|cannot|CUDA" "$out" | head -1)
        R_ERR="${R_ERR:-FAILED}"
        return 1
    fi

    R_PP=$(grep -oP  'pp\s+\d+.*?\(\s*\K[\d.]+(?=\s*tok/s\))'  "$out" | head -1)
    R_TG=$(grep -oP  'tg\s+\d+.*?\(\s*\K[\d.]+(?=\s*tok/s\))'  "$out" | head -1)
    R_VRAM=$(grep -oP 'GPU memory: \K\d+(?= MiB used)'           "$out" | head -1)

    R_PP="${R_PP:-—}"; R_TG="${R_TG:-—}"; R_VRAM="${R_VRAM:-?}"
    return 0
}

# ── run_llama: Run llama-bench and parse results ──────────────────────────
# Sets: LL_PP LL_TG
run_llama() {
    local gguf="$1" pp="$2" tg="$3"
    local out="$TMPDIR/llama.json"
    LL_PP="—" LL_TG="—"

    if ! "$LLAMA_BENCH" -m "$gguf" -p "$pp" -n "$tg" -ngl 99 -fa 1 \
         -r "$REPS" -o json > "$out" 2>/dev/null; then
        return 1
    fi

    LL_PP=$(python3 -c "
import json
data = json.load(open('$out'))
pp = [r for r in data if r.get('n_prompt',0) > 0 and r.get('n_gen',0) == 0]
print(f'{pp[0][\"avg_ts\"]:.2f}' if pp else '—')
" 2>/dev/null || echo "—")
    LL_TG=$(python3 -c "
import json
data = json.load(open('$out'))
tg = [r for r in data if r.get('n_gen',0) > 0 and r.get('n_prompt',0) == 0]
print(f'{tg[0][\"avg_ts\"]:.2f}' if tg else '—')
" 2>/dev/null || echo "—")
}

# ── JSON helpers ──────────────────────────────────────────────────────────
# Accumulate JSON results in a temp file; assemble at the end.
JSON_SCENARIOS="$TMPDIR/scenarios.json"
echo "[]" > "$JSON_SCENARIOS"

# Start a new scenario array entry
json_start_scenario() {
    local pp="$1" tg="$2"
    cat > "$TMPDIR/cur_scenario.json" <<PYEOF
{"pp": $pp, "tg": $tg, "reps": $REPS, "results": []}
PYEOF
}

json_add_result() {
    local name="$1" file_size="$2" gguf_path="$3"
    local imp_pp="$4" imp_tg="$5" imp_vram="$6"
    local ll_pp="$7" ll_tg="$8"
    local nvfp4_pp="$9" nvfp4_tg="${10}" nvfp4_vram="${11}"
    local fp8kv_pp="${12}" fp8kv_tg="${13}" fp8kv_vram="${14}"

    python3 - "$TMPDIR/cur_scenario.json" "$name" "$file_size" "$gguf_path" \
        "$imp_pp" "$imp_tg" "$imp_vram" "$ll_pp" "$ll_tg" \
        "$nvfp4_pp" "$nvfp4_tg" "$nvfp4_vram" \
        "$fp8kv_pp" "$fp8kv_tg" "$fp8kv_vram" << 'PYEOF'
import json, sys

scenario_file = sys.argv[1]
name, file_size, gguf_path = sys.argv[2], sys.argv[3], sys.argv[4]
imp_pp, imp_tg, imp_vram = sys.argv[5], sys.argv[6], sys.argv[7]
ll_pp, ll_tg = sys.argv[8], sys.argv[9]
nvfp4_pp, nvfp4_tg, nvfp4_vram = sys.argv[10], sys.argv[11], sys.argv[12]
fp8kv_pp, fp8kv_tg, fp8kv_vram = sys.argv[13], sys.argv[14], sys.argv[15]

def num_or_null(v):
    if v in ('\u2014', '?', '', 'ERR', 'SKIP'):
        return None
    try:
        return float(v)
    except:
        return None

def num_or_null_int(v):
    if v in ('\u2014', '?', '', 'ERR', 'SKIP'):
        return None
    try:
        return int(v)
    except:
        return None

entry = {
    'model': name,
    'file_size_gib': num_or_null(file_size),
    'gguf_path': gguf_path,
    'imp': {
        'pp_tok_s': num_or_null(imp_pp),
        'tg_tok_s': num_or_null(imp_tg),
        'vram_mib': num_or_null_int(imp_vram)
    },
    'llama': {
        'pp_tok_s': num_or_null(ll_pp),
        'tg_tok_s': num_or_null(ll_tg)
    } if ll_pp != 'SKIP' else None,
    'imp_nvfp4': {
        'pp_tok_s': num_or_null(nvfp4_pp),
        'tg_tok_s': num_or_null(nvfp4_tg),
        'vram_mib': num_or_null_int(nvfp4_vram)
    } if nvfp4_pp != 'SKIP' else None,
    'imp_fp8kv': {
        'pp_tok_s': num_or_null(fp8kv_pp),
        'tg_tok_s': num_or_null(fp8kv_tg),
        'vram_mib': num_or_null_int(fp8kv_vram)
    } if fp8kv_pp != 'SKIP' else None
}

with open(scenario_file) as f:
    scenario = json.load(f)
scenario['results'].append(entry)
with open(scenario_file, 'w') as f:
    json.dump(scenario, f)
PYEOF
}

json_end_scenario() {
    python3 - "$JSON_SCENARIOS" "$TMPDIR/cur_scenario.json" << 'PYEOF'
import json, sys
scenarios_file, cur_file = sys.argv[1], sys.argv[2]
with open(scenarios_file) as f:
    scenarios = json.load(f)
with open(cur_file) as f:
    scenario = json.load(f)
scenarios.append(scenario)
with open(scenarios_file, 'w') as f:
    json.dump(scenarios, f)
PYEOF
}

write_json() {
    local outfile="$1"
    python3 - "$JSON_SCENARIOS" "$outfile" "$TIMESTAMP" "$GPU_NAME" "$GPU_VRAM" \
        "$DRIVER_VER" "$CUDA_VER" "$OS_INFO" "$IMP_COMMIT" "$LLAMA_COMMIT" << 'PYEOF'
import json, sys

scenarios_file, outfile = sys.argv[1], sys.argv[2]
timestamp, gpu_name, gpu_vram = sys.argv[3], sys.argv[4], sys.argv[5]
driver_ver, cuda_ver, os_info = sys.argv[6], sys.argv[7], sys.argv[8]
imp_commit, llama_commit = sys.argv[9], sys.argv[10]

with open(scenarios_file) as f:
    scenarios = json.load(f)

result = {
    'timestamp': timestamp,
    'system': {
        'gpu': gpu_name,
        'vram_mib': int(gpu_vram) if gpu_vram != '?' else None,
        'driver': driver_ver,
        'cuda': cuda_ver,
        'os': os_info,
        'imp_commit': imp_commit,
        'llama_commit': llama_commit if llama_commit != '—' else None
    },
    'scenarios': scenarios
}

with open(outfile, 'w') as f:
    json.dump(result, f, indent=2)
PYEOF
}

write_markdown() {
    local json_file="$1" md_file="$2"
    python3 - "$json_file" "$md_file" << 'PYEOF'
import json, sys

json_file = sys.argv[1]
md_file = sys.argv[2]

with open(json_file) as f:
    data = json.load(f)

s = data['system']
lines = []
lines.append("# imp vs llama.cpp Benchmark Report\n")
lines.append(f"**Date:** {data['timestamp']}")
lines.append(f"**GPU:** {s['gpu']} ({s['vram_mib']} MiB)" if s['vram_mib'] else f"**GPU:** {s['gpu']}")
lines.append(f"**Driver:** {s['driver']} | **CUDA:** {s['cuda']}")
commit_line = f"**imp:** `{s['imp_commit']}`"
if s.get('llama_commit'):
    commit_line += f" | **llama.cpp:** `{s['llama_commit']}`"
lines.append(commit_line)
lines.append(f"**OS:** {s['os']}\n")

for sc in data['scenarios']:
    lines.append(f"## pp={sc['pp']}, tg={sc['tg']} ({sc['reps']} reps)\n")

    # Determine which columns to show
    has_llama = any(r.get('llama') for r in sc['results'])
    has_nvfp4 = any(r.get('imp_nvfp4') for r in sc['results'])
    has_fp8kv = any(r.get('imp_fp8kv') for r in sc['results'])

    # Main table
    hdr = "| Model | Size |"
    sep = "|---|---|"
    if has_llama:
        hdr += " llama pp | llama tg |"
        sep += "---|---|"
    hdr += " imp pp | imp tg |"
    sep += "---|---|"
    if has_llama:
        hdr += " Δpp | Δtg |"
        sep += "---|---|"
    hdr += " VRAM |"
    sep += "---|"
    lines.append(hdr)
    lines.append(sep)

    for r in sc['results']:
        imp = r.get('imp', {})
        ll = r.get('llama', {})

        def fmt(v):
            if v is None: return "—"
            if isinstance(v, float): return f"{v:.1f}" if v < 1000 else f"{v:.0f}"
            return str(v)

        def delta(base, new):
            if base is None or new is None or base == 0: return "—"
            d = (new - base) / base * 100
            return f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"

        row = f"| {r['model']} | {fmt(r.get('file_size_gib'))}G |"
        if has_llama:
            row += f" {fmt(ll.get('pp_tok_s'))} | {fmt(ll.get('tg_tok_s'))} |"
        row += f" {fmt(imp.get('pp_tok_s'))} | {fmt(imp.get('tg_tok_s'))} |"
        if has_llama:
            row += f" {delta(ll.get('pp_tok_s'), imp.get('pp_tok_s'))} | {delta(ll.get('tg_tok_s'), imp.get('tg_tok_s'))} |"
        vram = imp.get('vram_mib')
        row += f" {vram}M |" if vram else " — |"
        lines.append(row)

    lines.append("")

    # NVFP4 table
    if has_nvfp4:
        lines.append("### NVFP4 Decode Mode\n")
        lines.append("| Model | imp tg | nvfp4 tg | Δtg | nvfp4 VRAM |")
        lines.append("|---|---|---|---|---|")
        for r in sc['results']:
            nv = r.get('imp_nvfp4')
            imp = r.get('imp', {})
            if not nv: continue
            d = "—"
            if imp.get('tg_tok_s') and nv.get('tg_tok_s') and imp['tg_tok_s'] > 0:
                dd = (nv['tg_tok_s'] - imp['tg_tok_s']) / imp['tg_tok_s'] * 100
                d = f"+{dd:.1f}%" if dd >= 0 else f"{dd:.1f}%"
            vram = nv.get('vram_mib')
            lines.append(f"| {r['model']} | {fmt(imp.get('tg_tok_s'))} | {fmt(nv.get('tg_tok_s'))} | {d} | {vram}M |" if vram else f"| {r['model']} | {fmt(imp.get('tg_tok_s'))} | {fmt(nv.get('tg_tok_s'))} | {d} | — |")
        lines.append("")

    # FP8 KV table
    if has_fp8kv:
        lines.append("### FP8 KV Cache\n")
        lines.append("| Model | imp tg | fp8kv tg | Δtg | fp8kv VRAM |")
        lines.append("|---|---|---|---|---|")
        for r in sc['results']:
            fk = r.get('imp_fp8kv')
            imp = r.get('imp', {})
            if not fk: continue
            d = "—"
            if imp.get('tg_tok_s') and fk.get('tg_tok_s') and imp['tg_tok_s'] > 0:
                dd = (fk['tg_tok_s'] - imp['tg_tok_s']) / imp['tg_tok_s'] * 100
                d = f"+{dd:.1f}%" if dd >= 0 else f"{dd:.1f}%"
            vram = fk.get('vram_mib')
            lines.append(f"| {r['model']} | {fmt(imp.get('tg_tok_s'))} | {fmt(fk.get('tg_tok_s'))} | {d} | {vram}M |" if vram else f"| {r['model']} | {fmt(imp.get('tg_tok_s'))} | {fmt(fk.get('tg_tok_s'))} | {d} | — |")
        lines.append("")

with open(md_file, 'w') as f:
    f.write('\n'.join(lines) + '\n')
PYEOF
}

# ── Table formatting ──────────────────────────────────────────────────────
THICK_SEP="═══════════════════════════════════════════════════════════════════════════════════════════════════════"
THIN_SEP="───────────────────────────────────────────────────────────────────────────────────────────────────────"

print_table_header() {
    local pp="$1" tg="$2"

    echo ""
    printf "Scenario: pp=%s tg=%s reps=%s\n\n" "$pp" "$tg" "$REPS"

    if [[ $HAS_LLAMA -eq 1 ]]; then
        printf "%-30s %5s  %9s %9s  %9s %9s  %7s %7s  %6s\n" \
            "Model" "Size" "llama pp" "imp pp" "llama tg" "imp tg" "Δpp" "Δtg" "VRAM"
    else
        printf "%-30s %5s  %9s  %9s  %6s\n" \
            "Model" "Size" "pp tok/s" "tg tok/s" "VRAM"
    fi
    echo "$THIN_SEP"
}

print_table_row() {
    local name="$1" size="$2"
    local ll_pp="$3" ll_tg="$4" imp_pp="$5" imp_tg="$6" vram="$7"

    local vram_str="—"
    [[ "$vram" != "?" && "$vram" != "—" && -n "$vram" ]] && vram_str="${vram}M"

    if [[ $HAS_LLAMA -eq 1 ]]; then
        local dpp dtg
        dpp=$(fmt_delta "$ll_pp" "$imp_pp")
        dtg=$(fmt_delta "$ll_tg" "$imp_tg")
        dpp="${dpp:-—}"
        dtg="${dtg:-—}"
        printf "%-30s %5s  %9s %9s  %9s %9s  %7s %7s  %6s\n" \
            "$name" "${size}G" "$ll_pp" "$imp_pp" "$ll_tg" "$imp_tg" "$dpp" "$dtg" "$vram_str"
    else
        printf "%-30s %5s  %9s  %9s  %6s\n" \
            "$name" "${size}G" "$imp_pp" "$imp_tg" "$vram_str"
    fi
}

print_nvfp4_header() {
    echo ""
    echo "NVFP4 Decode Mode:"
    printf "%-30s  %9s %9s  %7s  %6s\n" "Model" "imp tg" "nvfp4 tg" "Δtg" "VRAM"
    echo "$THIN_SEP"
}

print_nvfp4_row() {
    local name="$1" imp_tg="$2" nv_tg="$3" nv_vram="$4"
    local dtg vram_str="—"
    dtg=$(fmt_delta "$imp_tg" "$nv_tg")
    dtg="${dtg:-—}"
    [[ "$nv_vram" != "?" && "$nv_vram" != "—" && -n "$nv_vram" ]] && vram_str="${nv_vram}M"
    printf "%-30s  %9s %9s  %7s  %6s\n" "$name" "$imp_tg" "$nv_tg" "$dtg" "$vram_str"
}

print_fp8kv_header() {
    echo ""
    echo "FP8 KV Cache:"
    printf "%-30s  %9s %9s  %7s  %6s\n" "Model" "imp tg" "fp8kv tg" "Δtg" "VRAM"
    echo "$THIN_SEP"
}

print_fp8kv_row() {
    local name="$1" imp_tg="$2" fk_tg="$3" fk_vram="$4"
    local dtg vram_str="—"
    dtg=$(fmt_delta "$imp_tg" "$fk_tg")
    dtg="${dtg:-—}"
    [[ "$fk_vram" != "?" && "$fk_vram" != "—" && -n "$fk_vram" ]] && vram_str="${fk_vram}M"
    printf "%-30s  %9s %9s  %7s  %6s\n" "$name" "$imp_tg" "$fk_tg" "$dtg" "$vram_str"
}

# ── Main ──────────────────────────────────────────────────────────────────
echo "$THICK_SEP"
echo "imp benchmark — $(date '+%Y-%m-%d %H:%M') — $GPU_NAME (${GPU_VRAM} MiB)"
printf "imp: %s" "$IMP_COMMIT"
[[ $HAS_LLAMA -eq 1 ]] && printf " | llama.cpp: %s" "$LLAMA_COMMIT"
printf " | CUDA %s | Driver %s\n" "$CUDA_VER" "$DRIVER_VER"
echo "$THICK_SEP"

if [[ $HAS_LLAMA -eq 0 && $NO_LLAMA -eq 0 ]]; then
    echo "llama-bench: not found (use LLAMA_BENCH= to set path, or --no-llama to skip)"
fi

# Create output dir
mkdir -p "$OUTPUT_DIR"

for scenario_name in $SCENARIO_LIST; do
    IFS=',' read -r sc_pp sc_tg <<< "${SCENARIOS[$scenario_name]}"

    # Apply overrides
    local_pp="${PP_OVERRIDE:-$sc_pp}"
    local_tg="${TG_OVERRIDE:-$sc_tg}"

    print_table_header "$local_pp" "$local_tg"
    json_start_scenario "$local_pp" "$local_tg"

    # Accumulators for optional mode tables (printed after main table)
    declare -a nvfp4_names=() nvfp4_imp_tgs=() nvfp4_tgs=() nvfp4_vrams=()
    declare -a fp8kv_names=() fp8kv_imp_tgs=() fp8kv_tgs=() fp8kv_vrams=()

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name dir_pattern extra_flags <<< "$entry"

        # Filter
        if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
            continue
        fi

        gguf=$(find_gguf "$dir_pattern")
        if [[ -z "$gguf" ]]; then
            printf "%-30s  SKIP (not found)\n" "$name"
            continue
        fi

        fsize=$(file_size_gib "$gguf")
        model_start=$(date +%s)

        # ── llama-bench ──
        cur_ll_pp="SKIP" cur_ll_tg="SKIP"
        if [[ $HAS_LLAMA -eq 1 ]]; then
            if run_llama "$gguf" "$local_pp" "$local_tg"; then
                cur_ll_pp="$LL_PP"
                cur_ll_tg="$LL_TG"
            else
                cur_ll_pp="ERR"
                cur_ll_tg="ERR"
            fi
        fi

        # ── imp baseline ──
        cur_imp_pp="—" cur_imp_tg="—" cur_imp_vram="?"
        # shellcheck disable=SC2086
        if run_imp "$gguf" "$local_pp" "$local_tg" "$extra_flags" --no-nvfp4; then
            cur_imp_pp="$R_PP"
            cur_imp_tg="$R_TG"
            cur_imp_vram="$R_VRAM"
        else
            cur_imp_pp="ERR"
            cur_imp_tg="ERR"
        fi

        # Print main table row
        print_table_row "$name" "$fsize" \
            "$cur_ll_pp" "$cur_ll_tg" "$cur_imp_pp" "$cur_imp_tg" "$cur_imp_vram"

        # ── NVFP4 (optional) ──
        cur_nv_pp="SKIP" cur_nv_tg="SKIP" cur_nv_vram="SKIP"
        if [[ $WITH_NVFP4 -eq 1 ]]; then
            # shellcheck disable=SC2086
            if run_imp "$gguf" "$local_pp" "$local_tg" "$extra_flags" --decode-nvfp4-only; then
                cur_nv_pp="$R_PP"
                cur_nv_tg="$R_TG"
                cur_nv_vram="$R_VRAM"
                nvfp4_names+=("$name")
                nvfp4_imp_tgs+=("$cur_imp_tg")
                nvfp4_tgs+=("$cur_nv_tg")
                nvfp4_vrams+=("$cur_nv_vram")
            else
                cur_nv_pp="ERR"
                cur_nv_tg="ERR"
                cur_nv_vram="?"
            fi
        fi

        # ── FP8 KV (optional) ──
        cur_fk_pp="SKIP" cur_fk_tg="SKIP" cur_fk_vram="SKIP"
        if [[ $WITH_FP8KV -eq 1 ]]; then
            # shellcheck disable=SC2086
            if run_imp "$gguf" "$local_pp" "$local_tg" "$extra_flags" --no-nvfp4 --kv-fp8; then
                cur_fk_pp="$R_PP"
                cur_fk_tg="$R_TG"
                cur_fk_vram="$R_VRAM"
                fp8kv_names+=("$name")
                fp8kv_imp_tgs+=("$cur_imp_tg")
                fp8kv_tgs+=("$cur_fk_tg")
                fp8kv_vrams+=("$cur_fk_vram")
            else
                cur_fk_pp="ERR"
                cur_fk_tg="ERR"
                cur_fk_vram="?"
            fi
        fi

        model_end=$(date +%s)
        elapsed=$((model_end - model_start))
        printf "  (%ds)\n" "$elapsed" >&2

        # Add to JSON
        json_add_result "$name" "$fsize" "$gguf" \
            "$cur_imp_pp" "$cur_imp_tg" "$cur_imp_vram" \
            "$cur_ll_pp" "$cur_ll_tg" \
            "$cur_nv_pp" "$cur_nv_tg" "$cur_nv_vram" \
            "$cur_fk_pp" "$cur_fk_tg" "$cur_fk_vram"
    done

    # Print optional mode tables
    if [[ ${#nvfp4_names[@]} -gt 0 ]]; then
        print_nvfp4_header
        for i in "${!nvfp4_names[@]}"; do
            print_nvfp4_row "${nvfp4_names[$i]}" "${nvfp4_imp_tgs[$i]}" \
                "${nvfp4_tgs[$i]}" "${nvfp4_vrams[$i]}"
        done
    fi

    if [[ ${#fp8kv_names[@]} -gt 0 ]]; then
        print_fp8kv_header
        for i in "${!fp8kv_names[@]}"; do
            print_fp8kv_row "${fp8kv_names[$i]}" "${fp8kv_imp_tgs[$i]}" \
                "${fp8kv_tgs[$i]}" "${fp8kv_vrams[$i]}"
        done
    fi

    json_end_scenario
    echo ""
done

# ── Write output files ────────────────────────────────────────────────────
JSON_FILE="$OUTPUT_DIR/${TIMESTAMP_SHORT}.json"
MD_FILE="$OUTPUT_DIR/${TIMESTAMP_SHORT}.md"

write_json "$JSON_FILE"
write_markdown "$JSON_FILE" "$MD_FILE"

echo "$THICK_SEP"
echo "Results saved:"
echo "  JSON:     $JSON_FILE"
echo "  Markdown: $MD_FILE"
echo "Done."
