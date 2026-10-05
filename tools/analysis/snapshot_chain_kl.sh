#!/usr/bin/env bash
# #2419 chained-restore A/B: prompts P_1 ⊂ P_2 ⊂ ... P_N (corpus prefixes); every turn restores the previous
# turn's snapshot (int8 error compounds per turn in the int8 arm). Per turn: first-token top-20 logprobs.
# Usage: snapshot_chain_kl.sh <model> <turns> <chunk_chars>; OUT=<dir> keeps chain_<arm>.jsonl, MODELS_DIR
set -uo pipefail
MODEL="$1"; TURNS="${2:-8}"; CH="${3:-2500}"
S="${OUT:-$(mktemp -d)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CORPUS=$REPO/tools/analysis/ppl_corpus_45k.txt
for arm in false true; do
  bash "$REPO/scripts/require_free_gpu.sh" >/dev/null || { echo BUSY; exit 3; }
  docker rm -f snapch >/dev/null 2>&1
  docker run -d --name snapch --gpus all -v "$REPO:/src" -w /src -v "${MODELS_DIR:-$HOME/models}:/models:ro" \
    -p 127.0.0.1:8099:8080 -e IMP_DETERMINISTIC=1 imp:toolchain build-dev/imp-server --host 0.0.0.0 --port 8080 \
    --model "/models/$MODEL" --set server.recurrent_snapshot_int8=$arm --set speculative.mtp_k=0 >/dev/null
  for _ in $(seq 1 600); do curl -sf http://127.0.0.1:8099/health >/dev/null 2>&1 && break; sleep 1; done
  M=$(curl -s http://127.0.0.1:8099/v1/models | jq -r '.data[0].id')
  : > "$S/chain_$arm.jsonl"
  for t in $(seq 1 "$TURNS"); do
    P="$(head -c $((t * CH)) "$CORPUS")"
    curl -s -m 600 http://127.0.0.1:8099/v1/completions -H 'Content-Type: application/json' \
      -d "$(jq -n --arg m "$M" --arg p "$P" '{model:$m,prompt:$p,max_tokens:1,temperature:0,logprobs:20}')" |
      jq -c --argjson t "$t" '{t:$t, cached:.usage.prompt_tokens_details.cached_tokens, ptok:.usage.prompt_tokens,
                              top:.choices[0].logprobs.top_logprobs[0]}' >> "$S/chain_$arm.jsonl"
  done
  docker rm -f snapch >/dev/null 2>&1
  echo "$arm: $(jq -r '"t\(.t) \(.cached)/\(.ptok)"' "$S/chain_$arm.jsonl" | tr '\n' ' ')"
done
docker run --rm -i -v "$S:/s" python:3.12-slim python - <<'PY'
import json, math
a = [json.loads(l) for l in open("/s/chain_false.jsonl")]
b = [json.loads(l) for l in open("/s/chain_true.jsonl")]
for x, y in zip(a, b):
    pa, pb = x["top"] or {}, y["top"] or {}
    keys = set(pa) & set(pb)
    t1a = max(pa, key=pa.get) if pa else None
    t1b = max(pb, key=pb.get) if pb else None
    # KL(exact || int8) over the shared top-20 support, renormalised
    za = sum(math.exp(pa[k]) for k in keys); zb = sum(math.exp(pb[k]) for k in keys)
    kl = sum(math.exp(pa[k]) / za * ((pa[k] - math.log(za)) - (pb[k] - math.log(zb))) for k in keys) if keys else float("nan")
    dmax = max(abs(pa[k] - pb[k]) for k in keys) if keys else float("nan")
    print(f"turn {x['t']}: top1 {'same' if t1a == t1b else 'DIFF'} ({t1a!r} vs {t1b!r}) KL={kl:.2e} max|dlogp|={dmax:.3f} shared={len(keys)}")
PY
