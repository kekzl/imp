#!/usr/bin/env bash
# Rebuilds tools/analysis/ppl_corpus_45k.txt (the PPL KPI corpus, 44994 bytes, 13 537 tokens)
# byte-identical from git history; reads blobs only, working tree state is irrelevant.
# Usage: tools/analysis/make_ppl_corpus.sh [out-file]
set -euo pipefail

OUT="${1:-tools/analysis/ppl_corpus_45k.txt}"
SRC=2e920fe328c2124492fb4ce9c71a5f1d2c0579a5
SHA256=cfbcc59b79dd4c8f7d258794569149985fcb778eed2cdf326d55de1304b83c83
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
TMP="$(mktemp)"
trap 'rm -f "$TMP" "$TMP.cat"' EXIT

# Five docs at $SRC, concatenated with no separator, first 45000 bytes.
for f in docs/architecture.md docs/sm120.md docs/BENCHMARKING.md README.md docs/GOAL.md; do
    git -C "$ROOT" show "$SRC:$f"
done > "$TMP.cat"
head -c 45000 "$TMP.cat" > "$TMP"

# #1922 (4599a6cd) path moves, applied after the cut: 2 x 3 bytes shorter, 45000 -> 44994.
sed -e 's#src/runtime/storage_planner\.cpp#src/exec/storage_planner.cpp#' \
    -e 's#src/runtime/process_diag\.h#src/core/process_diag.h#' "$TMP" > "$OUT"

if [ "$(sha256sum "$OUT" | cut -d' ' -f1)" != "$SHA256" ]; then
    echo "error: $OUT sha256 mismatch, expected $SHA256" >&2
    exit 1
fi
echo "wrote $OUT ($(wc -c < "$OUT") bytes, sha256 $SHA256)"
