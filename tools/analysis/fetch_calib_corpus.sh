#!/usr/bin/env bash
# Assemble the activation-calibration corpus for imp-quantize --calib.
#
# Deliberately NOT ppl_corpus_45k.txt. That file is imp's own architecture doc
# and it is what the quantizer is SCORED on; calibrating on it would tune the
# scales to the evaluation text and report a gain that does not exist off it.
# This corpus is general public-domain English prose instead, so a perplexity
# win on the technical corpus is a win the calibration generalised to.
#
# ~150k characters ≈ 35-40k tokens, which is the order AWQ-class methods use.
# Not checked in: it is a fetched artifact, not source.
#
# usage: tools/analysis/fetch_calib_corpus.sh [out-file]
set -euo pipefail

OUT="${1:-/tmp/imp_calib_corpus.txt}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Pride and Prejudice (Project Gutenberg #1342, public domain). The leading
# ~1500 lines are front matter and licence boilerplate — skipped so the
# statistics come from prose rather than from a licence header.
curl -sfL --max-time 120 -o "$TMP/pg.txt" \
    "https://www.gutenberg.org/files/1342/1342-0.txt"
# `head -c` closing the pipe early would SIGPIPE `tail` under `pipefail`, so
# the trim is done in two steps rather than one pipeline.
tail -n +120 "$TMP/pg.txt" > "$TMP/pg_body.txt"
head -c 120000 "$TMP/pg_body.txt" > "$TMP/a.txt"

# A second register (dialogue-heavy verse) so the corpus is not one author's
# sentence-length distribution.
curl -sfL --max-time 120 -o "$TMP/sh.txt" \
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
head -c 30000 "$TMP/sh.txt" > "$TMP/b.txt"

cat "$TMP/a.txt" "$TMP/b.txt" > "$OUT"
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
