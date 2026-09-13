#!/usr/bin/env bash
# Assembles the activation-calibration corpus for imp-quantize --calib. Deliberately NOT
# ppl_corpus_45k.txt (imp's own arch doc, what the quantizer is SCORED on): calibrating and
# scoring on one text would report a gain that doesn't generalise.
# ~150k chars (~35-40k tokens), the order AWQ-class methods use. Not checked in (fetched
# artifact, not source). Usage: tools/analysis/fetch_calib_corpus.sh [out-file].
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
