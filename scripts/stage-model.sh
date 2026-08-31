#!/bin/bash
# Stage a HuggingFace checkpoint and quantize it to NVFP4, in one command.
#
# The gap this closes: imp reads NVFP4 SafeTensors and refuses to fetch anything
# itself (clean-host policy, src/model/hf_hub.h), so a newcomer with a 5090 and a
# model in mind has to work out the download, the conversion, the container
# invocation and the disk arithmetic before seeing a single token. For a model
# with no usable published export that is a long way from "try it".
#
#   scripts/stage-model.sh Qwen/Qwen3.8-27B-FP8 ~/models/my-Qwen3.8-NVFP4
#
# curl and jq only, no Python and no huggingface-cli: this runs on the host,
# which by policy has neither.
set -uo pipefail

REPO="${1:-}"
OUT="${2:-}"
IMAGE="${IMP_IMAGE:-imp:test}"

if [ -z "$REPO" ] || [ -z "$OUT" ]; then
    cat <<'USAGE'
usage: scripts/stage-model.sh <hf-repo-id> <output-dir> [--keep-source]

  <hf-repo-id>   e.g. Qwen/Qwen3.8-27B-FP8
  <output-dir>   where the NVFP4 checkpoint is written

  --keep-source  keep the downloaded source after quantizing (default: kept,
                 pass --drop-source to delete it once the output exists)

Environment:
  IMP_IMAGE      container image to quantize with (default imp:test)
  STAGE_DIR      where the source is downloaded (default <output-dir>.src)

Prefer an FP8 or BF16 source. A published NVFP4 export needs no staging: point
imp at it directly.
USAGE
    exit 2
fi

DROP_SOURCE=0
for a in "$@"; do [ "$a" = "--drop-source" ] && DROP_SOURCE=1; done
SRC="${STAGE_DIR:-${OUT}.src}"

for tool in curl jq docker; do
    command -v "$tool" >/dev/null || { echo "need $tool on PATH"; exit 1; }
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "image $IMAGE not found. Build it with 'make build', or set IMP_IMAGE."
    exit 1
}

API="https://huggingface.co/api/models/${REPO}"
echo "== ${REPO}"
LISTING=$(curl -sf "$API") || { echo "cannot reach $API (private or gated repo?)"; exit 1; }

# Size the download before starting it: a 27B BF16 source is ~52 GiB, and
# running out of disk halfway through leaves a directory that looks complete.
NEED=$(echo "$LISTING" | jq '[.siblings[].rfilename] | length')
BYTES=$(curl -sf "https://huggingface.co/api/models/${REPO}/tree/main" \
        | jq '[.[] | (.size // .lfs.size // 0)] | add // 0')
HAVE=$(df -PB1 "$(dirname "$OUT")" | awk 'NR==2{print $4}')
printf "   %s files, %.1f GiB to download, %.1f GiB free\n" \
       "$NEED" "$(echo "$BYTES" | awk '{print $1/1073741824}')" \
       "$(echo "$HAVE" | awk '{print $1/1073741824}')"
# The output lands beside the source, so both have to fit at once.
if [ "$BYTES" -gt 0 ] && [ "$HAVE" -lt "$BYTES" ]; then
    echo "   not enough free disk for the source alone. Aborting."
    exit 1
fi

mkdir -p "$SRC" || exit 1
echo "== downloading to $SRC"
for f in $(echo "$LISTING" | jq -r '.siblings[].rfilename'); do
    case "$f" in */*) mkdir -p "$SRC/$(dirname "$f")";; esac
    # -C - resumes, so an interrupted run continues instead of restarting.
    curl -sfL -C - -o "$SRC/$f" "https://huggingface.co/${REPO}/resolve/main/${f}" \
        || { echo "   FAILED $f"; exit 1; }
    printf '.'
done
echo " done"

# A repo that is ALREADY imp-readable NVFP4 needs no conversion, only the
# download — which is the part that is awkward without git-lfs (not installed
# here by policy, so `git clone` fetches pointer files instead of weights) and
# without huggingface-cli (Python, likewise absent). Detect it and stop here
# rather than spend 25 minutes proving there is nothing to quantize.
if [ -f "$SRC/hf_quant_config.json" ] && grep -q '"quant_algo"[[:space:]]*:[[:space:]]*"NVFP4"' \
        "$SRC/hf_quant_config.json" 2>/dev/null; then
    echo "== already NVFP4, no conversion needed"
    if [ "$SRC" != "$OUT" ]; then
        mkdir -p "$(dirname "$OUT")" && rm -rf "$OUT" && mv "$SRC" "$OUT"
    fi
    cat <<EOF

Ready. Serve it with:

  docker run --rm --gpus all -p 8080:8080 -v $(dirname "$OUT"):/models $IMAGE \\
      imp-server --host 0.0.0.0 --model /models/$(basename "$OUT")
EOF
    exit 0
fi

# Forecast before converting: this prints the output size and whether it fits
# the card, in seconds, rather than after the ~25 minute write.
echo "== forecast"
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$(cd "$SRC" && pwd)":/src:ro "$IMAGE" \
    imp-quantize --model /src --out /tmp/forecast --dry-run 2>&1 \
    | grep -E "^(size:|card:)|^ +this checkpoint|^ +workspaces|kept at source" || true

echo "== quantizing to $OUT"
mkdir -p "$OUT" || exit 1
# --user keeps the output owned by the caller: a container writing as root
# leaves a checkpoint the user cannot delete without sudo.
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$(cd "$SRC" && pwd)":/src:ro -v "$(cd "$OUT" && pwd)":/out "$IMAGE" \
    imp-quantize --model /src --out /out || { echo "quantization failed"; exit 1; }

if [ "$DROP_SOURCE" = "1" ]; then
    echo "== removing source $SRC"
    rm -rf "$SRC"
fi

# The container serves as its own uid (1001), not as the caller. A checkpoint
# under a 0700 directory therefore quantizes fine and then fails to LOAD with a
# bare "filesystem error: status: Permission denied", which says nothing about
# the cause. Check it here, where the fix is one chmod, rather than leaving it
# to be discovered as a crash.
if ! docker run --rm -v "$(cd "$(dirname "$OUT")" && pwd)":/probe:ro "$IMAGE" \
        test -r "/probe/$(basename "$OUT")/config.json" 2>/dev/null; then
    echo
    echo "WARNING: the container cannot read $OUT."
    echo "  It runs as uid $(docker run --rm "$IMAGE" id -u 2>/dev/null), and some directory on the path is not"
    echo "  world-readable. imp would fail at load with a bare 'Permission denied'."
    echo "  Fix with:  chmod o+rX $(dirname "$OUT") $OUT"
fi

cat <<EOF

Ready. Serve it with:

  docker run --rm --gpus all -p 8080:8080 -v $(dirname "$OUT"):/models $IMAGE \\
      imp-server --host 0.0.0.0 --model /models/$(basename "$OUT")

or run a single prompt:

  docker run --rm --gpus all -v $(dirname "$OUT"):/models $IMAGE \\
      imp-cli --model /models/$(basename "$OUT") --prompt "Hello" --max-tokens 64
EOF
