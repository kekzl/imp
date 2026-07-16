#!/usr/bin/env bash
# #1007 stage-2 EXTERNAL gate: a real third-party coding agent (aider) driving
# imp-server over the OpenAI dialect through a real edit loop. Proves the whole
# loop survives an ACTUAL agent binary, not just our own driver.
#
# Boots imp-server (imp:test), runs aider (imp:agents) with OPENAI_API_BASE
# pointed at it on a throwaway git repo, asks for a one-function edit, and asserts
# the function landed in the file. Opt-in (heavier: builds a harness image + uses
# --network host) — the hard gate is `make test-agents` (agent_task_loop.py).
#
# Usage: tools/analysis/agent_external_smoke.sh [MODEL] [PORT]
set -euo pipefail

MODEL="${1:-Qwen3-8B-Q8_0.gguf}"
PORT="${2:-8080}"
IMG_SERVER="${DOCKER_IMG:-imp:test}"
IMG_AGENTS="imp:agents"
SRV_NAME="imp-agent-external"
WORK="$(mktemp -d)"

cleanup() {
    docker rm -f "$SRV_NAME" >/dev/null 2>&1 || true
    rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== #1007 external agent smoke: aider -> imp-server (model=$MODEL) ==="

# 1. Harness image (aider). Baked deps => the run itself is offline.
docker build -q -f tools/Dockerfile.agents -t "$IMG_AGENTS" tools/ >/dev/null
echo "harness image ready ($IMG_AGENTS)"

# 2. Boot imp-server.
docker rm -f "$SRV_NAME" >/dev/null 2>&1 || true
docker run -d --name "$SRV_NAME" --gpus all -p "${PORT}:8080" \
    -v "$HOME/models:/models" "$IMG_SERVER" \
    imp-server --host 0.0.0.0 --model "/models/$MODEL" >/dev/null
echo -n "waiting for server"
for _ in $(seq 1 90); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then echo " ok"; break; fi
    echo -n "."; sleep 2
done
curl -sf "http://localhost:${PORT}/health" >/dev/null || { echo "server did not come up"; exit 1; }

# 3. Throwaway repo + the edit task.
git -C "$WORK" init -q
git -C "$WORK" config user.email t@t.t
git -C "$WORK" config user.name t
printf '# math helpers\n' > "$WORK/math_utils.py"
git -C "$WORK" add -A
git -C "$WORK" commit -qm init

# 4. Real agent: aider makes its own edit decisions and applies the diff.
docker run --rm --network host -v "$WORK:/work" -w /work \
    -e OPENAI_API_BASE="http://localhost:${PORT}/v1" -e OPENAI_API_KEY=dummy \
    "$IMG_AGENTS" aider --model "openai/${MODEL}" \
    --yes-always --no-auto-commits --no-check-update --no-show-model-warnings \
    --no-gitignore --map-tokens 0 \
    --message "Add a function add(a, b) that returns a + b to math_utils.py" \
    math_utils.py 2>&1 | tail -15

# 5. Gate: the function must have landed in the file.
if grep -Eq 'def[[:space:]]+add[[:space:]]*\([[:space:]]*a[[:space:]]*,[[:space:]]*b' "$WORK/math_utils.py"; then
    echo "PASS: aider applied a real edit (add(a, b)) via imp-server"
    exit 0
fi
echo "FAIL: expected add(a, b) in math_utils.py, got:"; cat "$WORK/math_utils.py"
exit 1
