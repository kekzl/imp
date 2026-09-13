#!/usr/bin/env bash
# #1007 stage-2 EXTERNAL gate: real third-party agent binaries driving imp-server through a
# genuine edit loop, proving the loop survives an ACTUAL agent rather than our own probes.
# Three legs: aider -> /v1/chat/completions (imp:agents), Claude Code -> /v1/messages
# (imp:claude-code), OpenAI Agents SDK -> /v1/responses (imp:agents-sdk). Each asks for a
# one-function edit on a throwaway repo and asserts it landed. Claude Code leg also asserts no
# raw chain-of-thought reaches the user-visible channel.
# Opt-in (builds harness images, uses --network host); hard gate is make test-agents.
# Usage: tools/analysis/agent_external_smoke.sh [MODEL] [PORT] [LEG=all|aider|claude-code|agents-sdk].
set -euo pipefail

MODEL="${1:-Qwen3-8B-Q8_0.gguf}"
PORT="${2:-8080}"
LEG="${3:-all}"
IMG_SERVER="${DOCKER_IMG:-imp:test}"
IMG_AGENTS="imp:agents"
IMG_CC="imp:claude-code"
IMG_SDK="imp:agents-sdk"
SRV_NAME="imp-agent-external"
WORK="$(mktemp -d)"
FAILED=0

cleanup() {
    docker rm -f "$SRV_NAME" >/dev/null 2>&1 || true
    # Agents run as root in their containers; anything they left in the mount is
    # not ours to delete, and a failed cleanup must not mask the gate's verdict.
    rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== #1007 external agent smoke: real agent binaries -> imp-server (model=$MODEL) ==="

# Boot imp-server once; both legs share it.
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

# A throwaway repo per leg — an agent that edits nothing must not pass because
# the other leg already wrote the function.
new_repo() {
    local dir="$WORK/$1"
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" config user.email t@t.t
    git -C "$dir" config user.name t
    printf '# math helpers\n' > "$dir/math_utils.py"
    git -C "$dir" add -A
    git -C "$dir" commit -qm init
    echo "$dir"
}

edit_landed() {
    grep -Eq 'def[[:space:]]+add[[:space:]]*\([[:space:]]*a[[:space:]]*,[[:space:]]*b' "$1/math_utils.py"
}

# ---------------------------------------------------------------------------
# Leg 1 — aider over the OpenAI dialect.
# ---------------------------------------------------------------------------
if [ "$LEG" = "all" ] || [ "$LEG" = "aider" ]; then
    echo "--- leg: aider -> /v1/chat/completions ---"
    docker build -q -f tools/Dockerfile.agents -t "$IMG_AGENTS" tools/ >/dev/null
    REPO="$(new_repo aider)"
    docker run --rm --network host -v "$REPO:/work" -w /work \
        -e OPENAI_API_BASE="http://localhost:${PORT}/v1" -e OPENAI_API_KEY=dummy \
        "$IMG_AGENTS" aider --model "openai/${MODEL}" \
        --yes-always --no-auto-commits --no-check-update --no-show-model-warnings \
        --no-gitignore --map-tokens 0 \
        --message "Add a function add(a, b) that returns a + b to math_utils.py" \
        math_utils.py 2>&1 | tail -12
    if edit_landed "$REPO"; then
        echo "PASS: aider applied a real edit (add(a, b)) via imp-server"
    else
        echo "FAIL: aider — expected add(a, b) in math_utils.py, got:"; cat "$REPO/math_utils.py"
        FAILED=1
    fi
fi

# Leg 2, Claude Code over the Anthropic dialect: the demanding client (~20K system prompt, 25
# tool definitions, cache_control, extended-thinking fields, streaming, all in one request).
if [ "$LEG" = "all" ] || [ "$LEG" = "claude-code" ]; then
    echo "--- leg: Claude Code -> /v1/messages ---"
    docker build -q -f tools/Dockerfile.claude-code -t "$IMG_CC" tools/ >/dev/null
    REPO="$(new_repo claude-code)"
    OUT="$WORK/cc_out.txt"
    docker run --rm --network host -v "$REPO:/work" -w /work \
        -e ANTHROPIC_BASE_URL="http://localhost:${PORT}" \
        -e ANTHROPIC_AUTH_TOKEN=dummy \
        -e ANTHROPIC_MODEL="$MODEL" -e ANTHROPIC_SMALL_FAST_MODEL="$MODEL" \
        -e DISABLE_TELEMETRY=1 -e DISABLE_AUTOUPDATER=1 -e DISABLE_ERROR_REPORTING=1 \
        -e HOME=/tmp/cchome \
        "$IMG_CC" claude -p \
        "Use the Edit tool to add a function add(a, b) returning a + b to math_utils.py" \
        --permission-mode acceptEdits --allowedTools "Edit,Write,Read" >"$OUT" 2>&1 || true
    tail -6 "$OUT"

    if edit_landed "$REPO"; then
        echo "PASS: Claude Code applied a real edit (add(a, b)) via /v1/messages"
    else
        echo "FAIL: Claude Code — expected add(a, b) in math_utils.py, got:"
        cat "$REPO/math_utils.py"
        FAILED=1
    fi

    # Reasoning must stay in the thinking channel. A leak is otherwise silent:
    # the loop still "works", the user just reads the chain of thought.
    if grep -q '</think>\|<think>' "$OUT"; then
        echo "FAIL: Claude Code — think markers reached the visible channel"
        FAILED=1
    else
        echo "PASS: no think markers in the user-visible output"
    fi
fi

# Leg 3, OpenAI Agents SDK over /v1/responses (what the Agents SDK and Codex speak; aider
# covers chat-completions, Claude Code covers /v1/messages).
# Driver pins temperature=0, max_tokens=400: on Qwen3-8B-Q8_0 a 400-token budget yields
# reasoning+function_call, while 1400 yields a bare message (model reasons past the call given
# room). imp emits both shapes correctly; pinning the budget tests the DIALECT, not the model's
# appetite for deliberation.
if [ "$LEG" = "all" ] || [ "$LEG" = "agents-sdk" ]; then
    echo "--- leg: OpenAI Agents SDK -> /v1/responses ---"
    docker build -q -f tools/Dockerfile.agents-sdk -t "$IMG_SDK" tools/ >/dev/null
    REPO="$(new_repo agents-sdk)"
    OUT="$WORK/sdk_out.txt"
    docker run --rm --network host -v "$REPO:/work" \
        -v "$(pwd)/tools/analysis:/drv:ro" -w /work \
        -e OPENAI_BASE_URL="http://localhost:${PORT}/v1" \
        -e OPENAI_API_KEY=dummy \
        -e IMP_MODEL="$MODEL" \
        "$IMG_SDK" python /drv/agents_sdk_edit.py >"$OUT" 2>&1 || true
    tail -6 "$OUT"

    if edit_landed "$REPO"; then
        echo "PASS: the Agents SDK applied a real edit (add(a, b)) via /v1/responses"
    else
        echo "FAIL: Agents SDK — expected add(a, b) in math_utils.py, got:"
        cat "$REPO/math_utils.py" 2>/dev/null || echo "(no file)"
        FAILED=1
    fi

    # The run has to go through an actual tool call, not a description of one.
    if grep -q "ToolCallItem" "$OUT"; then
        echo "PASS: the run contains a real function_call item"
    else
        echo "FAIL: Agents SDK — no function_call in the run; the model only talked"
        FAILED=1
    fi
fi

exit "$FAILED"
