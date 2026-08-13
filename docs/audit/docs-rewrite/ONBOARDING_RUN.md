---
layer: L3
audience: agents
verified: 2026-08-13
commit: b61fbf5f
---

# Onboarding run

The dispatch's acceptance criterion: someone with no CUDA knowledge gets from
zero to an answered completion using the README alone. Executed 2026-08-13
against the published image, following the README verbatim and stopping at every
point where a reader would have to guess.

**Honest scoping.** The operator here is not CUDA-naive, so this is not a user
study. What it *does* test, and what the criterion is actually about, is whether
the README's instructions are complete and correct when followed literally: no
missing step, no command that fails, no claim that does not hold. Every command
below was run as written, and the outputs are pasted, not summarised.

## Step 1, requirements

README says an `sm_120a` GPU, a driver, the container toolkit. Checked:

```
$ nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
0
$ docker ps -q | wc -l
0
```

The README tells the reader to check `docker ps` as well as `nvidia-smi`, which
is the non-obvious half on WSL2. **Complete.**

## Step 2, start the server

Ran the README block verbatim, changing only the host port (8099) to avoid the
user's own service on 8080, and the model path to a checkpoint that exists here:

```
docker run -d --gpus all -v /home/kekz/models:/models \
  -v imp-onboarding-cache:/home/imp/.cache/imp \
  -p 8099:8080 ghcr.io/kekzl/imp:latest --model /models/Qwen3-8B-Q8_0.gguf
```

Result: container healthy. **Worked as written.**

## Step 3, the two calls

```
$ curl -s http://localhost:8099/v1/models | jq -r '.data[].id'
Qwen3-8B-Q8_0.gguf

$ curl -s http://localhost:8099/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-8B-Q8_0.gguf","messages":[{"role":"user","content":"Name the capital of France."}],"max_tokens":64}'
'The capital of France is **Paris**. 🇫🇷'
```

**Zero to an answered completion: reached.** The README's warning that `model` is
required and equals the basename is load-bearing; without it the first call
fails and the error would not obviously point at the fix.

## Step 4, the other claims the README and QUICKSTART make

| claim | result |
|---|---|
| built-in chat UI at `GET /` | HTTP 200, 23 367 bytes. **Holds** |
| `/v1/messages` answers natively | `{"content":[{"type":"thinking",...},{"type":"text","text":"The capital of France is Paris."}], "stop_reason":"end_turn"}`. **Holds** |
| per-token SSE on `/v1/messages`, not synthetic | **28 `content_block_delta` events** for a 24-token reply. **Holds, and this is now measured rather than read off the code** |

## The one thing the docs did not say, and now do

The first `/v1/messages` probe read `content[0].text` and got an empty string,
which looks exactly like a broken endpoint. It is not: a reasoning model returns
a `thinking` block first and the answer second, so `content[0]` has no `text`
field at all.

That is a client-visible trap with no error attached to it, and it was missing
from the documentation. Added to [`API.md`](../../API.md) with the selector a
client should use. **This is the run paying for itself**: the defect was in the
docs, and only running the documented calls surfaced it.

## Verdict

The README path is complete and correct. One documentation gap found and closed.
