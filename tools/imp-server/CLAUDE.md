---
layer: L3
audience: agents
verified: 2026-08-13
commit: 81ffa573
---

# tools/imp-server — the HTTP surface

OpenAI, Anthropic and OpenAI-Responses dialects over one engine. Thin wire-format
adapters on a shared core; the engine lives in `src/runtime/`.

## Invariants

- **One shared per-token streaming driver** (`handlers_chat_stream.cpp`). The
  three dialects are wire-format adapters over it. Do not grow a second token
  loop: they were hand-copied once and drifted twice (#892, #941).
- **Every error is a JSON envelope**, never a bare status with an empty body.
  `/v1/messages*` gets the Anthropic error shape.
- **A request the server cannot honour is a 4xx, not a best-effort answer.** An
  uncompilable constraint, an unreadable image, a `tool_choice` naming an absent
  function: all 400. This is a deliberate, breaking-by-design stance.
- **Reasoning goes to `reasoning_content`**, never into `content`. This was once
  wrong on the streaming path only, which our own batteries could not see.

## Entry points

- `main.cpp` — route registration, CORS, auth
- `handlers_chat.cpp` / `handlers_chat_stream.cpp` — OpenAI + the shared driver
- `handlers_messages.cpp` — Anthropic wire format
- `handlers_responses.cpp` — `/v1/responses`
- `handlers_chat_params.cpp` — parameter parsing and validation
- `webui/index.html` — embedded at build time by `cmake/embed_webui.cmake`

## Build & test

```
make dev-test            # CPU lane, includes the mock API battery
make test-server         # boots a real imp-server (needs a GPU)
make test-agents-external # real aider / Claude Code / OpenAI Agents SDK
```

The `Real API contract` CI job runs the API tests against the built binary
without a GPU. Before it existed, all 82 assertions described the mock.

## Pitfalls

- **Testing against the mock proves nothing about the server.** The two
  disagreed on `n=2` for months.
- A client that streams needs `proxy_buffering off` at any reverse proxy, or
  every TTFT measurement equals total latency.
- CORS is wide open on purpose (the built-in UI calls the API directly). Do not
  "fix" it; document the proxy instead.

## Do not touch

`webui/index.html` is deliberately one file with no build step and no
dependencies. Keep it that way.

## See also

[`docs/API.md`](../../docs/API.md) for the field-by-field surface,
[`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md) for auth and proxying. Skill
`server-api` carries the playbook.
