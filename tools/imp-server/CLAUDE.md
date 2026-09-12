<!--
layer: L3
audience: agents
verified: 2026-09-13
commit: 81d22eff
-->

# tools/imp-server - the HTTP surface

OpenAI, Anthropic and OpenAI-Responses dialects: thin wire-format adapters over one engine (`src/runtime/`).

## Invariants

- One shared per-token streaming driver (`handlers_chat_stream.cpp`); the three dialects adapt it. No second token loop (#892, #941).
- Every error is a JSON envelope, never a bare status; `/v1/messages*` gets the Anthropic error shape.
- A request the server cannot honour is a 4xx, by design: uncompilable constraint, unreadable image, `tool_choice` naming an absent function are all 400.
- The server connects to a host a request names only with `--allow-remote-images` (`image_url`): destination classified before the connect, no redirects, error strings independent of the destination (#1610, `image_fetch.h`).
- Reasoning goes to `reasoning_content`, never into `content`, on streaming and non-streaming paths.

## Entry points

- `main.cpp`: route registration, CORS, auth
- `handlers_chat.cpp` / `handlers_chat_stream.cpp`: OpenAI + the shared driver
- `handlers_messages.cpp`: Anthropic wire format
- `handlers_responses.cpp`: `/v1/responses`
- `handlers_chat_params.cpp`: parameter parsing and validation
- `webui/index.html`: embedded at build time by `cmake/embed_webui.cmake`

## Test

```
make dev-test             # CPU lane, includes the mock API battery
make test-server          # boots a real imp-server (GPU)
make test-agents-external # real aider / Claude Code / OpenAI Agents SDK
```

- CI job `Real API contract` runs the API tests against the built binary, no GPU.
- Web UI without a GPU: `webui/dev/mock_server.py` (imp-shaped SSE per token) + `webui/dev/drive.js` (Playwright image); commands in the file headers.
- `tests/api/mock_server.py` sends the SSE body in one write: it does not exercise streaming.

## Pitfalls

- A test against the mock proves nothing about the server.
- A streaming client needs `proxy_buffering off` at any reverse proxy, or TTFT equals total latency.
- CORS is wide open on purpose (the built-in UI calls the API directly): document the proxy, do not "fix" it.

## Do not touch

`webui/index.html` stays one file: no build step, no dependencies.

## See also

[`docs/API.md`](../../docs/API.md) (field-by-field surface), [`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md) (auth, proxying). Skill `server-api`.
