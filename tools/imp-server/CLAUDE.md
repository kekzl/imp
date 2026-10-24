<!--
layer: L3
audience: agents
verified: 2026-10-24
commit: 28185525
-->

# tools/imp-server - the HTTP surface

OpenAI, Anthropic, OpenAI-Responses: thin wire-format adapters over one engine (`src/runtime/`).

## Invariants

- One shared per-token streaming driver (`handlers_chat_stream.cpp`), no second token loop (#892, #941).
- Every error is a JSON envelope; `/v1/messages*` uses the Anthropic shape.
- Unservable request = 4xx: uncompilable constraint, unreadable image, `tool_choice` naming an absent function = 400.
- Remote hosts only with `--allow-remote-images` (`image_url`): classify before connect, no redirects, destination-independent errors (#1610, `image_fetch.h`).
- Reasoning goes to `reasoning_content`, never `content`, streaming and non-streaming.

## Entry points

- `main.cpp`: routes, CORS, auth
- `handlers_chat.cpp` / `handlers_chat_stream.cpp`: OpenAI + shared driver
- `handlers_completions.cpp` / `fim_request.cpp`: `/v1/completions`, `POST /infill` (prompt assembly `src/model/fim.h`)
- `handlers_messages.cpp`: Anthropic
- `handlers_responses.cpp`: `/v1/responses`
- `handlers_chat_params.cpp`: parameter parsing, validation
- `sampling_fields.h`: the one sampling-field parser + dialect pass-through table; new field = one row (#2461)
- `webui/index.html`: embedded by `cmake/embed_webui.cmake`

## Test

```
make dev-test             # CPU lane, includes the mock API battery
make test-server          # real imp-server (GPU)
make test-agents-external # real aider / Claude Code / OpenAI Agents SDK
```

- CI job `Real API contract`: API tests against the built binary, no GPU.
- Web UI without GPU: `webui/dev/mock_server.py` + `webui/dev/drive.py` (image `mcr.microsoft.com/playwright/python:v1.55.0-noble`, `markup_test.py` for the renderer); commands in file headers.
- Docs screenshots `docs/webui*.png`: `webui/dev/live_shots.py` against a running server.
- `tests/api/mock_server.py` sends SSE in one write: no streaming coverage.

## Pitfalls

- A mock test proves nothing about the server.
- CORS off unless `--cors-origins` lists the Origin (#2402). Policy: `cors_allow_origin` in `cors.h`, mirrored in `tests/api/mock_server.py`.

## Do not touch

`webui/index.html` stays one file: no build step, no dependencies.

## See also

[`docs/API.md`](../../docs/API.md), [`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md). Skill `server-api`.
