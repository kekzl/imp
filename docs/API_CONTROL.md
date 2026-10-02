<!--
layer: L1
audience: operators
verified: 2026-10-02
commit: db9162d5
-->

# API: controlling a running request

Endpoint list and shared error rules: [`API.md`](API.md).

## Ending a think block

`POST /v1/requests/{id}/end_thinking` (#2420, llama.cpp #23971): the next engine step forces the closer the think budget forces; the answer continues on the same KV.

| item | behaviour |
|---|---|
| `id` | chat `chatcmpl-...`, `/v1/messages` `msg_...` or its `request-id` header, `/v1/responses` `resp_...` (streaming), `/v1/completions` `cmpl-...`, or the client `X-Request-Id` on any dialect (the only id a non-streaming client holds). A reused `X-Request-Id` names the newest request |
| closer | `<think>` families: `"\n"` then `</think>`; gpt-oss Harmony: `<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>` |
| 200 | `{"id", "object": "request.end_thinking", "status"}`: `"ending"` (in a think block, or not admitted yet: applied once it is), `"already_closed"` (no open think block). Idempotent |
| 404 | `code: "request_not_found"`, `param: "id"`: unknown or finished id; always on a model-less server |
| 409 | `code: "no_reasoning_closer"`: in a think block of a model without a closer token id |
| latency | the next host step; a CUDA-graph burst already on the device finishes first |
| `/metrics` | `imp_think_end_requests_total`: requests flagged (first call each) |
