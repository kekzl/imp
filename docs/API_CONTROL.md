<!--
layer: L1
audience: operators
verified: 2026-10-30
commit: 4666a109
-->

# API: controlling a running request

Endpoint list and shared error rules: [`API.md`](API.md).

## Ending a think block

`POST /v1/requests/{id}/end_thinking` (#2420): the next engine step forces the closer the think budget forces; the answer continues on the same KV.

| item | behaviour |
|---|---|
| `id` | chat `chatcmpl-...`, `/v1/messages` `msg_...` or its `request-id` header, `/v1/responses` `resp_...` (streaming), `/v1/completions` `cmpl-...`, or the client `X-Request-Id` on any dialect (the only id a non-streaming client holds). A reused `X-Request-Id` names the newest request |
| closer | `<think>` families: `"\n"` then `</think>`; gpt-oss Harmony: `<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>` |
| 200 | `{"id", "object": "request.end_thinking", "status"}`: `"ending"` (in a think block, or not admitted yet: applied once it is), `"already_closed"` (no open think block). Idempotent |
| 404 | `code: "request_not_found"`, `param: "id"`: unknown or finished id; always on a model-less server |
| 409 | `code: "no_reasoning_closer"`: in a think block of a model without a closer token id |
| latency | the next host step; a CUDA-graph burst already on the device finishes first |
| `/metrics` | `imp_think_end_requests_total`: requests flagged (first call each) |

## Steering with a control vector

`"control_vector": NAME` or `{"name": NAME, "scale": s}` (default scale 1) on chat requests adds `s * direction.l` to the hidden state after every layer `l` the file carries (roadmap row 114, `src/exec/executor_control_vector.cu`). Vectors load at startup: `--control-vector NAME=PATH`, llama.cpp `cvector-generator` GGUF (`direction.N`, F32 [d_model], N >= 1).

| item | behaviour |
|---|---|
| batching | one vector + scale is active engine-wide: a request asking for another waits for in-flight requests, then the worker switches (graphs re-capture), as with `lora` |
| prefix cache | keyed by vector id and scale: steered KV never serves an unsteered prompt |
| 400 | `code: "control_vector_not_loaded"` (unknown name), `"invalid_control_vector"` (not a name or `{name, scale}`, non-finite scale) |
| refused at load | width != d_model, `direction.0` or past the last layer, non-F32, gated-residual models (Qwen4Exp) |
| fused paths | while a vector is active, the batched add+norm and the NVFP4 norm-fold epilogues are off (they read the residual before the add) |

## Reading and clamping SAE features

`imp-server --sae DIR` loads one JumpReLU sparse autoencoder in the Gemma Scope 2 layout (`DIR/config.json` with `hf_hook_point_in: "model.layers.N.output"`, `DIR/params.safetensors` F32 `w_enc`, `w_dec`, `b_enc`, `threshold`, `b_dec`) on the residual after layer N (roadmap row 116, `src/exec/executor_sae.cu`).

| item | behaviour |
|---|---|
| `"sae_features": N` | 1..32: each chat logprobs content entry gains `"sae": {"l0", "err", "features": [{"id", "act"}]}`, the N strongest features of the residual that predicted the token; implies `logprobs` |
| `"sae_clamp": {"ID": value}` | 1..16 features: the residual after layer N becomes `x + (value - a_ID) w_dec[ID]` at every position, prompt included |
| `err` | `‖x - decode(encode(x))‖ / ‖x‖` of the unclamped residual: gemma-3-12b-it Q4_K_M with `resid_post/layer_24_width_16k_l0_medium`, 33 tokens: L0 58.5, err 0.055 (the same file declared at layer 10: L0 2193, err 99.4) |
| path | eager decode, no ragged prefill, no prefill graph; the batched add+norm and NVFP4 norm-fold epilogues are off while armed |
| prefix cache | keyed by the clamp list: clamped KV never serves an unclamped prompt |
| 400 | no `--sae` loaded, a clamp id >= the SAE width, `sae_features` outside 0..32 |
| model swap | drops the SAE; suspend/resume re-loads it |
