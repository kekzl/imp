<!--
layer: L1
audience: operators
verified: 2026-08-31
commit: a81792d8
-->

# Deployment

Running imp as a service. Read [`LIMITATIONS.md`](LIMITATIONS.md) first:
several entries there are deployment decisions, in particular that one GPU
holds one model and that nothing is covered by an SLO.

## Compose

```yaml
services:
  imp:
    image: ghcr.io/kekzl/imp:latest
    command: ["--model", "/models/your-model.gguf", "--api-key", "${IMP_API_KEY}"]
    ports: ["127.0.0.1:8080:8080"]   # widen only together with --api-key
    volumes:
      - ./models:/models
      - imp-cache:/home/imp/.cache/imp
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
volumes:
  imp-cache:
```

The cache volume is not optional in a service context: without it every
restart re-transforms the weights. Must be writable by the container user; a
fresh volume owned by root silently disables the warm weight cache and the
library-reserve measurement.

## Configuration

Three layers, later wins: `imp.conf` → `--config <file>` → `--set key=value`.

```bash
--set runtime.max_seq_len=32768 --set kv_cache.dtype=fp8
```

**A `--set` key that does not exist is an error, not a warning.** Deliberate:
a typo used to measure the default silently. An unknown key in `imp.conf`
stays a warning, because a config file may outlive the build that understood
every key in it.

Every key, with defaults: `src/runtime/config.h` is the source of truth;
[`usage.md`](usage.md#configuration--impconf) is the readable version.

### From a container

The image entrypoint turns `IMP_*` environment variables into flags, so a
compose file needs no `command:`. Two of them reach the whole surface above:

| variable | becomes |
|---|---|
| `IMP_CONFIG` | `--config <path>` |
| `IMP_SET` | one `--set` per whitespace- or newline-separated `key=value` |

```yaml
environment:
  - IMP_MODEL=/models/your-model.gguf
  - IMP_SET=attention.sparse_topk_tokens=8192 kv_cache.growable=true
```

A value containing a space cannot be written in `IMP_SET`; pass that one in
`command:` instead.

The rest - `IMP_MODEL`, `IMP_HOST`, `IMP_PORT`, `IMP_API_KEY`,
`IMP_TRUSTED_PROXY`, `IMP_MODELS_DIR`, `IMP_MAX_TOKENS`, `IMP_GPU_LAYERS`,
`IMP_DEVICE`, `IMP_CHAT_TEMPLATE`, `IMP_MMPROJ`, `IMP_PREFILL_CHUNK_SIZE`,
`IMP_THINK_BUDGET`, `IMP_KV_FP8`, `IMP_KV_INT8`, `IMP_DECODE_NVFP4`,
`IMP_DECODE_NVFP4_ONLY`, `IMP_NO_NVFP4`, `IMP_NO_CUDA_GRAPHS`, `IMP_SSM_FP16`
- is one hand-written name per setting, and that list is frozen. Anything added
since arrives through `IMP_SET`.

**A pin can invert without being edited.** `IMP_KV_FP8=1` saves KV memory on
most families and costs it on one, because `auto` resolves NVFP4 where that has
been measured safe and FP8 is the wider type there. The engine now says so at
startup instead of leaving it to be discovered
([`plans/2026-08-29-qwen38-long-context-posture.md`](plans/2026-08-29-qwen38-long-context-posture.md)).
The legacy KV names also outrank a `kv_cache.dtype` from `IMP_SET` in either
order - `--kv-fp8` sets the dtype directly, ahead of the config file - so the
entrypoint warns when both are set.

The settings that most often need changing in a deployment:

| key | why you would touch it |
|---|---|
| `runtime.max_seq_len` | context ceiling. Auto-caps at 128K, VRAM- and model-bounded |
| `kv_cache.dtype` | `auto` takes the widest saving measured safe for the family - the model author's FP8 hint, or NVFP4. An explicit `fp8` is not always a saving: where the default is NVFP4, FP8 is the wider type |
| `runtime.max_batch_size` | 0 = auto from post-load headroom. Pin it for any A/B measurement |
| `server.model_swap` | default on: a request naming another model in the directory swaps to it |
| `moe.expert_cache_budget_pct` | only relevant when MoE experts do not fit; see [`PERF.md`](PERF.md) |

## Exit codes

The binaries return the C API's error taxonomy rather than a bare 1 (#1585):
a supervisor can tell a bad argument from a full GPU without parsing prose.

| code | meaning | retry? |
|---|---|---|
| 0 | success | |
| 1 | invalid argument, including a usage error | no, fix the call |
| 2 | out of memory (host) | maybe, with less concurrency |
| 3 | CUDA error | no |
| 4 | file not found | no |
| 5 | invalid model | no |
| 6 | unsupported | no |
| 7 | internal error | worth one retry |
| 8 | cancelled | n/a |
| 9 | capacity: the KV pool cannot fit this prompt | yes, shorter prompt or more VRAM |

Codes above 9 are unused. `imp-quantize` returned 2 for usage errors before
this; it returns 1 now, with every other invalid argument.

## Auth and exposure

```bash
--api-key "$IMP_API_KEY"      # bearer auth on the inference endpoints
--metrics-require-auth        # fold /metrics behind the same key
--trusted-proxy 10.0.0.5      # believe X-Forwarded-For from these peers only
```

**No default credential, no default refusal.** Without `--api-key` every
endpoint is open to whoever can reach the port: the shipped compose file
publishes on `127.0.0.1`, and widening it is a two-part change, `IMP_BIND`
plus `IMP_API_KEY` together (#1619).

**`--trusted-proxy` makes rate limiting work behind a proxy.** Without it
`X-Forwarded-For` is ignored and every request from the proxy shares one
bucket; with it the header is believed from those peers only. Believing it
from anyone else would let a client vary the header per request and bypass
the limit (#1614).

Per-request work is capped independently of the rate limit (one request can
ask for many units of it):

| flag | default | bounds |
|---|---|---|
| `--max-n` | 8 | `n` completions per chat request |
| `--max-batch-items` | 512 | rerank `documents`, embeddings `input` |
| `--max-logit-bias` | 1024 | `logit_bias` entries |
| `--http-read-timeout` | 60 s | socket read |
| `--http-write-timeout` | 600 s | socket write, must outlast a stream |
| `--http-keep-alive-max` | 100 | requests per connection |

**CORS is wide open by design** (`Access-Control-Allow-Origin: *` plus an
`OPTIONS` catch-all): the built-in web UI and browser clients call the API
directly. imp does not do TLS; if reachable beyond your own network,
terminate TLS and enforce origin policy at a reverse proxy.

A minimal nginx front:

```nginx
location / {
    proxy_pass         http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header   Connection "";
    proxy_buffering    off;        # required: SSE must not be buffered
    proxy_read_timeout 3600s;      # a long generation is one long response
}
```

`proxy_buffering off` is the line people forget: with buffering on, streaming
responses arrive in one lump at the end and every client reports TTFT equal
to total latency.

## Health, metrics, lifecycle

| endpoint | use |
|---|---|
| `GET /health` | liveness. Answers before a model is loaded |
| `GET /metrics` | Prometheus. Latency histograms (request, TTFT, inter-token, queue), decode batch size, refusal and cancellation counters, and a memory breakdown that separates capacity from occupancy |
| `GET /v1/models` | what is loaded, and what else is in the models directory |
| `POST /admin/suspend` | park the weights in host RAM and free the GPU completely. Inference answers 503 while suspended |
| `POST /admin/resume` | restore warm, in seconds, without re-reading weights |

Suspend/resume frees the GPU temporarily without paying a cold load
afterwards. Sessions and KV do not survive it; weights do.

### Which series answer which question

| question | series |
|---|---|
| how long did a request take | `imp_request_duration_seconds` (histogram) |
| how long until the first token | `imp_ttft_seconds`. Recorded on **both** transports since #1578; before that, streaming only, while `imp_requests_total` counted everything |
| how fast do tokens come out | `imp_inter_token_seconds`, **one observation per token** on a millisecond ladder (#1577). It used to be one per-request mean on the request-duration ladder, whose first bucket is 5 ms - at imp's own decode rates every observation landed in it, so `histogram_quantile` returned the bucket bounds rather than the data |
| is the server busy or slow | `imp_queue_time_seconds` is the admission wait, prefill excluded (#1580). A rising queue time with flat request duration is load; the reverse is the model |
| how much batching is happening | `rate(imp_decode_batch_rows_total[5m]) / rate(imp_decode_batch_steps_total[5m])`, and `imp_decode_batch_max` since start (#1580) |
| did the server break, or refuse | `imp_requests_failed_total` is 5xx. Every refusal this server is designed to make is a **4xx**, and those are `imp_requests_rejected_total` (#1579). Two series because they want different alerts |

`monitoring/grafana/dashboards/imp.json` plots the percentiles from those
histograms. The `stat` panels beside them show the last value only, which
cannot show a tail.

## Capacity planning

**Plan capacity, do not discover it.** Two platform properties make runtime
probing unreliable:

- A successful `cudaMalloc` proves nothing about free VRAM: WSL2/WDDM
  oversubscribes into host memory and returns success. Symptom is a 6.5x
  bandwidth cliff, not an error.
- Free VRAM only ever decreases within a process: the driver never returns a
  process's peak commitment, so anything sized from `cudaMemGetInfo` reads a
  moving floor.

Size the deployment from the model plus the KV pool you intend to serve; pin
`runtime.max_seq_len` rather than auto-fitting against a moving number. Tier
model and invariants: [`internals/MEMORY.md`](internals/MEMORY.md).

## Serving more than one model

One GPU holds one model. `server.model_swap` (default on): a request naming a
different model from the models directory swaps to it. In-flight generations
drain first, never cancelled; a failed load restores the previous model. The
requesting call pays the load, cheap on repeats via the warm weight cache.

Strict single-model semantics: `server.model_swap=false`.
