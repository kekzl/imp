<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# Deployment

Running imp as a service rather than from a terminal. Read
[`LIMITATIONS.md`](LIMITATIONS.md) first if you have not: several entries there
are deployment decisions in disguise, in particular that one GPU holds one model
and that nothing here is covered by an SLO.

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

The cache volume is not optional in a service context: without it every restart
re-transforms the weights. It must be writable by the container user; a fresh
volume owned by root silently disables both the warm weight cache and the
library-reserve measurement.

## Configuration

Three layers, later wins: `imp.conf` → `--config <file>` → `--set key=value`.

```bash
--set runtime.max_seq_len=32768 --set kv_cache.dtype=fp8
```

**A `--set` key that does not exist is an error, not a warning.** That is
deliberate: a typo used to measure the default silently. An unknown key in
`imp.conf` stays a warning, because a config file may outlive the build that
understood every key in it.

Every key, with defaults: `src/runtime/config.h` is the source of truth;
[`usage.md`](usage.md#configuration--impconf) is the readable version.

The settings that most often need changing in a deployment:

| key | why you would touch it |
|---|---|
| `runtime.max_seq_len` | context ceiling. Auto-caps at 128K, VRAM- and model-bounded |
| `kv_cache.dtype` | `auto` honours the model's own FP8 hint where the quality gate passed. `fp8` halves KV memory |
| `runtime.max_batch_size` | 0 = auto from post-load headroom. Pin it for any A/B measurement |
| `server.model_swap` | default on: a request naming another model in the directory swaps to it |
| `moe.expert_cache_budget_pct` | only relevant when MoE experts do not fit; see [`PERF.md`](PERF.md) |

## Auth and exposure

```bash
--api-key "$IMP_API_KEY"      # bearer auth on the inference endpoints
--metrics-require-auth        # fold /metrics behind the same key
--trusted-proxy 10.0.0.5      # believe X-Forwarded-For from these peers only
```

**There is no default credential and no default refusal.** Without `--api-key`
every endpoint is open to whoever can reach the port, which is why the shipped
compose file publishes on `127.0.0.1` and why widening it is a two-part change:
`IMP_BIND` and `IMP_API_KEY` together (#1619).

**`--trusted-proxy` is what makes rate limiting work behind a proxy.** Without
it `X-Forwarded-For` is ignored and every request from the proxy shares one
bucket; with it, the header is believed from those peers and the limit is
per-client again. It is not believed from anyone else, because a client that
can write the header can otherwise vary it per request and bypass the limit
entirely (#1614).

Per-request work is capped independently of the rate limit, because one request
can ask for many units of it:

| flag | default | bounds |
|---|---|---|
| `--max-n` | 8 | `n` completions per chat request |
| `--max-batch-items` | 512 | rerank `documents`, embeddings `input` |
| `--max-logit-bias` | 1024 | `logit_bias` entries |
| `--http-read-timeout` | 60 s | socket read |
| `--http-write-timeout` | 600 s | socket write, must outlast a stream |
| `--http-keep-alive-max` | 100 | requests per connection |

**CORS is wide open by design** (`Access-Control-Allow-Origin: *` plus an
`OPTIONS` catch-all), because the built-in web UI and browser clients call the
API directly. If imp is reachable from anywhere other than your own network,
terminate TLS and enforce origin policy at a reverse proxy. imp does not do TLS.

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

`proxy_buffering off` is the line people forget. With it on, streaming responses
arrive in one lump at the end and every client reports a TTFT equal to the total
latency.

## Health, metrics, lifecycle

| endpoint | use |
|---|---|
| `GET /health` | liveness. Answers before a model is loaded |
| `GET /metrics` | Prometheus. TTFT and inter-token-latency histograms, cancellation counters, and a memory breakdown that separates capacity from occupancy |
| `GET /v1/models` | what is loaded, and what else is in the models directory |
| `POST /admin/suspend` | park the weights in host RAM and free the GPU completely. Inference answers 503 while suspended |
| `POST /admin/resume` | restore warm, in seconds, without re-reading weights |

Suspend/resume is the answer to "I need the GPU for something else for ten
minutes" without paying a cold load afterwards. Sessions and KV do not survive
it; weights do.

## Capacity planning

**Plan capacity, do not discover it.** Two properties of this platform make
runtime probing unreliable:

- A successful `cudaMalloc` proves nothing about free VRAM. WSL2/WDDM
  oversubscribes into host memory and returns success. The symptom is a 6.5x
  bandwidth cliff, not an error.
- Free VRAM only ever decreases within a process. The driver never returns a
  process's peak commitment, however cleanly CUDA released it, so anything sized
  from `cudaMemGetInfo` is reading a moving floor.

Size the deployment from the model plus the KV pool you intend to serve, and pin
`runtime.max_seq_len` rather than letting it auto-fit against a moving number.
The tier model and the invariants behind this are in
[`internals/MEMORY.md`](internals/MEMORY.md).

## Serving more than one model

One GPU holds one model. `server.model_swap` (default on) lets a request name a
different model from the models directory: in-flight generations drain first and
are never cancelled, and a failed load restores the previous model rather than
leaving the server empty. The requesting call pays the load, which the warm
weight cache makes cheap on repeats.

If you need strict single-model semantics, set `server.model_swap=false`.
