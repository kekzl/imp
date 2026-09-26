<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# Deployment

Running imp as a service. [`LIMITATIONS.md`](LIMITATIONS.md) first: several
entries there are deployment decisions - one GPU holds one model, nothing is covered by an SLO.

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

The cache volume is not optional in a service context: without it every restart re-transforms the weights into `<model-name>-<hash>.impwcache` under `~/.cache/imp/warm` (`[warm_cache] dir` to override) again.

- Must be writable by the container user; a fresh volume owned by root silently disables the warm weight cache (on by default, `[warm_cache] enabled = false` to opt out) and the library-reserve measurement.
- Raw quant payloads are never duplicated: near-zero size for raw-served GGUF/NVFP4-prequant, ~model-size for BF16-dense.
- A stale cache (changed model file) is detected and ignored.

## Configuration

Three layers, later wins: `imp.conf` → `--config <file>` → `--set key=value`.

```bash
--set runtime.max_seq_len=32768 --set kv_cache.dtype=fp8
```

**A `--set` key that does not exist is an error, not a warning.** Deliberate:
a typo used to measure the default silently. An unknown key in `imp.conf`
stays a warning, because a config file may outlive the build that understood
every key in it.

Every key, with defaults: `src/runtime/config.h` is the source of truth; [`CONFIG.md`](CONFIG.md#impconf) is the readable version.

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
most families and costs it on one, because `auto` resolves NVFP4 where that
has been measured safe and FP8 is the wider type there - the engine says so at
startup ([`plans/2026-08-29-qwen38-long-context-posture.md`](plans/2026-08-29-qwen38-long-context-posture.md)).
The legacy KV names also outrank a `kv_cache.dtype` from `IMP_SET` in either
order - `--kv-fp8` sets the dtype directly - so the entrypoint warns when both are set.

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

Codes above 9 are unused; `imp-quantize` returns 1 for usage errors (was 2), same as every other invalid argument.

## Auth and exposure

```bash
--api-key "$IMP_API_KEY"      # bearer auth on the inference endpoints
--metrics-require-auth        # fold /metrics behind the same key
--trusted-proxy 10.0.0.5      # believe X-Forwarded-For from these peers only
```

**No default credential, no default refusal.** Without `--api-key` every
endpoint is open to whoever can reach the port: the shipped compose file
publishes on `127.0.0.1`; widening it is a two-part change, `IMP_BIND` plus `IMP_API_KEY` together (#1619).

**`--trusted-proxy` makes rate limiting work behind a proxy.** Without it
`X-Forwarded-For` is ignored and every request from the proxy shares one
bucket; with it the header is believed from those peers only - believing it
from anyone else would let a client vary the header and bypass the limit (#1614).

Per-request work is capped independently of the rate limit (one request can
ask for many units of it):

| flag | default | bounds |
|---|---|---|
| `--max-n` | 8 | `n` completions per chat request |
| `--max-batch-items` | 512 | rerank `documents`, embeddings `input` |
| `--max-logit-bias` | 1024 | `logit_bias` entries |
| `--max-images-per-request` | 8 | `image_url` parts; each is decoded at full resolution on a worker, a side is capped at 16384 px in the decoder |
| `--http-read-timeout` | 60 s | socket read |
| `--http-write-timeout` | 600 s | socket write, must outlast a stream |
| `--http-keep-alive-max` | 100 | requests per connection |

**CORS is wide open by design** (`Access-Control-Allow-Origin: *` plus an `OPTIONS` catch-all): the built-in web UI and browser clients call the API
directly. imp does not do TLS; terminate TLS and enforce origin policy at a reverse proxy if reachable beyond your own network.

A minimal nginx front:

```nginx
limit_conn_zone $binary_remote_addr zone=imp_conn:1m;

location / {
    proxy_pass         http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header   Connection "";
    proxy_buffering    off;        # required: SSE must not be buffered
    proxy_read_timeout 3600s;      # a long generation is one long response
    limit_conn         imp_conn 16; # per-peer connection cap, see below
}
```

`proxy_buffering off` is the line people forget: with buffering on, streaming
responses arrive in one lump at the end and every client reports TTFT equal
to total latency. `limit_conn` is the connection-level backpressure imp
itself does not have: `--max-concurrent`/`--rate-limit` run only after a
request body is fully read, and the worker pool (`--max-concurrent + 8`
threads, 60 s read timeout) can be tied up by a peer holding open connections
that never finish a body - cap connections per peer at the proxy.

## Health, metrics, lifecycle

| endpoint | use |
|---|---|
| `GET /health` | liveness. Answers before a model is loaded, and 200 while suspended or mid-swap |
| `GET /ready` | readiness. 200 only when an inference request would be taken now; 503 with `code` `no_model`, `suspended`, `swapping` or `draining` otherwise. Point an orchestrator's readiness probe here, its liveness probe at `/health` |
| `GET /metrics` | Prometheus. Latency histograms (request, TTFT, inter-token, queue), decode batch size, refusal and cancellation counters, and a memory breakdown that separates capacity from occupancy |
| `GET /v1/models` | what is loaded, and what else is in the models directory. Hidden entries (a leading `.`) are skipped. A request naming another listed model swaps to it (`server.model_swap`, default on, 2 s for Qwen3-4B Q8_0); a name not in the directory gets `404 model_not_found` |
| `POST /admin/suspend` | park the weights in host RAM (`[suspend] device_reset`, default on: also `cudaDeviceReset()` so `nvidia-smi` reads ~0 MiB) and free the GPU. Inference answers 503, `/health` reports `"suspended": true` at 200. Fails cleanly (507) when host `MemAvailable` is below snapshot size + `[suspend] host_ram_headroom_mb`. Models whose device buffers are transformed in place after upload (native MXFP4 GGUF, gpt-oss, Gemma-4 fused-expert split) are refused with 501 |
| `POST /admin/resume` | restore from RAM (no mmap re-read, no requantization), serving again in seconds. Sessions/KV do not survive; only weights stay warm |

**Shutdown** (SIGTERM/SIGINT): the listener stops, requests already accepted answer 503, `/ready` reports `draining`, and in-flight generations get `server.model_swap_drain_ms` (60 s default) to finish before teardown.

- `docker-compose.yml` sets `stop_grace_period: 75s` to cover that; a bare `docker stop` (10 s default) kills a draining server.
- While a swap or `/admin/suspend` holds the engine lock, inference answers 503 (`overloaded_error`) within 250 ms rather than parking a worker thread.

**Context-window reporting.** `/v1/models` carries vLLM's `max_model_len` and llama.cpp's `meta.n_ctx_train`; `GET /props` returns llama.cpp's `n_ctx` (top-level and under `default_generation_settings`); `GET /info` TGI's `max_total_tokens`/`max_input_tokens`.

- All three report the smaller of the resolver's plan and the pool's clamp since #1542 (97204 vs 52256 measured on a tight Qwen3.8-27B-NVFP4 card).
- A growable pool (`kv_cache.growable`, default on) counts by its ceiling: starts at `kv_cache.growable_initial_pct` (25%) of the plan, grows at admission (measured: 875 of 13264 blocks committed, 131072 advertised).
- `/health` reports both: `kv_capacity_tokens` for the commit, `kv_ceiling_blocks` for the ceiling.

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

**Host RAM: read `RssAnon`, not `VmRSS`.** The loader maps the checkpoint and faults it in, dropping those pages again once weights are on the GPU (#1934).

- Before that it held the whole file resident: Qwen3.8-27B-NVFP4-vllm read 21.53 GiB of `VmRSS` (18.48 GiB mapping, 1.01 GiB anonymous) while `docker stats` showed 3.2 GiB, because the cgroup does not account a shared file mapping the way `ps` does.
- Weights left on host by an offload placement refault on demand and are counted again while being read.

## Serving more than one model

One GPU holds one model; `server.model_swap` (default on, table above) drains in-flight generations first, never cancels them, and a failed load restores the previous model.

- The requesting call pays the load, cheap on repeats via the warm weight cache.
- Strict single-model semantics: `server.model_swap=false`.
