#pragma once

#include "common/args_common.h"

#include <string>
#include <utility>
#include <vector>

struct ServerArgs : CommonArgs {
    // Shared flags live in CommonArgs (#1209). Inherited, not composed, so
    // every existing `args.<field>` use site is unchanged.

    // imp.conf integration. --config overrides the search-path default;
    // --set is a repeatable key=value applied on top.

    std::string host = "127.0.0.1";
    int port = 8080;
    int max_batch_size = 0;  // --max-batch: decode batch / KV+workspace sizing (0 = engine auto)
    // --lora NAME=PATH (repeatable): PEFT adapters loaded at startup,
    // selectable per request via the "lora" body field.
    std::vector<std::pair<std::string, std::string>> loras;
    std::string models_dir;                     // --models-dir: scan for .gguf files
    std::string api_key;                        // --api-key: require Bearer token auth
    // --metrics-require-auth: gates /metrics behind --api-key too. Off by default since the
    // standard Prometheus scrape is unauthenticated; on, because /metrics discloses the loaded
    // model name, d_model and cumulative token counts to anyone reaching the port.
    bool metrics_require_auth = false;
    std::string reasoning_format = "deepseek";  // --reasoning-format: deepseek or none
    float think_budget =
        0.5f;  // --think-budget: fraction of max_tokens for reasoning (1.0=unlimited, 0=disabled).
               // 0.5 matches docs/usage.md and guarantees answer headroom — at 1.0 a
               // rambling reasoning model eats max_tokens and returns empty content.
    // Server limits
    int max_concurrent = 64;        // --max-concurrent: max simultaneous requests (0=unlimited)
    int request_timeout = 300;      // --request-timeout: per-request timeout in seconds (0=unlimited)
    int rate_limit = 0;             // --rate-limit: max requests per minute per IP (0=unlimited)
    int max_input_tokens = 0;       // --max-input-tokens: reject prompts longer than this (0=unlimited)
    // --trusted-proxy: comma-separated addresses whose X-Forwarded-For this server believes.
    // Empty (default) ignores the header and uses the peer address as the rate-limit key; without
    // this the limit keys on a client-written string, so varying one header bypasses it and every
    // distinct value is a permanent tracker entry (#1614).
    std::string trusted_proxies;
    // Per-request work multipliers. Each of these turns one HTTP request into
    // N units of engine work while counting as one against --rate-limit and
    // --max-concurrent (#1616).
    int max_n = 8;              // --max-n: cap on `n` (chat/completions)
    int max_batch_items = 512;  // --max-batch-items: cap on rerank `documents` / embeddings `input`
    int max_logit_bias = 1024;  // --max-logit-bias: cap on logit_bias entries (#1617)
    int max_images = 8;         // --max-images-per-request: cap on image parts (AUDIT_arch_2026 F2-4)
    // Connection-level limits. Everything here was whatever the build-time
    // cpp-httplib happened to default to (#1622).
    int read_timeout = 60;     // --http-read-timeout: seconds
    int write_timeout = 600;   // --http-write-timeout: seconds, must outlast a long stream
    int keep_alive_max = 100;  // --http-keep-alive-max: requests per connection
    // --allow-remote-images: fetch http(s) image_url from a request body. Default OFF (#1610):
    // on, an unauthenticated caller chooses which host/port this server connects to, reaching
    // loopback, the compose network, or the cloud metadata endpoint - all unnecessary since a data
    // URI needs none of this. When on, the destination is classified and redirects are not
    // followed (image_fetch.h).
    bool allow_remote_images = false;
    std::string prefix_cache_path;  // --prefix-cache: path to persist prefix cache
    std::string log_requests_path;  // --log-requests: append JSONL of every chat/messages request
};

ServerArgs parse_server_args(int argc, char** argv);
void print_server_usage(const char* prog);
