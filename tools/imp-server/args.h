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
    // --metrics-require-auth: also gate /metrics behind --api-key. Off by
    // default because the standard Prometheus scrape (monitoring/) is
    // unauthenticated; on, because /metrics discloses the loaded model name,
    // d_model and cumulative token counts to anyone who can reach the port.
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
    // --allow-remote-images: fetch http(s) image_url from a request body.
    // Default OFF (#1610). With it on, an unauthenticated caller chooses which
    // host and port this server connects to, and the interesting targets are
    // the ones only the server can reach: loopback, the compose network, the
    // cloud metadata endpoint. A data URI needs none of this and is what every
    // real client sends. When on, the destination is classified and redirects
    // are not followed - see image_fetch.h.
    bool allow_remote_images = false;
    std::string prefix_cache_path;  // --prefix-cache: path to persist prefix cache
    std::string log_requests_path;  // --log-requests: append JSONL of every chat/messages request
};

ServerArgs parse_server_args(int argc, char** argv);
void print_server_usage(const char* prog);
