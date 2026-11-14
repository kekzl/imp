#pragma once

#include "core/dispatch_policy.h"  // the eight sections exec/ reads (F-10)

// imp.conf: central runtime configuration, loaded once at startup from a
// TOML file plus CLI-flag overrides.
//
// Loading precedence (first non-empty wins):
//   1. --config <path>              CLI flag (passed via load_with_path)
//   2. $IMP_CONFIG                  environment variable
//   3. ./imp.conf                   working-dir relative
//   4. ~/.config/imp/imp.conf       user config directory
//   5. embedded defaults            (no file)
//
// Per-run overrides come on top via apply_overrides({"section.field=value"}).

#include <string>
#include <string_view>
#include <vector>

namespace imp {

// kv_cache.swa_sizing tri-state (legacy bool literals map to On/Off).

struct RuntimeConfig {
    struct Runtime {
        bool deterministic_gemm = false;
        // Full run-to-run reproducibility for temp=0 agent evals: removes
        // MoE routing and top-k sampling races by selecting deterministic
        // kernel variants, and also implies deterministic_gemm. Off by
        // default (costs throughput via serial/ordered reductions).
        // Legacy env: IMP_DETERMINISTIC=1.
        bool deterministic = false;
        // "auto" | "always" | "never". "always" == "auto": both still demote
        // to eager on host-resident experts, streaming KV, or a failed
        // pinned buffer, since a captured decode there replays stale pointers.
        std::string cuda_graphs = "auto";
        // Engine warmup: two tiny BOS requests at init (~2-4 s on a 30B)
        // that pre-arm the decode graph pool (mark_process_warm), so the
        // first real request takes the same kernel path as later ones and
        // avoids greedy flips on near-tie logits. False trades this for
        // init time. Gemma-4/MXFP4 models keep their own skip (Engine::warmup).
        bool warmup = true;
        // NOTE: full determinism also needs stable cuBLAS algo selection
        // across processes, see runtime.deterministic_gemm.
        int max_seq_len = 0;               // 0 = use model default
        // Hard VRAM budget for this process, MiB (0 = uncapped). Every sizing
        // decision sees a virtual GPU of this size, so processes can share a
        // card; leave ~1 GiB real headroom (cuBLAS internals sit outside it).
        // Overridden by --vram-budget / ImpConfig.vram_budget_mb.
        int vram_budget_mb = 0;
        bool no_pdl = false;
        // Drop the checkpoint's Qwen3-VL tower at init: image/video parts answer 400, its arena
        // reservation goes to the expert cache (Flash-Next: 1079 MiB, 396 -> 422 slots/layer).
        bool no_vision = false;
        // Drop the checkpoint's Gemma-4 audio tower at init: input_audio parts answer 400, its
        // arena reservation (E4B, shape arithmetic: 566 MiB weights + 80 MiB clip scratch) stays free.
        bool no_audio = false;
        // Qwen3-VL patch budget, in 16x16 patches: ceiling that pulls the
        // preprocessor's max_pixels down and sizes every encoder workspace
        // (not a refusal check). 4096 = 1024x1024.
        int vision_max_patches = 4096;
        // EVS video token pruning rate q (vLLM video_prune/evs.py): a Qwen3-VL video keeps
        // max(tokens per frame pair, int(tokens * (1 - q))), the least changed tokens go.
        // 0 = off; a video part's "pruning_rate" overrides it. Outside [0, 1): rejected.
        float video_pruning_rate = 0.0f;
        // Capture prefill into a CUDA graph in addition to decode. Default
        // ON: safe under relaxed capture mode (cuda_graph.cu,
        // prefill_graph_blockers_2026_05_14.md, Blocker B). Opt out per
        // model via `--set runtime.prefill_graph=false` if one regresses. Serial prefill only
        // (not the ragged scheduler path): offset-0 chunks, F16 or calibrated FP8 KV, a shape
        // captured on its second sighting and kept across context resets.
        bool prefill_graph = true;
        // Prompts up to N tokens keep a prefill graph each, captured at startup (two dummy prefills
        // per length, F16 KV): a first request of that length replays instead of running eager
        // (Gemma-4 12.4 -> 6.6 ms TTFT; +150 MiB, +0.9 s at 32). Longer prompts share one runner.
        int prefill_graph_prewarm = 32;
        // 0 = auto: engine sizes the decode batch from the model's weight
        // footprint (a >20 GiB MoE auto-picks 1). Positive value forces it.
        int max_batch_size = 0;
        // Cap the prefill chunk (tokens) while other sequences are DECODING:
        // prefill and decode share one stream, so a chunk forward blocks
        // decode steps by its own latency. Lower = better decode latency,
        // worse ingest TTFT; 512 for latency-critical multi-tenant serving,
        // 0 to disable (full chunk, returns once nobody is decoding). -1 =
        // per-arch default (2048 where chunked prefill is supported, else
        // 0). CLI flag wins over the file value (#1645).
        int prefill_chunk_size = -1;
        // Paces the post-wave-start prefill to ~2 prompts per engine step,
        // bounding a concurrent STREAMER's inter-token gap during another
        // session's large ingest (#1643). Raise (e.g. 4096) for burst-shaped
        // rather than mixed workloads; 0 = off. GDN hybrids with riders cap
        // at min(this, 512), and the cap scales with waiters per decoder
        // (prefill_pacing.h, kPrefillCapFairness).
        int prefill_chunk_decode_cap = 1024;
        // Tokens of `max_tokens` reserved for the ANSWER on a reasoning
        // model: the engine injects `</think>` once reasoning reaches
        // max_tokens - max(think_answer_reserve, max_tokens/4), or the
        // `think_budget` fraction, whichever is LATER. Below 0 reads as 0
        // (no floor: the fraction alone decides). Raise for models with long
        // structured answers, lower for a short-verdict agent loop.
        int think_answer_reserve = 256;
        // Upper bound on reasoning tokens before the engine injects `</think>`, whatever
        // max_tokens is (imp-server defaults max_tokens to the remaining context). 0 = no cap.
        int think_max_tokens = 32768;
        // Cross-sequence prefill batching (roadmap 0(d)): assembles the
        // prefill chunks of several admitted requests into ONE ragged
        // forward (GEMMs/norms/elementwise over concatenated rows, attention
        // and the GDN conv loop per sequence, GDN scan via a row-offset
        // table), instead of one sequence per forward. Requests with vision,
        // constraints, logprobs, embeddings/rerank scoring, and Mamba2/MLA
        // models, MTP, SWA sizing, gdn.ref_kernel
        // keep the serial path.
        bool prefill_batch = true;

        // KV admission (#2486): decode tokens reserved per request. 0 = DecodeLengthEstimate
        // (p90 of the last 64 finished outputs), -1 = max_tokens, > 0 = fixed. Below max_tokens
        // only when KV swap can catch an overrun (engine_kv_swap.cpp), else -1.
        int admission_decode_tokens = 0;
    } runtime;

    cfg::KVCache kv_cache;

    // Runtime RoPE-scaling override: stretches a model's usable context past
    // its native window without editing the checkpoint. Sets the same
    // ModelConfig fields the GGUF/HF loaders set from rope_scaling metadata,
    // before max_seq_len auto-detection, so it flows into KV sizing, YaRN
    // corr-dims, and the MTP draft head. Refused (logged) for per-dimension
    // frequency tables (LongRoPE/llama3), MLA (mscale entanglement), NoPE.
    struct Rope {
        // "" = off (model metadata only). "yarn" | "linear".
        std::string scaling;
        // Context-extension factor (> 1.0), e.g. 4.0 stretches 32k → 128k.
        float factor = 1.0f;
        // Original (native) training context the factor applies to.
        // 0 = auto: the model's declared rope_n_ctx_orig, else max_seq_len.
        int orig_ctx = 0;
        // HF attn_factor multiplier. The kernel already applies the YaRN
        // mscale 1 + 0.1*ln(factor) internally; leave at 1.0 unless matching
        // a checkpoint that shipped a custom attn_factor.
        float attn_factor = 1.0f;
        float beta_fast = 32.0f;
        float beta_slow = 1.0f;
    } rope;

    // VRAM budget-planner tuning. Governs compute_vram_budget() only;
    // pre-dequant phases keep their own internal reserve floors.
    struct Vram {
        // Commits slot-shaped pools on demand instead of at init (SSM/GDN
        // state slab per admitted sequence, engine arena per tenant, vision
        // tower on first image). The plan still charges every byte and
        // reserves address space at init; a commit is refused (request
        // waits) rather than spilled when the card cannot spare it. Needs
        // CUDA VMM, else every pool is fixed as before.
        bool lazy_commit = true;
        // Fraction of post-reserve/post-weight-cache VRAM the KV pool
        // targets. Clamped to [0.05, 1.0] at use; 1.0 = no cap (B-29f: the
        // library reserve + 4 % allocator headroom are the only margin).
        float kv_fraction = 1.0f;
        // Free-VRAM reserve floor as % of (budget-visible) total VRAM,
        // floored at the 4 % allocator headroom. Clamped to [0, 50] at use.
        // Default 0 = the headroom alone (1304 MiB on 32 GB); raise to opt in.
        int reserve_floor_pct = 0;
        // Fixed CUDA/cuBLASLt charge on the first forward pass, MiB;
        // not a workspace imp allocates, so the budget pass cannot otherwise
        // see it (docs/internals/MEMORY.md A1.5). 16-973 MiB measured 2026-10-31 (1536 default), invariant
        // to batch/context on this target.
        //   -1 = built-in measured constant (default)
        //    0 = charge nothing (driver/toolkit where it does not apply)
        //   >0 = measured value for THIS host, MiB
        // Plan input since B69: the KV plan charges it; warmup re-plans KV at this start's measurement
        // (#2436), weight caches and IMMA planes keep the charge.
        int library_reserve_mb = -1;
        // Where to remember what the first forward ACTUALLY claimed, so a
        // later start on the same model charges the measured value instead
        // of the constant (AUDIT B41/B49). Empty = default location, "off" =
        // disabled. A cache miss or unwritable path is never fatal.
        std::string library_reserve_cache;
    } vram;

    cfg::Attention attention;

    cfg::MoE moe;

    cfg::GDN gdn;

    cfg::GEMM gemm;

    // Model-specific knobs (e.g. Gemma4) live in ModelConfig::Overrides, not
    // here: see src/model/model_config.h.

    struct Server {
        // Prefix caching: reuse KV blocks for shared prompt prefixes. Default
        // ON for server/CLI (#758: OFF meant the prebuilt image never cached
        // without an imp.conf). Library/C-API embedders drive EngineConfig
        // directly and stay off-by-default; the engine ORs this in at init.
        // Ship gate: PrefixCacheE2ETest. Hybrid (SSM/GDN) models additionally
        // need the recurrent snapshot store below.
        bool prefix_cache = true;
        // Cap on cache_control/cache_prompt-pinned blocks, % of the KV pool.
        int prefix_pin_budget_pct = 25;
        // session_id pins (#2407): released after this many idle seconds. 0 = close or budget only.
        int session_ttl_s = 600;
        // Serve a model other than the loaded one by swapping to it instead
        // of answering 404. Swap is serial: in-flight generations drain
        // first (same contract as /admin/suspend), then the old model is
        // torn down and the requested one loaded. The name must resolve
        // inside the models directory, so a typo still 404s. Cost mitigated
        // by the warm weight cache (#956); false keeps the single-model
        // contract and fails fast instead.
        bool model_swap = true;
        // How long a swap waits for in-flight generations to finish before
        // giving up and keeping the current model (503, nothing torn down).
        int model_swap_drain_ms = 60000;
        // Device budget, MiB, for recurrent-state snapshots: what makes
        // prefix caching work on hybrid (SSM/GDN) models, since KV blocks
        // alone cannot skip prefill there. One snapshot = one per-sequence
        // state slab, saved per prefill, LRU-evicted; pre-allocated at
        // engine init and accounted in the expert-offload reserve. imp-cli
        // --bench pins this to 0. 0 disables snapshots and hybrid prefix
        // caching (dense models unaffected).
                        int recurrent_snapshot_mb = 256;
        // Pinned host memory for recurrent snapshots evicted from the device
        // tier (hybrid prefix caching beyond recurrent_snapshot_mb / slab
        // slots): more concurrent multi-turn sessions keep their tail-only
        // prefill. 0 = off.
        int recurrent_snapshot_host_mb = 2048;
        // OpenTelemetry span export (OTLP/HTTP, JSON): the full traces URL,
        // e.g. http://localhost:4318/v1/traces; "" = off. One SERVER span
        // per generation request (queue / prefill / decode children), joined
        // to the client's trace when it sends a W3C `traceparent` header.
        std::string otlp_endpoint;
        std::string otlp_service_name = "imp-server";
    } server;

    struct WarmCache {
        // On-disk warm weight cache: persists TRANSFORMED weight uploads
        // (BF16->FP16 conversions, dequants, split layouts) next to the
        // model so later boots skip re-converting. Raw-from-source uploads
        // are never stored, so size is near-zero for raw GGUF/NVFP4-prequant
        // and ~model-size only for BF16-dense. Guarded by a format version
        // and a content fingerprint; any mismatch is a normal cold load.
        bool enabled = true;
        // Where to store cache files. Empty = next to the model
        // ("<file>.impwcache" / "<dir>/.imp_warm_cache"). Point at a writable
        // volume when the model dir is read-only for the serving user (the
        // prebuilt container runs as uid 1001); files are then named
        // "<model-basename>-<path-hash>.impwcache" inside it.
        std::string dir;
    } warm_cache;

    struct Suspend {
        // Suspend-to-RAM (/admin/suspend): after teardown, also
        // cudaDeviceReset() so the CUDA primary context (~300-600 MiB) is
        // released and the GPU reads ~0 MiB for this process. False if a
        // foreign library holds CUDA state the reset would orphan (imp
        // re-arms everything at the next init).
        bool device_reset = true;
        // Host RAM the snapshot must leave free (MemAvailable gate) on top of
        // the snapshot bytes themselves.
        int host_ram_headroom_mb = 2048;
    } suspend;

    struct Paths {
        std::string mmproj;
    } paths;

    // n-gram (prompt-lookup) speculative decoding: drafts come from suffix
    // matches against the request's own prompt+output tokens, no draft
    // model or MTP head. Verify replays the draft as a teacher-forced chunk
    // and accepts the longest argmax-matching prefix, so output is
    // token-identical to plain greedy decode. spec_verify_gates_ok_ confines
    // engagement to batch-1/greedy/no-penalty-window/no-json/no-logprobs/
    // non-recurrent requests (MoE additionally needs native-NVFP4 experts,
    // see `moe` below); everything else falls back cleanly, so default-ON
    // is a no-op elsewhere.
    cfg::Speculative speculative;

    cfg::Diagnostics diagnostics;

    // Collects per-input-channel activation magnitudes during a forward
    // pass for imp-quantize's AWQ scale search (not an inference feature: a
    // prefill over a corpus whose only output is this file). Also disables
    // CUDA graphs, since the accumulator is allocated lazily per weight and
    // capture forbids that.
    struct Calibration {
        bool enabled = false;
        // Where imp_calibration_write() puts the file. Empty means the caller
        // supplies the path.
        std::string out_path;
        // Also collect mean x^T x per weight input for imp-quantize --gptq (row 101), written
        // next to the file as <out>.hess. FP32 [K, K] each: Qwen3-1.7B about 5.6 GB.
        bool hessian = false;
    } calibration;

    // ----- Loading -----

    // Find a config file in the search-path order documented above.
    // Returns empty string if no file is found.
    static std::string find_default_path();

    // Load from disk; returns true on success. On parse error, the struct
    // is left at its default state and an error is logged.
    [[nodiscard]] bool load_from_file(const std::string& path);

    // Apply key=value strings (e.g. "kv_cache.dtype=fp8") via dotted-section
    // lookup. Returns entries that bound to nothing: a `--set` naming an
    // unknown key is a typo and should stop the caller. An unknown key in
    // imp.conf stays a warning only, since a config file may outlive the
    // build that understood it.
    [[nodiscard]] std::vector<std::string> apply_overrides(const std::vector<std::string>& kvs);

    // Convenience: locate + load + apply overrides + log a one-line summary.
    // Pass empty path to use the search-path default. Pass `rejected` to take
    // the unbound overrides and decide yourself (both tool mains exit on them);
    // leave it null and they are only logged.
    static RuntimeConfig load(const std::string& explicit_path, const std::vector<std::string>& overrides,
                              std::vector<std::string>* rejected = nullptr);

    // Dotted keys this configuration actually took from a file or a
    // `--set`, in application order. Needed so an "auto" default that
    // resolves a pair of keys knows which half the operator set explicitly
    // (e.g. speculative.mtp_k=-1 pairing with ngram off) instead of
    // silently overriding it.
    std::vector<std::string> explicit_keys;
    [[nodiscard]] bool was_set(std::string_view dotted_key) const {
        for (const auto& k : explicit_keys)
            if (k == dotted_key)
                return true;
        return false;
    }
};

// Token burst of the autonomous decode graph loop (non-streaming, non-speculative, not
// deterministic): the worker polls cancel between bursts; output is identical, 128 trades
// relaunch overhead against cancel latency. Also the SWA burst span the KV group is sized for.
inline constexpr int kDecodeBurst = 128;
// A prompt whose block-aligned prefix is shorter takes no prefix-cache snapshot and no prefill
// split at the boundary (two eager chunks plus a sync per first turn). The finish-time
// transcript snapshot splits nothing and ignores this floor.
inline constexpr int kSnapshotMinPromptTokens = 256;

// ---- Pending-config handoff (tool-main → Engine) -----------------------
//
// Tool mains (imp-cli, imp-server) load a RuntimeConfig from imp.conf + CLI
// overrides at startup and hand it to Engine::init without passing it
// through the ABI-stable ImpConfig C struct (Engine is constructed inside
// src/api/imp_api.cpp).
//
// Workflow: tool main calls set_pending_runtime_config(loaded_cfg) once;
// imp_context_create() later pulls it via take_pending_runtime_config() and
// passes it to Engine::init. Lifetime is bounded to a single Engine
// construction; there is no per-call accessor.
void set_pending_runtime_config(RuntimeConfig cfg);
RuntimeConfig take_pending_runtime_config();

}  // namespace imp
