#pragma once

#include "core/dispatch_policy.h"  // the nine sections exec/ reads (F-10)

// imp.conf — central runtime configuration.
//
// Replaces ~50 ad-hoc IMP_*-prefixed environment variables that were scattered
// over ~80 getenv() call sites in src/runtime/ and src/exec/. The same values
// now flow through a single RuntimeConfig struct loaded once at startup,
// optionally from a TOML file, with CLI-flag overrides on top.
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
#include <vector>

namespace imp {

// kv_cache.swa_sizing tri-state (legacy bool literals map to On/Off).

struct RuntimeConfig {
    struct Runtime {
        bool deterministic_gemm = false;
        // Opt-in full reproducibility mode for temperature=0 agent evals.
        // When true, run-to-run non-determinism in MoE token routing
        // (atomic expert-bucket scatter ordering) and top-k sampling
        // (atomicMax/atomicAdd softmax-stat races) is eliminated by
        // selecting deterministic kernel variants. It ALSO implies
        // deterministic_gemm (timing-based cuBLAS algo selection is itself
        // a non-determinism source), so a single switch covers GEMM +
        // routing + sampling. Costs a little throughput (serial / ordered
        // reductions), so it is strictly OFF by default — the default code
        // path runs the exact same kernels as before with zero overhead.
        // Legacy env: IMP_DETERMINISTIC=1.
        bool deterministic = false;
        std::string cuda_graphs = "auto";  // "auto" | "always" | "never"
        // Engine warmup (two tiny BOS requests at init, ~2-4 s on a 30B).
        // Default ON since the greedy request-order-independence fix: warmup
        // pre-arms the decode graph pool (mark_process_warm), so the FIRST
        // real request takes the same graph-kernel path at the same step
        // indices as every later one. Without it the first request runs one
        // step on a different kernel mix and greedy output can flip on
        // near-tie logits (the 30B-NVFP4-MoE temp=0 flipper). Set false to
        // trade reproducibility for init time (dev/CI). Gemma-4 and MXFP4
        // models keep their warmup skips (Engine::warmup).
        bool warmup = true;
        // NOTE: full run-to-run determinism additionally needs stable cuBLAS
        // algo selection across processes — see runtime.deterministic_gemm.
        int max_seq_len = 0;               // 0 = use model default
        // Hard VRAM budget for THIS process (MiB, 0 = uncapped). Every sizing
        // decision (weight caches, KV clamp, expert offload, workspaces,
        // upload gates) sees a virtual GPU of this size, so multiple
        // imp-server processes can share one card without overcommitting it.
        // Best-effort: leave ~1 GiB real headroom between the sum of budgets
        // and the card (small fixed buffers + cuBLAS internals sit outside).
        // CLI flag --vram-budget / C-API ImpConfig.vram_budget_mb override.
        int vram_budget_mb = 0;
        bool no_pdl = false;
        bool debug_raw = false;        // raw stream debug
        bool no_vision_graph = false;  // disable SigLIP graph capture
        // Qwen3-VL patch budget: the largest image the dynamic-resolution
        // encoder will accept, in 16x16 patches. It is a CEILING that pulls
        // the preprocessor's max_pixels down, not a check that refuses an
        // image, and it sizes every encoder workspace. 4096 = 1024x1024.
        int vision_max_patches = 4096;
        // cudaStreamCaptureMode passed to begin_capture / conditional bodies:
        // "global" | "relaxed" (default) | "thread_local". "relaxed" drops the
        // cross-thread sync constraint that CUTLASS 3.x grouped-GEMM
        // collective scheduler is suspected to deadlock on under prefill
        // capture (Blocker B in prefill_graph_blockers_2026_05_14). Default
        // flipped to "relaxed" 2026-05-16 as the M3-probe for prefill_graph
        // unblock — `cudaStreamCaptureModeRelaxed` is a strict superset of
        // capturable behaviors so the decode fast path that previously
        // worked under "global" continues to work, while the prefill path
        // gets a real chance of capturing without hanging.
        //
        // Set graph_capture_mode = "global" via imp.conf to opt back into
        // the legacy strict mode (any decode regression should also be
        // investigated under "thread_local" before assuming relaxed is the
        // cause).
        std::string graph_capture_mode = "relaxed";
        // Capture prefill into a CUDA graph (in addition to decode). Default
        // flipped 2026-05-17 after the M3 Phase 4 A/B sweep across
        // Gemma-4-26B-NVFP4 / Qwen3.6-35B-NVFP4 / Qwen3-Coder-30B-FP4
        // (3 trials × 4 capture modes each, harness at
        // /tmp/imp-bench-results/run_bench.sh): no hang on any
        // (model, capture_mode) combination — Blocker B
        // (`prefill_graph_blockers_2026_05_14.md`) is gone now that
        // `graph_capture_mode = "relaxed"` is the default. Decode tg
        // is flat ±1-2% across all four capture-mode configs for every
        // model; prefill pp is variance-dominated (cuBLAS algo-selection
        // noise documented in CLAUDE.md) but the candidate (relaxed)
        // never regressed below baseline. Opt out via
        // `--set runtime.prefill_graph=false` or imp.conf if a model
        // regresses.
        bool prefill_graph = true;
        // 0 = auto: the engine sizes the decode batch from the model's weight
        // footprint (a >20 GiB MoE auto-picks 1). A positive value forces it.
        // (Was 4, which both contradicted the documented "0 = auto" semantics
        // and never reached engine sizing — it only acted as the decode cap.)
        int max_batch_size = 0;
        // Bound for the autonomous decode graph loop on a NON-streaming request
        // with speculation off (which would otherwise run UNBOUNDED to
        // max_tokens on-device, so a client disconnect/timeout — polled only
        // between bursts — couldn't interrupt it and burned a full generation).
        // The loop runs in bursts of this many tokens and returns to the host
        // to re-poll cancellation; output is identical (same decode, chunked).
        // Larger = less relaunch overhead but higher cancel latency. Streaming
        // and speculation paths are unaffected. <=0 restores the old unbounded
        // behavior. IGNORED when `deterministic` is set: the unbounded loop is
        // the only greedy-bit-reproducible decode path, and evals run to
        // completion (no mid-burst cancel needed), so determinism wins there.
        int decode_burst = 128;
        // Cap the prefill chunk while other sequences are DECODING: prefill
        // and decode share one stream, so every chunk forward inserts its
        // full latency between two of their decode steps. Measured (Qwen3-8B
        // Q8, 7.2k-token ingest against one active decoder): 2048 → decoder
        // p95 inter-token 164 ms; 1024 → 94 ms at +27% ingest TTFT; 512 →
        // 65 ms at +85%. 1024 is the default compromise; set 512 for
        // latency-critical multi-tenant serving, 0 to disable (full chunk).
        // The full chunk returns as soon as nobody is decoding.
        int prefill_chunk_decode_cap = 1024;
        // Hybrid (SSM/GDN) decode fairness: the recurrent scan kernels are
        // single-sequence, so concurrent sessions time-slice the decode.
        // This is the slice length in tokens — after it, the engine rotates
        // to the next DECODING request (round-robin). Rotation re-captures
        // the decode graphs for the new sequence's state slot (~10-20 ms),
        // so smaller values buy latency fairness at capture overhead
        // (128 ≈ 1-2% at typical hybrid decode rates). 0 restores the old
        // head-of-line behavior (first request runs to completion).
        int hybrid_decode_quantum = 128;
        // Pipelined batched decode (n>=2, CUDA graphs on): keep ONE decode
        // step in flight — step N+1 (device-side token chain + graph replay
        // + sampler enqueue) is enqueued BEFORE step N's tokens are read
        // back, so host bookkeeping/SSE delivery overlaps GPU compute
        // instead of idling it (the ~15-20% step tail at n=16, 2026-07-12).
        // Engages for async-sampleable rows (greedy / top-k<=128 / top-p /
        // min-p / typical-p; rep/freq/presence penalties served via a
        // device-side token history) on non-SSM models. Rows with DRY/
        // mirostat/logit-bias/constraints/logprobs, and SWA/StreamingLLM/
        // residual-KV configs, keep the per-step path.
        bool decode_pipeline = true;
    } runtime;

    cfg::KVCache kv_cache;

    // Runtime RoPE-scaling override — stretch a model's usable context past
    // its native window without editing the checkpoint. Sets the same
    // ModelConfig fields the GGUF/HF loaders set from model-declared
    // rope_scaling metadata, before max_seq_len auto-detection, so the
    // extended window flows into KV sizing, YaRN corr-dims, and the MTP
    // draft head. Refused (logged) for models with per-dimension frequency
    // tables (LongRoPE / llama3), MLA (mscale entanglement), and NoPE.
    struct Rope {
        // "" = off (model metadata only). "yarn" | "linear".
        std::string scaling;
        // Context-extension factor (> 1.0), e.g. 4.0 stretches 32k → 128k.
        float factor = 1.0f;
        // Original (native) training context the factor applies to.
        // 0 = auto: the model's declared rope_n_ctx_orig, else max_seq_len.
        int orig_ctx = 0;
        // HF attn_factor multiplier. The kernel already applies the YaRN
        // paper mscale 1 + 0.1*ln(factor) internally — leave at 1.0 unless
        // matching a checkpoint that shipped a custom attn_factor.
        float attn_factor = 1.0f;
        float beta_fast = 32.0f;
        float beta_slow = 1.0f;
    } rope;

    // VRAM budget-planner tuning. Governs compute_vram_budget() only —
    // the pre-dequant phases keep their own internal reserve floors.
    struct Vram {
        // Fraction of post-reserve/post-weight-cache VRAM the KV pool
        // targets. Clamped to [0.05, 0.95] at use.
        float kv_fraction = 0.8f;
        // Free-VRAM reserve floor as % of (budget-visible) total VRAM,
        // still floored at 256 MiB absolute. Clamped to [0, 50] at use.
        // Default 10 ≈ 3.2 GiB on a 32 GiB card — lower to trade headroom
        // for KV pool.
        int reserve_floor_pct = 10;
        // The fixed charge CUDA/cuBLAS/CUTLASS claim on the FIRST forward
        // pass, in MiB. Measured at ~3900 on this target and invariant to
        // batch (1..16) and context (1024..4096) — it is not a workspace imp
        // allocates, and the old budget pass cannot see it, which is why it
        // hands the KV pool a number that much too optimistic
        // (docs/internals/MEMORY.md A1.5).
        //   -1 = use the built-in measured constant (default)
        //    0 = charge nothing (for a driver/toolkit where it does not apply)
        //   >0 = the measured value for THIS host, in MiB
        // Advisory until A7 step 6: today it only feeds the shadow plan that
        // is logged next to the live budget.
        int library_reserve_mb = -1;
        // Where to remember what the first forward ACTUALLY claimed, so the
        // second start on a model charges the measured value instead of the
        // constant (AUDIT B41/B49). Empty = the default cache location;
        // "off" disables it. A cache miss or an unwritable path is never fatal.
        std::string library_reserve_cache;
    } vram;

    cfg::Attention attention;

    cfg::MoE moe;

    cfg::GDN gdn;

    cfg::GEMM gemm;

    // (RuntimeConfig::Gemma4 lived here through Phase 4 of the architecture
    // refactor. Phase 5 Track A moved it to ModelConfig::Overrides::Gemma4 —
    // see src/model/model_config.h. Model-specific knobs do not belong on a
    // global runtime singleton.)

    cfg::Generation generation;

    struct Server {
        // Prefix caching: reuse KV blocks for shared prompt prefixes. Default
        // ON for the server/CLI — this is the documented behaviour (README,
        // imp.conf.example) and what delivers the advertised warm-prompt TTFT
        // win + cache_read_input_tokens reporting (#758: shipping OFF meant the
        // prebuilt image never cached unless an imp.conf opted in). Library /
        // C-API embedders are unaffected — they drive EngineConfig directly
        // (off-by-default there). The engine ORs this into
        // EngineConfig.use_prefix_caching at init. PrefixCacheE2ETest is the
        // ship gate. For hybrid (SSM/GDN) models it additionally requires the
        // recurrent snapshot store below.
        bool prefix_cache = true;
        // Cap on cache_control/cache_prompt-pinned blocks, % of the KV pool.
        int prefix_pin_budget_pct = 25;
        // Serve a model other than the loaded one by swapping to it, instead of
        // answering 404. Agent harnesses run a big model beside a small one
        // (router, sub-agents, autocomplete) and 32 GB fits one at a time, so
        // the swap is serial: in-flight generations drain first (never
        // cancelled, same contract as /admin/suspend), then the old model is
        // torn down and the requested one loaded. The requested name must
        // resolve inside the models directory — an unknown name is still a 404,
        // so a typo cannot trigger a load. Cost is one model load on the
        // requesting call (the warm weight cache, #956, makes repeats cheap);
        // set false to keep the single-model contract and fail fast instead.
        bool model_swap = true;
        // How long a swap waits for in-flight generations to finish before
        // giving up and keeping the current model (503, nothing torn down).
        int model_swap_drain_ms = 60000;
        // Device budget (MiB) for recurrent-state snapshots — what makes
        // prefix caching work on hybrid (SSM/GDN) models: KV blocks alone
        // cannot skip prefill there, the recurrent state at the skip boundary
        // must be restored too. One snapshot = one per-sequence state slab
        // (~64 MiB for Qwen3.6-35B), saved per prefill, LRU-evicted. Buffers
        // are pre-allocated at engine init (free VRAM is ~0 at serving time
        // by design) and accounted in the expert-offload reserve. imp-cli
        // --bench pins this to 0 (baseline semantics unchanged).
        // 0 disables snapshots AND hybrid prefix caching (dense unaffected).
        int recurrent_snapshot_mb = 256;
        // Green Contexts / prefill-decode overlap streams in the server engine.
        // OFF by default (suspected memSyncDomain race on sm_120 fallback
        // streams — gemma-3-12b IMA); opt in via [server] green_contexts = true.
        bool green_contexts = false;
    } server;

    struct WarmCache {
        // On-disk warm weight cache: at the first fully-cold load, persist the
        // TRANSFORMED weight uploads (BF16->FP16 conversions, dequants, split
        // layouts) next to the model; later boots restore them instead of
        // re-converting. Raw-from-source uploads are never stored (the model
        // file already holds those bytes), so the cache is near-zero for
        // raw-served GGUF quants / NVFP4-prequant SafeTensors and ~model-size
        // only for BF16-dense checkpoints. Guarded by a format version and a
        // model-content fingerprint; any mismatch = normal cold load.
        bool enabled = true;
        // Where to store cache files. Empty (default) = next to the model
        // ("<file>.impwcache" / "<dir>/.imp_warm_cache"). Point this at a
        // writable volume when the model directory is read-only for the
        // serving user (the prebuilt container runs as uid 1001): files are
        // then named "<model-basename>-<path-hash>.impwcache" inside it.
        std::string dir;
    } warm_cache;

    struct Suspend {
        // Suspend-to-RAM (/admin/suspend): after model/engine teardown, also
        // cudaDeviceReset() so the CUDA primary context (+ module code,
        // ~300-600 MiB) is released and the GPU reads ~0 MiB for this process.
        // Escape hatch: set false if a foreign library holds CUDA state the
        // reset would orphan (imp itself re-arms everything at the next init).
        bool device_reset = true;
        // Host RAM the snapshot must leave free (MemAvailable gate) on top of
        // the snapshot bytes themselves.
        int host_ram_headroom_mb = 2048;
    } suspend;

    struct Bench {
        bool generate = false;
    } bench;

    struct Paths {
        std::string mmproj;
    } paths;

    // n-gram (prompt-lookup) speculative decoding. Drafts come from suffix
    // matches against the request's own prompt+output tokens — no draft
    // model, no MTP head. Greedy-only Phase 1: the verify step replays the
    // draft as a teacher-forced continuation chunk and accepts the longest
    // argmax-matching prefix, so output is token-identical to plain greedy
    // decode. The verify loop runs eager (no async conditional graph loop);
    // burst_rearm + miss_burst keep draft-miss fragmentation ~free, so the old
    // tg128 -15% draft-poor downside no longer reproduces (-0.2%/-0.9% on
    // dense Q8/NVFP4, 2026-06-16) — hence default-ON. spec_ngram_gates_ok_
    // confines engagement to batch-1 / greedy / no-penalty-window / no-json /
    // no-logprobs / non-recurrent requests (MoE additionally requires
    // native-NVFP4 experts, see `moe` below); everything else falls back
    // cleanly, so default-on is a no-op for sampled chat, tool/JSON calls,
    // concurrent batches, and GGUF-MoE (which the async loop carries).
    cfg::Speculative speculative;

    // Constrained decoding (json_mode / json_schema).
    struct Constrained {
        // Jump-ahead over schema-forced spans (#844): when the schema FSM
        // forces the next CHARACTERS (skeleton keys/punctuation — the text
        // is forced even though its tokenization is not), one speculative
        // chunk forward drafts the canonical tokenization and materializes
        // per-position logits rows; subsequent tokens are then sampled from
        // those rows without running forwards. Exact for greedy AND
        // sampling — each row is the true logits given the accepted prefix;
        // a token that diverges from the draft re-enters normal pipelining
        // (one wasted chunk forward, nothing else). OPT-IN: measured net
        // -3-5% on Qwen3-8B (Q8 + NVFP4, 2026-07-03) — the model picks
        // context-dependent tokenization splits the canonical draft misses,
        // so wasted chunks outweigh consumed rows. Also note the chunk path
        // is not bit-identical to per-token decode (prefill vs decode
        // kernels), so free-text AFTER a consumed span can diverge from a
        // jump-off run (same cross-path property as spec-ngram verify).
        bool jump_ahead = false;
        // Minimum draft length (tokens) worth the speculative chunk;
        // shorter forced spans stay on the per-token pipeline.
        int jump_min_run = 4;
    } constrained;

    cfg::FFN ffn;

    cfg::Diagnostics diagnostics;

    // ----- Activation calibration (offline quantizer input) -----
    //
    // Collect per-input-channel activation magnitudes during a forward pass and
    // write them for imp-quantize's AWQ scale search. Not an inference feature:
    // a calibration run is a prefill over a corpus whose only output is this
    // file. Turning it on also turns CUDA graphs off, because the collector
    // allocates a per-weight accumulator lazily and a capture forbids that.
    struct Calibration {
        bool enabled = false;
        // Where imp_calibration_write() puts the file. Empty means the caller
        // supplies the path.
        std::string out_path;
    } calibration;

    // ----- Loading -----

    // Find a config file in the search-path order documented above.
    // Returns empty string if no file is found.
    static std::string find_default_path();

    // Load from disk; returns true on success. On parse error, the struct
    // is left at its default state and an error is logged.
    bool load_from_file(const std::string& path);

    // Apply key=value strings (e.g. "kv_cache.dtype=fp8"). Each entry is
    // parsed via dotted-section lookup. Returns the entries that bound to
    // nothing — a `--set` naming a key this build does not have is a typo,
    // and a measurement flag that silently does nothing is worse than one
    // that stops. (An unknown key in imp.conf stays a warning: a config file
    // may legitimately outlive the build that understood it.)
    [[nodiscard]] std::vector<std::string> apply_overrides(const std::vector<std::string>& kvs);

    // Convenience: locate + load + apply overrides + log a one-line summary.
    // Pass empty path to use the search-path default. Pass `rejected` to take
    // the unbound overrides and decide yourself (both tool mains exit on them);
    // leave it null and they are only logged.
    static RuntimeConfig load(const std::string& explicit_path, const std::vector<std::string>& overrides,
                              std::vector<std::string>* rejected = nullptr);
};

// ---- Pending-config handoff (tool-main → Engine) -----------------------
//
// The C API constructs Engine inside src/api/imp_api.cpp. Tool mains
// (imp-cli, imp-server) load a RuntimeConfig from imp.conf + CLI
// overrides at startup and need to hand that to Engine::init without
// passing it through the ABI-stable ImpConfig C struct.
//
// Workflow: tool main calls set_pending_runtime_config(loaded_cfg) once,
// then later imp_context_create() pulls it via take_pending_runtime_config()
// and passes to Engine::init. This replaces the former
// RuntimeConfig::install() process-wide singleton (Phase 5 Track D
// follow-up, 2026-05-20) — the lifetime is now bounded to a single
// Engine construction; there is no per-call accessor.
void set_pending_runtime_config(RuntimeConfig cfg);
RuntimeConfig take_pending_runtime_config();

}  // namespace imp
