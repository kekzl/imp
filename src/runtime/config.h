#pragma once

#include "core/dispatch_policy.h"  // the nine sections exec/ reads (F-10)

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
        // Pre-captures the per-batch-size decode graph pool at init (server
        // shapes only; no-op unless max_batch_size > 1 and graphs are on),
        // walking one staggered dummy batch from max_batch_size down to 1 so
        // no batch size pays its capture cost during live traffic. One
        // anchor request carries a ~1000-token prompt so captures bake the
        // 1024 context bucket, not the 64 one. False trades this for init time.
        bool graph_prewarm = true;
        // NOTE: full determinism also needs stable cuBLAS algo selection
        // across processes, see runtime.deterministic_gemm.
        int max_seq_len = 0;               // 0 = use model default
        // Hard VRAM budget for this process, MiB (0 = uncapped). Every sizing
        // decision sees a virtual GPU of this size, so processes can share a
        // card; leave ~1 GiB real headroom (cuBLAS internals sit outside it).
        // Overridden by --vram-budget / ImpConfig.vram_budget_mb.
        int vram_budget_mb = 0;
        bool no_pdl = false;
        bool debug_raw = false;        // raw stream debug
        bool no_vision_graph = false;  // disable SigLIP graph capture
        // Qwen3-VL patch budget, in 16x16 patches: ceiling that pulls the
        // preprocessor's max_pixels down and sizes every encoder workspace
        // (not a refusal check). 4096 = 1024x1024.
        int vision_max_patches = 4096;
        // cudaStreamCaptureMode: "global" | "relaxed" (default) | "thread_local".
        // "relaxed" avoids a suspected CUTLASS grouped-GEMM deadlock under
        // prefill capture (prefill_graph_blockers_2026_05_14.md, Blocker B);
        // it is a strict superset of "global" so the decode path is
        // unaffected. Regression: try "thread_local" before reverting to "global".
        std::string graph_capture_mode = "relaxed";
        // Capture prefill into a CUDA graph in addition to decode. Default
        // ON: safe now that graph_capture_mode defaults to "relaxed"
        // (prefill_graph_blockers_2026_05_14.md, Blocker B). Opt out per
        // model via `--set runtime.prefill_graph=false` if one regresses.
        bool prefill_graph = true;
        // 0 = auto: engine sizes the decode batch from the model's weight
        // footprint (a >20 GiB MoE auto-picks 1). Positive value forces it.
        int max_batch_size = 0;
        // Token burst size for the autonomous decode graph loop on a
        // non-streaming, non-speculative request: bounds it so a client
        // cancel can be polled between bursts instead of running unbounded
        // to max_tokens. Output is identical either way (same decode,
        // chunked). Larger = less relaunch overhead, higher cancel latency;
        // <=0 = unbounded. Ignored when `deterministic` is set (needs the
        // unbounded path for bit-reproducible greedy decode).
        int decode_burst = 128;
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
        // rather than mixed workloads; 0 = off.
        int prefill_chunk_decode_cap = 1024;
        // Scales the cap above by W x (requests waiting for prefill) /
        // (requests decoding) once that ratio exceeds 1, clamped to the full
        // chunk (prefill_pacing.h). W = how many waiters one decoder's
        // smoothness is worth, 0 = off. Higher W favors decoder ITL over
        // ingest TTFT (docs/roadmap.md, Server and latency).
        int prefill_cap_fairness = 4;
        // Caps the NUMBER of prefill chunk forwards per engine step while
        // other sequences are DECODING (#1643); the size cap above bounds
        // only ONE chunk. 0 (default) = unbounded, since the token-charged
        // budget in engine_scheduler.cpp's prefill loop now does the
        // protective job instead: one full-sized chunk plus several small
        // prompts (each charged a 256-token launch-cost floor) per step.
        // Rotates the starting index so no ingest is starved. The cap does
        // not apply when nobody is decoding. Positive = hard forward-count
        // cap on top of the token budget.
        int prefill_batch_decode_cap = 0;
        // Mixed prefill+decode step: while prompts prefill, decoding
        // requests ride the ragged prefill forward as one-row members
        // (sampled by the decode sampler) instead of a separate decode
        // forward. Dense models only; a decoder the ragged path does not
        // admit (vision, logprobs, constraints, MTP) keeps the whole batch
        // on the separate decode step (docs/roadmap.md, Server and latency).
        bool prefill_mixed_decode = true;
        // Tokens of `max_tokens` reserved for the ANSWER on a reasoning
        // model: the engine injects `</think>` once reasoning reaches
        // max_tokens - max(think_answer_reserve, max_tokens/4), or the
        // `think_budget` fraction, whichever is LATER. Below 0 reads as 0
        // (no floor: the fraction alone decides). Raise for models with long
        // structured answers, lower for a short-verdict agent loop.
        int think_answer_reserve = 256;
        // Hybrid (SSM/GDN) decode time-slice length in tokens: the recurrent
        // scan is single-sequence, so the engine round-robins DECODING
        // requests every this many tokens. Rotation re-captures the decode
        // graph for the new sequence's state slot (~10-20 ms), so smaller =
        // fairer at more capture overhead (128 ~= 1-2% cost). 0 = head-of-line
        // (no rotation).
        int hybrid_decode_quantum = 128;
        // Batches concurrent GDN/SSM sequences into ONE decode step instead
        // of time-slicing (hybrid_decode_quantum above): the recurrent scan
        // is sequential in tokens, not sequences, so sequences parallelize
        // across blockIdx.y. Without this the whole step (FFN and attention
        // GEMMs included) runs at M=1. False falls back to the rotation.
        bool gdn_batched_decode = true;
        // Pipelined batched decode (n>=2, graphs on): step N+1 (device-side
        // token chain + graph replay + sampler enqueue) is enqueued BEFORE
        // step N's tokens are read back, so host bookkeeping/SSE delivery
        // overlaps GPU compute instead of idling it. Engages for
        // async-sampleable rows (greedy/top-k<=128/top-p/min-p/typical-p;
        // penalties via a device-side token history) on non-SSM models.
        // DRY/mirostat/logit-bias/constraints/logprobs and
        // SWA/StreamingLLM/residual-KV rows keep the per-step path.
        bool decode_pipeline = true;
        // Cross-sequence prefill batching (roadmap 0(d)): assembles the
        // prefill chunks of several admitted requests into ONE ragged
        // forward (GEMMs/norms/elementwise over concatenated rows, attention
        // and the GDN conv loop per sequence, GDN scan via a row-offset
        // table), instead of one sequence per forward. Requests with vision,
        // constraints, logprobs, embeddings/rerank scoring, and Mamba2/MLA
        // models, MTP, SWA sizing, residual KV, gdn.fp32_scan/ref_kernel
        // keep the serial path.
        bool prefill_batch = true;

        // Runs the paced prefill chunk CONCURRENT with the in-flight batched
        // decode step (prefill on the low-priority stream, decode in the
        // dual-workspace slot 1), instead of serially between two decode
        // steps. Implies the green-context stream pair (falls back to
        // priority streams + distinct memSyncDomains on sm_120,
        // LIMITATIONS.md). Engages only when every gate holds: decode
        // workspace allocated at max_batch, no MoE, NVFP4-native model (no
        // GGUF decode overlay), decode batch >= 2 this step (spec-verify
        // chunks share the CUTLASS activation scratch and only exist at
        // batch 1). Default OFF: without real SM partitioning on sm_120, two
        // compute-bound streams just displace each other instead of
        // overlapping. Revisit on hardware with real SM partitioning.
        // Design: docs/plans/2026-08-27-prefill-decode-overlap.md
        bool prefill_overlap = false;
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
        // targets. Clamped to [0.05, 0.95] at use.
        float kv_fraction = 0.8f;
        // Free-VRAM reserve floor as % of (budget-visible) total VRAM,
        // floored at 256 MiB absolute. Clamped to [0, 50] at use.
        // Default 10 ~= 3.2 GiB on a 32 GiB card; lower for more KV pool.
        int reserve_floor_pct = 10;
        // Fixed CUDA/cuBLAS/CUTLASS charge on the first forward pass, MiB;
        // not a workspace imp allocates, so the budget pass cannot otherwise
        // see it (docs/internals/MEMORY.md A1.5). ~3900 measured, invariant
        // to batch/context on this target.
        //   -1 = built-in measured constant (default)
        //    0 = charge nothing (driver/toolkit where it does not apply)
        //   >0 = measured value for THIS host, MiB
        // Advisory only: feeds the shadow plan logged next to the live budget.
        int library_reserve_mb = -1;
        // Where to remember what the first forward ACTUALLY claimed, so a
        // later start on the same model charges the measured value instead
        // of the constant (AUDIT B41/B49). Empty = default location, "off" =
        // disabled. A cache miss or unwritable path is never fatal.
        std::string library_reserve_cache;
        // Pinned staging ring for the weight upload (#1653): depth x chunk
        // MiB. Pinning host memory is expensive on WDDM, so the profitable
        // ring shape is a property of the host's pinning cost, not of imp;
        // hence a configurable key rather than a constant.
        int upload_ring_depth = 4;
        int upload_ring_chunk_mib = 4;
    } vram;

    cfg::Attention attention;

    cfg::MoE moe;

    cfg::GDN gdn;

    cfg::GEMM gemm;

    // Model-specific knobs (e.g. Gemma4) live in ModelConfig::Overrides, not
    // here: see src/model/model_config.h.

    cfg::Generation generation;

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
        // Tokens the streaming driver holds back on a TOOL request while it
        // is unknown whether the model is reasoning: a tool request renders
        // a pre-closed think block, and the held prefix keeps a model that
        // reasons anyway from streaming that as the answer
        // (stream_driver.cpp). Paid only on a prose reply
        // (AUDIT_arch_2026 E-4); lower it on a model that never reasons.
        // Plain chat requests hold a fixed 8 tokens and release on the first word.
        int agent_scan_limit = 256;
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
        // A prompt whose block-aligned prefix is shorter than this takes no
        // prefix-cache snapshot (recurrent slab or SWA window) and no
        // prefill split at the boundary, since the split costs every first
        // turn two eager chunks plus a sync. 0 = snapshot every
        // block-aligned prompt.
        int snapshot_min_prompt_tokens = 256;
        // Green Contexts / prefill-decode overlap streams in the server
        // engine. OFF by default (suspected memSyncDomain race on sm_120
        // fallback streams, gemma-3-12b IMA); opt in via [server] green_contexts = true.
                bool green_contexts = false;
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

    struct Bench {
        bool generate = false;
    } bench;

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

    // Constrained decoding (json_mode / json_schema).
    struct Constrained {
        // Jump-ahead over schema-forced spans (#844): when the schema FSM
        // forces the next CHARACTERS, one speculative chunk forward drafts
        // the canonical tokenization and samples subsequent tokens from its
        // logits rows without running forwards. Exact for greedy and
        // sampling; a token that diverges from the draft just re-enters
        // normal pipelining. OPT-IN: measured net negative, since
        // context-dependent tokenization splits the canonical draft misses
        // make wasted chunks outweigh consumed rows. Not bit-identical to
        // per-token decode past a consumed span (same cross-path property
        // as spec-ngram verify).
        bool jump_ahead = false;
        // Minimum draft length (tokens) worth the speculative chunk;
        // shorter forced spans stay on the per-token pipeline.
        int jump_min_run = 4;
    } constrained;

    cfg::FFN ffn;

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
    } calibration;

    // ----- Loading -----

    // Find a config file in the search-path order documented above.
    // Returns empty string if no file is found.
    static std::string find_default_path();

    // Load from disk; returns true on success. On parse error, the struct
    // is left at its default state and an error is logged.
    bool load_from_file(const std::string& path);

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
