// Byte layout of the serving metadata pool (engine_kv_cache_init.cpp): the
// per-chunk / per-request int arrays every forward path uploads, one fixed
// region per path so none of them allocates while serving (invariant I2,
// docs/internals/MEMORY.md A3.2). Pure sizes-in, offsets-out, so the CPU lane
// can check what the engine relies on: every region 256-byte aligned, disjoint,
// inside `total`.
#pragma once

#include <cstddef>

namespace imp {

struct ServingMetadataLayout {
    // Serial prefill (engine_prefill.cpp): one chunk, one block table.
    size_t pf_tok = 0, pf_pos = 0, pf_bt = 0, pf_bt_swa = 0, pf_ctx = 0;
    // Ragged prefill (engine_prefill_ragged.cpp): rows in total, seq_cap tables.
    size_t rg_tok = 0, rg_pos = 0, rg_bt = 0, rg_ctx = 0, rg_soff = 0, rg_slots = 0;
    // Single-sequence graph loops (engine_graph_decode.cpp): one table each.
    size_t gl_bt = 0, gl_bt_swa = 0, agl_bt = 0, agl_bt_swa = 0;
    // Constrained pipeline: table, sampled token + sampler scratch, pos, ctx.
    size_t cp_bt = 0, cp_token = 0, cp_pos = 0, cp_ctx = 0;
    size_t total = 0;

    static constexpr size_t kAlign = 256;  // the sampler scratch is read as wider types

    // rows: token rows per prefill chunk / ragged wave (max_seq_len).
    // seq_cap: ragged members per wave (max_batch_size).
    // bt_cap: ints per block table. swa: SWA-group mirrors present.
    // sample_scratch_bytes: SAMPLE_SCRATCH_BYTES.
    static ServingMetadataLayout compute(int rows, int seq_cap, int bt_cap, bool swa,
                                         size_t sample_scratch_bytes) {
        ServingMetadataLayout l;
        size_t off = 0;
        auto region = [&off](size_t bytes) {
            const size_t at = off;
            off += (bytes + kAlign - 1) & ~(kAlign - 1);
            return at;
        };
        const size_t r = static_cast<size_t>(rows > 0 ? rows : 0);
        const size_t s = static_cast<size_t>(seq_cap > 0 ? seq_cap : 1);
        const size_t bt_bytes = static_cast<size_t>(bt_cap > 0 ? bt_cap : 0) * sizeof(int);
        const size_t swa_bytes = swa ? bt_bytes : 0;
        l.pf_tok = region(r * sizeof(int));
        l.pf_pos = region(r * sizeof(int));
        l.pf_bt = region(bt_bytes);
        l.pf_bt_swa = region(swa_bytes);
        l.pf_ctx = region(sizeof(int));
        l.rg_tok = region(r * sizeof(int));
        l.rg_pos = region(r * sizeof(int));
        l.rg_bt = region(s * bt_bytes);
        l.rg_ctx = region(s * sizeof(int));
        l.rg_soff = region((s + 1) * sizeof(int));
        l.rg_slots = region(s * sizeof(int));
        l.gl_bt = region(bt_bytes);
        l.gl_bt_swa = region(swa_bytes);
        l.agl_bt = region(bt_bytes);
        l.agl_bt_swa = region(swa_bytes);
        l.cp_bt = region(bt_bytes);
        l.cp_token = region(sample_scratch_bytes);
        l.cp_pos = region(sizeof(int));
        l.cp_ctx = region(sizeof(int));
        l.total = off;
        return l;
    }
};

}  // namespace imp
