// The batched MTP head pass's tables: validated, packed into pinned staging and uploaded with one H2D
// (mtp_head_pass_upload). Host code only; the feed kernels that read them live in mtp_forward.cu.

#include "compute/mtp_forward.h"
#include "compute/mtp_forward_internal.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace imp {

bool mtp_pass_staging_create(MtpDraftWorkspace& ws) {
    ws.h_pass_tables = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                             kMtpPassTableInts * sizeof(int32_t));
    return !ws.h_pass_tables.empty() && ws.pass_tables_copied.create();
}

bool mtp_feed_tables_valid(const MtpDraftWorkspace& ws, const int* h_slots, const int* h_pos, int n_rows,
                           int* max_pos) {
    *max_pos = 0;
    for (int r = 0; r < n_rows; ++r) {
        if (h_slots[r] < 0 || h_slots[r] >= ws.n_kv_slots || h_pos[r] < 0 || h_pos[r] >= ws.max_seq_len)
            return false;
        *max_pos = std::max(*max_pos, h_pos[r]);
    }
    return true;
}

bool mtp_head_pass_upload(MtpDraftWorkspace& ws, const int32_t* toks, const int* src, const int* slots,
                          const int* pos, int n, const int* last, const int* chain_slots,
                          const int* chain_pos, int m, cudaStream_t stream, MtpPassTables& out) {
    if (ws.d_pass_tables == nullptr || ws.h_pass_tables.empty() || ws.pass_tables_copied.get() == nullptr ||
        n <= 0 || m <= 0 || m > n || n > ws.feed_rows_cap || (4 * n) + (3 * m) > kMtpPassTableInts)
        return false;
    int max_pos = 0;
    // The previous pass's copy must have left the staging before it is rewritten.
    if (!mtp_feed_tables_valid(ws, slots, pos, n, &max_pos) ||
        cudaEventSynchronize(ws.pass_tables_copied) != cudaSuccess)
        return false;
    int* h = ws.h_pass_tables.as<int>();
    const int* parts[] = {toks, src, slots, pos, last, chain_slots, chain_pos};
    const int lens[] = {n, n, n, n, m, m, m};
    const int* dev[7] = {};
    int off = 0;
    for (int i = 0; i < 7; ++i) {
        std::copy_n(parts[i], lens[i], h + off);
        dev[i] = ws.d_pass_tables + off;
        off += lens[i];
    }
    if (cudaMemcpyAsync(ws.d_pass_tables, h, static_cast<size_t>(off) * sizeof(int), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        !ws.pass_tables_copied.record(stream))
        return false;
    out = MtpPassTables{};
    out.tokens = dev[0];
    out.src = dev[1];
    out.slots = dev[2];
    out.pos = dev[3];
    out.last = dev[4];
    out.chain_slots = dev[5];
    out.chain_pos = dev[6];
    out.n = n;
    out.m = m;
    out.max_pos = max_pos;
    out.chain_max_pos = *std::max_element(chain_pos, chain_pos + m);
    return true;
}

}  // namespace imp
