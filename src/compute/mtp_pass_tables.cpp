// The batched MTP head pass's tables: validated, packed into pinned staging and uploaded with one H2D
// (mtp_head_pass_upload), and the scan nodes of its captured graph (mtp_scan_nodes_*). Host code only; the
// feed kernels that read them live in mtp_forward.cu.

#include "compute/mtp_forward.h"
#include "compute/mtp_forward_internal.h"

#include "core/logging.h"
#include "memory/mem_account.h"
#include "memory/vram_query.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace imp {

bool mtp_kv_allocate(MtpDraftWorkspace& ws, size_t kv_bytes, int n_kv_slots, Backend* growable) {
    if (growable && n_kv_slots > 1) {
        const size_t slot = kv_bytes / static_cast<size_t>(n_kv_slots);
        AcquireResult k = growable->acquire_growable(kv_bytes, slot, 256, RegionTag::SsmState);
        if (k) {
            AcquireResult v = growable->acquire_growable(kv_bytes, slot, 256, RegionTag::SsmState);
            if (v) {
                ws.k_region = std::move(k.region), ws.v_region = std::move(v.region);
                ws.kv_backend = growable;
                ws.slot_ready.assign(static_cast<size_t>(n_kv_slots), 0);
                ws.slot_ready[0] = 1;
                ws.d_k_cache_base = ws.k_region.base(), ws.d_v_cache_base = ws.v_region.base();
                const size_t committed = ws.k_region.committed() + ws.v_region.committed();
                const size_t counted = 2 * mtp_alloc_footprint(kv_bytes);
                ws.device_bytes += counted;
                ws.kv_pending = counted > committed ? counted - committed : 0;
                MemAccount::instance().note("mtp_workspace", static_cast<std::ptrdiff_t>(committed));
                vram_reserved_uncommitted_add(static_cast<std::ptrdiff_t>(ws.kv_pending));
                return true;
            }
        }
    }
    return mtp_dev_alloc(ws, &ws.d_k_cache_base, kv_bytes) && mtp_dev_alloc(ws, &ws.d_v_cache_base, kv_bytes);
}

void mtp_kv_release(MtpDraftWorkspace& ws) {
    if (!ws.kv_backend)
        return;
    // mtp_workspace_free takes device_bytes off the pool: the part no commit backed goes on it first.
    MemAccount::instance().note("mtp_workspace", static_cast<std::ptrdiff_t>(ws.kv_pending));
    vram_reserved_uncommitted_add(-static_cast<std::ptrdiff_t>(ws.kv_pending));
    ws.k_region.reset(), ws.v_region.reset();
    ws.d_k_cache_base = ws.d_v_cache_base = nullptr;
    ws.kv_backend = nullptr;
    ws.slot_ready.clear();
    ws.kv_pending = 0;
}

namespace {
bool slot_needs_commit(const MtpDraftWorkspace& ws, int slot) {
    return ws.kv_backend && slot >= 0 && static_cast<size_t>(slot) < ws.slot_ready.size() &&
           !ws.slot_ready[static_cast<size_t>(slot)];
}
}  // namespace

bool mtp_commit_slot(MtpDraftWorkspace& ws, int slot) {
    if (!slot_needs_commit(ws, slot))
        return true;
    // Queued launches hold pointers into the regions: let them end before the mapping changes.
    return cudaDeviceSynchronize() == cudaSuccess && mtp_commit_slot_now(ws, slot);
}

bool mtp_commit_slot_now(MtpDraftWorkspace& ws, int slot) {
    if (!slot_needs_commit(ws, slot))
        return true;
    const size_t bytes = ws.kv_slot_elems * 2, off = static_cast<size_t>(slot) * bytes;  // fp16 rows
    const size_t before = ws.k_region.committed() + ws.v_region.committed();
    MemError e = ws.kv_backend->commit_range(ws.k_region, off, bytes);
    if (e == MemError::Ok)
        e = ws.kv_backend->commit_range(ws.v_region, off, bytes);
    const size_t now = ws.k_region.committed() + ws.v_region.committed();
    const size_t added = std::min(now > before ? now - before : 0, ws.kv_pending);
    ws.kv_pending -= added;
    MemAccount::instance().note("mtp_workspace", static_cast<std::ptrdiff_t>(added));
    vram_reserved_uncommitted_add(-static_cast<std::ptrdiff_t>(added));
    if (e != MemError::Ok) {
        IMP_LOG_ERROR("mtp: head KV of slot %d not committed (%s), the slot drafts nothing", slot,
                      mem_error_name(e));
        return false;
    }
    ws.slot_ready[static_cast<size_t>(slot)] = 1;
    return true;
}

bool mtp_commit_slots(MtpDraftWorkspace& ws, const int* h_slots, int n) {
    bool ok = true;
    for (int r = 0; r < n && ok; ++r)
        ok = mtp_commit_slot(ws, h_slots[r]);
    return ok;
}

bool mtp_pass_staging_create(MtpDraftWorkspace& ws) {
    ws.h_pass_tables = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                             kMtpPassTableInts * sizeof(int32_t));
    return !ws.h_pass_tables.empty() && ws.pass_tables_copied.create();
}

bool mtp_feed_tables_valid(const MtpDraftWorkspace& ws, const int* h_slots, const int* h_pos, int n_rows,
                           int* max_pos) {
    *max_pos = 0;
    for (int r = 0; r < n_rows; ++r) {
        if (h_slots[r] < 0 || h_slots[r] >= ws.n_kv_slots || h_pos[r] < 0)
            return false;
        *max_pos = std::max(*max_pos, h_pos[r]);
    }
    return true;
}

bool mtp_head_pass_upload(MtpDraftWorkspace& ws, const int32_t* toks, const int* src, const int* slots,
                          const int* pos, int n, const int* last, const int* chain_slots,
                          const int* chain_pos, int m, cudaStream_t stream, MtpPassTables& out) {
    if (ws.d_pass_tables == nullptr || ws.h_pass_tables.empty() || ws.pass_tables_copied.get() == nullptr ||
        n <= 0 || m <= 0 || m > n || n > ws.feed_rows_cap ||
        (4 * n) + ((3 + kMtpPassChainLevels) * m) > kMtpPassTableInts)
        return false;
    int max_pos = 0;
    // The previous pass's copy must have left the staging before it is rewritten.
    if (!mtp_feed_tables_valid(ws, slots, pos, n, &max_pos) || !mtp_commit_slots(ws, slots, n) ||
        !mtp_commit_slots(ws, chain_slots, m) || cudaEventSynchronize(ws.pass_tables_copied) != cudaSuccess)
        return false;
    int* h = ws.h_pass_tables.as<int>();
    int off = 0;
    auto put = [&](const int* part, int len) {
        std::copy_n(part, len, h + off);
        off += len;
        return static_cast<const int*>(ws.d_pass_tables + off - len);
    };
    out = MtpPassTables{};
    out.tokens = put(toks, n);
    out.src = put(src, n);
    out.slots = put(slots, n);
    out.pos = put(pos, n);
    out.last = put(last, m);
    out.chain_slots = put(chain_slots, m);
    out.chain_rows = ws.d_pass_tables + off;
    for (int i = 0; i < m; ++i)
        h[off++] = i;
    for (int l = 0; l < kMtpPassChainLevels; ++l) {
        out.chain_pos[l] = ws.d_pass_tables + off;
        for (int i = 0; i < m; ++i)
            h[off++] = chain_pos[i] + l;
    }
    if (cudaMemcpyAsync(ws.d_pass_tables, h, static_cast<size_t>(off) * sizeof(int), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        !ws.pass_tables_copied.record(stream)) {
        out = MtpPassTables{};
        return false;
    }
    out.n = n;
    out.m = m;
    out.max_pos = max_pos;
    out.chain_max_pos = *std::max_element(chain_pos, chain_pos + m);
    return true;
}

bool mtp_scan_nodes_find(cudaGraph_t graph, MtpScanNodes& out) {
    out = MtpScanNodes{};
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess)
        return false;
    std::vector<cudaGraphNode_t> nodes(count);
    if (count > 0 && cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess)
        return false;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess)
            return false;
        if (type != cudaGraphNodeTypeKernel)
            continue;
        cudaKernelNodeParams p{};
        // Driver-API nodes (cuBLASLt) report no runtime func and set the last error: not a scan node.
        if (cudaGraphKernelNodeGetParams(node, &p) != cudaSuccess ||
            (p.func != mtp_attn_scan_rows_fn() && p.func != mtp_attn_scan_rows_split_fn() &&
             mtp_attn_scan_group_of(p.func) == 0)) {
            (void)cudaGetLastError();
            continue;
        }
        out.nodes.push_back(node);
        out.params.push_back(p);
    }
    return true;
}

bool mtp_scan_nodes_set_rows(cudaGraphExec_t exec, MtpScanNodes& scan, int rows) {
    for (size_t i = 0; i < scan.nodes.size(); ++i) {
        const auto smem = static_cast<unsigned>(mtp_scan_node_smem(scan.params[i].func, rows));
        if (scan.params[i].sharedMemBytes == smem)
            continue;
        scan.params[i].sharedMemBytes = smem;
        if (cudaGraphExecKernelNodeSetParams(exec, scan.nodes[i], &scan.params[i]) != cudaSuccess)
            return false;
    }
    return true;
}

}  // namespace imp
