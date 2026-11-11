// The batched MTP head pass's tables: validated, packed into pinned staging and uploaded with one H2D
// (mtp_head_pass_upload), and the scan nodes of its captured graph (mtp_scan_nodes_*). Host code only; the
// feed kernels that read them live in mtp_forward.cu.

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
    if (!mtp_feed_tables_valid(ws, slots, pos, n, &max_pos) ||
        cudaEventSynchronize(ws.pass_tables_copied) != cudaSuccess)
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
            (p.func != mtp_attn_scan_rows_fn() && p.func != mtp_attn_scan_rows_split_fn())) {
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
        const int scores = scan.params[i].func == mtp_attn_scan_rows_split_fn() ? mtp_scan_split_chunk(rows)
                                                                                : rows;
        const auto smem = static_cast<unsigned>(scores) * static_cast<unsigned>(sizeof(float));
        if (scan.params[i].sharedMemBytes == smem)
            continue;
        scan.params[i].sharedMemBytes = smem;
        if (cudaGraphExecKernelNodeSetParams(exec, scan.nodes[i], &scan.params[i]) != cudaSuccess)
            return false;
    }
    return true;
}

}  // namespace imp
