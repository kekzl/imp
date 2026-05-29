// Tile MMA-shape probe: fp16 64x64x64 matmul → forces tensor-core MMA so we can
// see whether cuda::tiles lowers to HMMA (sm_120 mma.sync — runnable) or
// tcgen05/UTCMMA (B200-only). AOT-compiled via tileiras (--tilecubin), disassembled.
#include "cuda_tile.h"
#include <cuda_fp16.h>
namespace ct = cuda::tiles;
using namespace ct::literals;

__tile_global__ void mm_h(__half* __restrict__ a, __half* __restrict__ b, __half* __restrict__ c) {
    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{64_ic, 64_ic}}, ct::shape{64_ic, 64_ic}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{64_ic, 64_ic}}, ct::shape{64_ic, 64_ic}};
    auto cView = ct::partition_view{ct::tensor_span{c, ct::extents{64_ic, 64_ic}}, ct::shape{64_ic, 64_ic}};
    auto acc = ct::full<ct::tile<__half, ct::shape<64, 64>>>(__half(0));
    auto aTile = aView.load_masked(0, 0);
    auto bTile = bView.load_masked(0, 0);
    acc = ct::mma(aTile, bTile, acc);
    cView.store_masked(acc, 0, 0);
}
