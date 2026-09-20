#pragma once
// Qwen4Exp PLE n-gram table: the F8_E4M3 embedding shards stay host-mapped in their SafeTensors
// file (51 GiB on this checkpoint, never resident), the I64 hash buffers come from the main shards.
// hash() reproduces Qwen4ExpTextNGramEmbedding.forward, gather() the embedding lookup.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imp {

// Pure hash, testable without a checkpoint. `ctx` holds ngram_size-1 tokens before tokens[0]
// (EOS-filled at a sequence start); a shift never crosses an EOS. ids_out is [n, n_heads].
struct NGramHashParams {
    const int64_t* multipliers;  // [ngram_size]
    const int64_t* offsets;      // [n_heads]
    const int64_t* vocab_sizes;  // [n_heads]
    int ngram_size;
    int n_heads;  // (ngram_size - 1) * heads_per_ngram
    int eos_token_id;
};
void ngram_hash(const NGramHashParams& p, const int32_t* ctx, const int32_t* tokens, int n, int64_t* ids_out);

class NGramTable {
public:
    ~NGramTable();
    NGramTable(const NGramTable&) = delete;
    NGramTable& operator=(const NGramTable&) = delete;

    // Resolves every `.layers.<layer>.ple.ple_embedding.*` tensor through the sharded index.
    // Null on any shortfall: without its table the model cannot be served.
    static std::unique_ptr<NGramTable> open(const std::string& model_dir, int layer, int eos_token_id);

    int n_heads() const { return n_heads_; }
    int head_dim() const { return head_dim_; }
    int embed_dim() const { return n_heads_ * head_dim_; }
    int ngram_size() const { return ngram_size_; }
    int context_len() const { return ngram_size_ - 1; }
    int64_t total_rows() const { return total_rows_; }
    float scale() const { return scale_; }

    void hash(const int32_t* ctx, const int32_t* tokens, int n, int64_t* ids_out) const;
    // out [n, embed_dim] FP16 bits = F8 row * scale. Prefetches every row (MADV_WILLNEED) first.
    void gather(const int64_t* ids, int n, uint16_t* out) const;

private:
    NGramTable() = default;
    struct Shard {
        int64_t row_start;
        int64_t rows;
        const uint8_t* data;
    };
    void* map_ = nullptr;
    size_t map_size_ = 0;
    std::vector<Shard> shards_;
    int64_t total_rows_ = 0;
    std::vector<int64_t> multipliers_, offsets_, vocab_sizes_;
    float scale_ = 1.0f;
    int n_heads_ = 0;
    int head_dim_ = 0;
    int ngram_size_ = 0;
    int eos_ = -1;
    mutable int64_t n_out_of_range_ = 0;
};

}  // namespace imp
