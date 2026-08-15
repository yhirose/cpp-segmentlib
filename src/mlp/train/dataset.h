#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "mlp/train/example.h"

// Minibatch assembly: shuffle example
// indices, then per batch gather embedding rows and mean-pool them into the
// dense input matrix X the GEMMs consume.

namespace segmentlib::mlp::train {

// One minibatch. Buffers are reused across fill_batch calls.
struct Batch {
    std::uint32_t size = 0;        // B (the last batch may be short)
    std::vector<float> x;          // B × 2w·d, row b = concat(v_0 .. v_{2w-1})
    std::vector<float> f;          // B × Fd dense binary features (empty if Fd=0)
    std::vector<float> targets;    // B

    // The constituent embedding rows behind each window slot, for the
    // backward scatter (II.3): slot (b, j) owns
    // slot_rows[slot_starts[b*2w+j] .. slot_starts[b*2w+j+1]). PAD slots
    // (window positions past the sentence ends) hold the single row kPadRow,
    // so PAD trains like any other row.
    std::vector<std::uint32_t> slot_rows;
    std::vector<std::uint32_t> slot_starts;  // size B*2w + 1
};

class Dataset {
public:
    // `set` must outlive the Dataset. `embed_dim` is d (the gather needs it).
    Dataset(const ExampleSet& set, std::uint16_t embed_dim,
            std::uint32_t batch_size);

    void shuffle(std::mt19937& rng);

    [[nodiscard]] std::size_t num_examples() const noexcept {
        return order_.size();
    }
    [[nodiscard]] std::size_t num_batches() const noexcept {
        return (order_.size() + batch_size_ - 1) / batch_size_;
    }

    // Fills `out` with batch `index`, gathering windows from `embedding`
    // (V×d row-major, the *current* fp32 embedding table).
    void fill_batch(std::size_t index, std::span<const float> embedding,
                    Batch& out) const;

private:
    const ExampleSet* set_;
    std::uint16_t d_;
    std::uint32_t batch_size_;
    std::vector<std::uint32_t> order_;  // example indices, shuffled
};

}  // namespace segmentlib::mlp::train
