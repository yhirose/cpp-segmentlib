#include "mlp/train/dataset.h"

#include <algorithm>
#include <numeric>

namespace segmentlib::mlp::train {

Dataset::Dataset(const ExampleSet& set, std::uint16_t embed_dim,
                 std::uint32_t batch_size)
    : set_(&set), d_(embed_dim), batch_size_(batch_size),
      order_(set.examples.size()) {
    std::iota(order_.begin(), order_.end(), 0u);
}

void Dataset::shuffle(std::mt19937& rng) {
    std::shuffle(order_.begin(), order_.end(), rng);
}

void Dataset::fill_batch(std::size_t index, std::span<const float> embedding,
                         Batch& out) const {
    const std::size_t first = index * batch_size_;
    const std::size_t last =
        std::min(first + batch_size_, order_.size());
    const auto b_count = static_cast<std::uint32_t>(last - first);
    const std::uint32_t slots = 2u * set_->window;
    const std::size_t in_dim = static_cast<std::size_t>(slots) * d_;
    const std::uint32_t fd = set_->dict_feature_count();

    out.size = b_count;
    out.x.assign(static_cast<std::size_t>(b_count) * in_dim, 0.0f);
    out.f.assign(static_cast<std::size_t>(b_count) * fd, 0.0f);
    out.targets.resize(b_count);
    out.slot_rows.clear();
    out.slot_starts.assign(1, 0u);

    for (std::uint32_t b = 0; b < b_count; ++b) {
        const std::uint32_t ex_id = order_[first + b];
        const ExampleSet::Example& ex = set_->examples[ex_id];
        const std::uint32_t m = set_->sentence_egcs(ex.sentence);
        out.targets[b] = ex.label;

        // Window slots j = 0..2w-1 cover sentence-local EGCs
        // boundary - w + 1 + j (design.ja.md 5.4); out-of-range slots are the
        // PAD pseudo-EGC (single constituent row kPadRow).
        float* x_row = out.x.data() + static_cast<std::size_t>(b) * in_dim;
        for (std::uint32_t j = 0; j < slots; ++j) {
            const std::int64_t egc = static_cast<std::int64_t>(ex.boundary) -
                                     set_->window + 1 + j;
            static constexpr std::uint32_t pad[] = {kPadRow};
            std::span<const std::uint32_t> rows = pad;
            if (egc >= 0 && egc < m) {
                rows = set_->egc_rows(ex.sentence,
                                      static_cast<std::uint32_t>(egc));
            }
            out.slot_rows.insert(out.slot_rows.end(), rows.begin(), rows.end());
            out.slot_starts.push_back(
                static_cast<std::uint32_t>(out.slot_rows.size()));

            // v_j = mean of the constituent codepoint embeddings (5.3).
            float* v = x_row + static_cast<std::size_t>(j) * d_;
            const float inv_n = 1.0f / static_cast<float>(rows.size());
            for (const std::uint32_t row : rows) {
                const float* e = embedding.data() +
                                 static_cast<std::size_t>(row) * d_;
                for (std::uint32_t c = 0; c < d_; ++c) {
                    v[c] += e[c] * inv_n;
                }
            }
        }

        for (std::uint32_t o = set_->feat_offsets[ex_id];
             o < set_->feat_offsets[ex_id + 1]; ++o) {
            out.f[static_cast<std::size_t>(b) * fd + set_->feat_indices[o]] =
                1.0f;
        }
    }
}

}  // namespace segmentlib::mlp::train
