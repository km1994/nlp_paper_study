#ifndef LM_BUILDER_ADJUST_COUNTS_H
#define LM_BUILDER_ADJUST_COUNTS_H

#include "lm/builder/discount.hh"
#include "lm/lm_exception.hh"
#include "util/exception.hh"

#include <vector>

#include <stdint.h>

namespace util { namespace stream { class ChainPositions; } }

namespace lm {
namespace builder {

class BadDiscountException : public util::Exception {
  public:
    BadDiscountException() throw();
    ~BadDiscountException() throw();
};

struct DiscountConfig {
  // Overrides discounts for orders [1,discount_override.size()].
  std::vector<Discount> overwrite;
  // If discounting fails for an order, copy them from here.
  Discount fallback;
  // What to do when discounts are out of range or would trigger divison by
  // zero.  It it does something other than THROW_UP, use fallback_discount.
  WarningAction bad_action;
};

/* Compute adjusted counts.
 * Input: unique suffix sorted N-grams (and just the N-grams) with raw counts.
 * Output: [1,N]-grams with adjusted counts.
 * [1,N)-grams are in suffix order
 * N-grams are in undefined order (they're going to be sorted anyway).
 */
class AdjustCounts {
  public:
    // counts: output
    // counts_pruned: output
    // discounts: mostly output.  If the input already has entries, they will be kept.
    // prune_thresholds: input.  n-grams with normal (not adjusted) count below this will be pruned.
    AdjustCounts(
        const std::vector<uint64_t> &prune_thresholds,
        std::vector<uint64_t> &counts,
        std::vector<uint64_t> &counts_pruned,
        const std::vector<bool> &prune_words,
        const DiscountConfig &discount_config,
        std::vector<Discount> &discounts)
      : prune_thresholds_(prune_thresholds), counts_(counts), counts_pruned_(counts_pruned),
        prune_words_(prune_words), discount_config_(discount_config), discounts_(discounts)
    {}

    void Run(const util::stream::ChainPositions &positions);

  private:
    const std::vector<uint64_t> &prune_thresholds_;
    std::vector<uint64_t> &counts_;
    std::vector<uint64_t> &counts_pruned_;
    const std::vector<bool> &prune_words_;

    DiscountConfig discount_config_;
    std::vector<Discount> &discounts_;
};

} // namespace builder
} // namespace lm

#endif // LM_BUILDER_ADJUST_COUNTS_H

