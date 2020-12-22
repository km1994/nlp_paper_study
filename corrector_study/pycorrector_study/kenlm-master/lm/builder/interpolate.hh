#ifndef LM_BUILDER_INTERPOLATE_H
#define LM_BUILDER_INTERPOLATE_H

#include "lm/common/special.hh"
#include "lm/word_index.hh"
#include "util/stream/multi_stream.hh"

#include <vector>

#include <stdint.h>

namespace lm { namespace builder {

/* Interpolate step.
 * Input: suffix sorted n-grams with (p_uninterpolated, gamma) from
 * InitialProbabilities.
 * Output: suffix sorted n-grams with complete probability
 */
class Interpolate {
  public:
    // Normally vocab_size is the unigram count-1 (since p(<s>) = 0) but might
    // be larger when the user specifies a consistent vocabulary size.
    explicit Interpolate(uint64_t vocab_size, const util::stream::ChainPositions &backoffs, const std::vector<uint64_t> &prune_thresholds, bool prune_vocab, bool output_q, const SpecialVocab &specials);

    void Run(const util::stream::ChainPositions &positions);

  private:
    float uniform_prob_;
    util::stream::ChainPositions backoffs_;
    const std::vector<uint64_t> prune_thresholds_;
    bool prune_vocab_;
    bool output_q_;
    const SpecialVocab specials_;
};

}} // namespaces
#endif // LM_BUILDER_INTERPOLATE_H
