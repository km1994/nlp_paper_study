#include "lm/builder/interpolate.hh"

#include "lm/builder/hash_gamma.hh"
#include "lm/builder/payload.hh"
#include "lm/common/compare.hh"
#include "lm/common/joint_order.hh"
#include "lm/common/ngram_stream.hh"
#include "lm/lm_exception.hh"
#include "util/fixed_array.hh"
#include "util/murmur_hash.hh"

#include <iostream>
#include <cassert>
#include <cmath>

namespace lm { namespace builder {
namespace {

/* Calculate q, the collapsed probability and backoff, as defined in
 * @inproceedings{Heafield-rest,
 *   author = {Kenneth Heafield and Philipp Koehn and Alon Lavie},
 *   title = {Language Model Rest Costs and Space-Efficient Storage},
 *   year = {2012},
 *   month = {July},
 *   booktitle = {Proceedings of the Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning},
 *   address = {Jeju Island, Korea},
 *   pages = {1169--1178},
 *   url = {http://kheafield.com/professional/edinburgh/rest\_paper.pdf},
 * }
 * This is particularly convenient to calculate during interpolation because
 * the needed backoff terms are already accessed at the same time.
 */
class OutputQ {
  public:
    explicit OutputQ(std::size_t order) : q_delta_(order) {}

    void Gram(unsigned order_minus_1, float full_backoff, ProbBackoff &out) {
      float &q_del = q_delta_[order_minus_1];
      if (order_minus_1) {
        // Divide by context's backoff (which comes in as out.backoff)
        q_del = q_delta_[order_minus_1 - 1] / out.backoff * full_backoff;
      } else {
        q_del = full_backoff;
      }
      out.prob = log10f(out.prob * q_del);
      // TODO: stop wastefully outputting this!
      out.backoff = 0.0;
    }

  private:
    // Product of backoffs in the numerator divided by backoffs in the
    // denominator.  Does not include
    std::vector<float> q_delta_;
};

/* Default: output probability and backoff */
class OutputProbBackoff {
  public:
    explicit OutputProbBackoff(std::size_t /*order*/) {}

    void Gram(unsigned /*order_minus_1*/, float full_backoff, ProbBackoff &out) const {
      // Correcting for numerical precision issues.  Take that IRST.
      out.prob = std::min(0.0f, log10f(out.prob));
      out.backoff = log10f(full_backoff);
    }
};

template <class Output> class Callback {
  public:
    Callback(float uniform_prob, const util::stream::ChainPositions &backoffs, const std::vector<uint64_t> &prune_thresholds, bool prune_vocab, const SpecialVocab &specials)
      : backoffs_(backoffs.size()), probs_(backoffs.size() + 2),
        prune_thresholds_(prune_thresholds),
        prune_vocab_(prune_vocab),
        output_(backoffs.size() + 1 /* order */),
        specials_(specials) {
      probs_[0] = uniform_prob;
      for (std::size_t i = 0; i < backoffs.size(); ++i) {
        backoffs_.push_back(backoffs[i]);
      }
    }

    ~Callback() {
      for (std::size_t i = 0; i < backoffs_.size(); ++i) {
        if(prune_vocab_ || prune_thresholds_[i + 1] > 0)
          while(backoffs_[i])
            ++backoffs_[i];

        if (backoffs_[i]) {
          std::cerr << "Backoffs do not match for order " << (i + 1) << std::endl;
          abort();
        }
      }
    }

    void Enter(unsigned order_minus_1, void *data) {
      NGram<BuildingPayload> gram(data, order_minus_1 + 1);
      BuildingPayload &pay = gram.Value();
      pay.complete.prob = pay.uninterp.prob + pay.uninterp.gamma * probs_[order_minus_1];
      probs_[order_minus_1 + 1] = pay.complete.prob;

      float out_backoff;
      if (order_minus_1 < backoffs_.size() && *(gram.end() - 1) != specials_.UNK() && *(gram.end() - 1) != specials_.EOS() && backoffs_[order_minus_1]) {
        if(prune_vocab_ || prune_thresholds_[order_minus_1 + 1] > 0) {
          //Compute hash value for current context
          uint64_t current_hash = util::MurmurHashNative(gram.begin(), gram.Order() * sizeof(WordIndex));

          const HashGamma *hashed_backoff = static_cast<const HashGamma*>(backoffs_[order_minus_1].Get());
          while(current_hash != hashed_backoff->hash_value && ++backoffs_[order_minus_1])
            hashed_backoff = static_cast<const HashGamma*>(backoffs_[order_minus_1].Get());

          if(current_hash == hashed_backoff->hash_value) {
            out_backoff = hashed_backoff->gamma;
            ++backoffs_[order_minus_1];
          } else {
            // Has been pruned away so it is not a context anymore
            out_backoff = 1.0;
          }
        } else {
          out_backoff = *static_cast<const float*>(backoffs_[order_minus_1].Get());
          ++backoffs_[order_minus_1];
        }
      } else {
        // Not a context.
        out_backoff = 1.0;
      }

      output_.Gram(order_minus_1, out_backoff, pay.complete);
    }

    void Exit(unsigned, void *) const {}

  private:
    util::FixedArray<util::stream::Stream> backoffs_;

    std::vector<float> probs_;
    const std::vector<uint64_t>& prune_thresholds_;
    bool prune_vocab_;

    Output output_;
    const SpecialVocab specials_;
};
} // namespace

Interpolate::Interpolate(uint64_t vocab_size, const util::stream::ChainPositions &backoffs, const std::vector<uint64_t>& prune_thresholds, bool prune_vocab, bool output_q, const SpecialVocab &specials)
  : uniform_prob_(1.0 / static_cast<float>(vocab_size)), // Includes <unk> but excludes <s>.
    backoffs_(backoffs),
    prune_thresholds_(prune_thresholds),
    prune_vocab_(prune_vocab),
    output_q_(output_q),
    specials_(specials) {}

// perform order-wise interpolation
void Interpolate::Run(const util::stream::ChainPositions &positions) {
  assert(positions.size() == backoffs_.size() + 1);
  if (output_q_) {
    typedef Callback<OutputQ> C;
    C callback(uniform_prob_, backoffs_, prune_thresholds_, prune_vocab_, specials_);
    JointOrder<C, SuffixOrder>(positions, callback);
  } else {
    typedef Callback<OutputProbBackoff> C;
    C callback(uniform_prob_, backoffs_, prune_thresholds_, prune_vocab_, specials_);
    JointOrder<C, SuffixOrder>(positions, callback);
  }
}

}} // namespaces
