#include "lm/builder/initial_probabilities.hh"

#include "lm/builder/discount.hh"
#include "lm/builder/hash_gamma.hh"
#include "lm/builder/payload.hh"
#include "lm/common/special.hh"
#include "lm/common/ngram_stream.hh"
#include "util/murmur_hash.hh"
#include "util/file.hh"
#include "util/stream/chain.hh"
#include "util/stream/io.hh"
#include "util/stream/stream.hh"

#include <vector>

namespace lm { namespace builder {

namespace {
struct BufferEntry {
  // Gamma from page 20 of Chen and Goodman.
  float gamma;
  // \sum_w a(c w) for all w.
  float denominator;
};

struct HashBufferEntry : public BufferEntry {
  // Hash value of ngram. Used to join contexts with backoffs.
  uint64_t hash_value;
};

// Reads all entries in order like NGramStream does.
// But deletes any entries that have CutoffCount below or equal to pruning
// threshold.
class PruneNGramStream {
  public:
    PruneNGramStream(const util::stream::ChainPosition &position, const SpecialVocab &specials) :
      current_(NULL, NGram<BuildingPayload>::OrderFromSize(position.GetChain().EntrySize())),
      dest_(NULL, NGram<BuildingPayload>::OrderFromSize(position.GetChain().EntrySize())),
      currentCount_(0),
      block_(position),
      specials_(specials)
    {
      StartBlock();
    }

    NGram<BuildingPayload> &operator*() { return current_; }
    NGram<BuildingPayload> *operator->() { return &current_; }

    operator bool() const {
      return block_;
    }

    PruneNGramStream &operator++() {
      assert(block_);
      if(UTIL_UNLIKELY(current_.Order() == 1 && specials_.IsSpecial(*current_.begin())))
        dest_.NextInMemory();
      else if(currentCount_ > 0) {
        if(dest_.Base() < current_.Base()) {
          memcpy(dest_.Base(), current_.Base(), current_.TotalSize());
        }
        dest_.NextInMemory();
      }

      current_.NextInMemory();

      uint8_t *block_base = static_cast<uint8_t*>(block_->Get());
      if (current_.Base() == block_base + block_->ValidSize()) {
        block_->SetValidSize(dest_.Base() - block_base);
        ++block_;
        StartBlock();
        if (block_) {
          currentCount_ = current_.Value().CutoffCount();
        }
      } else {
        currentCount_ = current_.Value().CutoffCount();
      }

      return *this;
    }

  private:
    void StartBlock() {
      for (; ; ++block_) {
        if (!block_) return;
        if (block_->ValidSize()) break;
      }
      current_.ReBase(block_->Get());
      currentCount_ = current_.Value().CutoffCount();

      dest_.ReBase(block_->Get());
    }

    NGram<BuildingPayload> current_; // input iterator
    NGram<BuildingPayload> dest_;    // output iterator

    uint64_t currentCount_;

    util::stream::Link block_;

    const SpecialVocab specials_;
};

// Extract an array of HashedGamma from an array of BufferEntry.
class OnlyGamma {
  public:
    explicit OnlyGamma(bool pruning) : pruning_(pruning) {}

    void Run(const util::stream::ChainPosition &position) {
      for (util::stream::Link block_it(position); block_it; ++block_it) {
        if(pruning_) {
          const HashBufferEntry *in = static_cast<const HashBufferEntry*>(block_it->Get());
          const HashBufferEntry *end = static_cast<const HashBufferEntry*>(block_it->ValidEnd());

          // Just make it point to the beginning of the stream so it can be overwritten
          // With HashGamma values. Do not attempt to interpret the values until set below.
          HashGamma *out = static_cast<HashGamma*>(block_it->Get());
          for (; in < end; out += 1, in += 1) {
            // buffering, otherwise might overwrite values too early
            float gamma_buf = in->gamma;
            uint64_t hash_buf = in->hash_value;

            out->gamma = gamma_buf;
            out->hash_value = hash_buf;
          }
          block_it->SetValidSize((block_it->ValidSize() * sizeof(HashGamma)) / sizeof(HashBufferEntry));
        }
        else {
          float *out = static_cast<float*>(block_it->Get());
          const float *in = out;
          const float *end = static_cast<const float*>(block_it->ValidEnd());
          for (out += 1, in += 2; in < end; out += 1, in += 2) {
            *out = *in;
          }
          block_it->SetValidSize(block_it->ValidSize() / 2);
        }
      }
    }

    private:
      bool pruning_;
};

class AddRight {
  public:
    AddRight(const Discount &discount, const util::stream::ChainPosition &input, bool pruning)
      : discount_(discount), input_(input), pruning_(pruning) {}

    void Run(const util::stream::ChainPosition &output) {
      NGramStream<BuildingPayload> in(input_);
      util::stream::Stream out(output);

      std::vector<WordIndex> previous(in->Order() - 1);
      // Silly windows requires this workaround to just get an invalid pointer when empty.
      void *const previous_raw = previous.empty() ? NULL : static_cast<void*>(&previous[0]);
      const std::size_t size = sizeof(WordIndex) * previous.size();

      for(; in; ++out) {
        memcpy(previous_raw, in->begin(), size);
        uint64_t denominator = 0;
        uint64_t normalizer = 0;

        uint64_t counts[4];
        memset(counts, 0, sizeof(counts));
        do {
          denominator += in->Value().UnmarkedCount();

          // Collect unused probability mass from pruning.
          // Becomes 0 for unpruned ngrams.
          normalizer += in->Value().UnmarkedCount() - in->Value().CutoffCount();

          // Chen&Goodman do not mention counting based on cutoffs, but
          // backoff becomes larger than 1 otherwise, so probably needs
          // to count cutoffs. Counts normally without pruning.
          if(in->Value().CutoffCount() > 0)
            ++counts[std::min(in->Value().CutoffCount(), static_cast<uint64_t>(3))];

        } while (++in && !memcmp(previous_raw, in->begin(), size));

        BufferEntry &entry = *reinterpret_cast<BufferEntry*>(out.Get());
        entry.denominator = static_cast<float>(denominator);
        entry.gamma = 0.0;
        for (unsigned i = 1; i <= 3; ++i) {
          entry.gamma += discount_.Get(i) * static_cast<float>(counts[i]);
        }

        // Makes model sum to 1 with pruning (I hope).
        entry.gamma += normalizer;

        entry.gamma /= entry.denominator;

        if(pruning_) {
          // If pruning is enabled the stream actually contains HashBufferEntry, see InitialProbabilities(...),
          // so add a hash value that identifies the current ngram.
          static_cast<HashBufferEntry*>(&entry)->hash_value = util::MurmurHashNative(previous_raw, size);
        }
      }
      out.Poison();
    }

  private:
    const Discount &discount_;
    const util::stream::ChainPosition input_;
    bool pruning_;
};

class MergeRight {
  public:
    MergeRight(bool interpolate_unigrams, const util::stream::ChainPosition &from_adder, const Discount &discount, const SpecialVocab &specials)
      : interpolate_unigrams_(interpolate_unigrams), from_adder_(from_adder), discount_(discount), specials_(specials) {}

    // calculate the initial probability of each n-gram (before order-interpolation)
    // Run() gets invoked once for each order
    void Run(const util::stream::ChainPosition &primary) {
      util::stream::Stream summed(from_adder_);

      PruneNGramStream grams(primary, specials_);

      // Without interpolation, the interpolation weight goes to <unk>.
      if (grams->Order() == 1) {
        BufferEntry sums(*static_cast<const BufferEntry*>(summed.Get()));
        // Special case for <unk>
        assert(*grams->begin() == kUNK);
        float gamma_assign;
        if (interpolate_unigrams_) {
          // Default: treat <unk> like a zeroton.
          gamma_assign = sums.gamma;
          grams->Value().uninterp.prob = 0.0;
        } else {
          // SRI: give all the interpolation mass to <unk>
          gamma_assign = 0.0;
          grams->Value().uninterp.prob = sums.gamma;
        }
        grams->Value().uninterp.gamma = gamma_assign;

        for (++grams; *grams->begin() != specials_.BOS(); ++grams) {
          grams->Value().uninterp.prob = discount_.Apply(grams->Value().count) / sums.denominator;
          grams->Value().uninterp.gamma = gamma_assign;
        }

        // Special case for <s>: probability 1.0.  This allows <s> to be
        // explicitly scored as part of the sentence without impacting
        // probability and computes q correctly as b(<s>).
        assert(*grams->begin() == specials_.BOS());
        grams->Value().uninterp.prob = 1.0;
        grams->Value().uninterp.gamma = 0.0;

        while (++grams) {
          grams->Value().uninterp.prob = discount_.Apply(grams->Value().count) / sums.denominator;
          grams->Value().uninterp.gamma = gamma_assign;
        }
        ++summed;
        return;
      }

      std::vector<WordIndex> previous(grams->Order() - 1);
      const std::size_t size = sizeof(WordIndex) * previous.size();
      for (; grams; ++summed) {
        memcpy(&previous[0], grams->begin(), size);
        const BufferEntry &sums = *static_cast<const BufferEntry*>(summed.Get());

        do {
          BuildingPayload &pay = grams->Value();
          pay.uninterp.prob = discount_.Apply(grams->Value().UnmarkedCount()) / sums.denominator;
          pay.uninterp.gamma = sums.gamma;
        } while (++grams && !memcmp(&previous[0], grams->begin(), size));
      }
    }

  private:
    bool interpolate_unigrams_;
    util::stream::ChainPosition from_adder_;
    Discount discount_;
    const SpecialVocab specials_;
};

} // namespace

void InitialProbabilities(
    const InitialProbabilitiesConfig &config,
    const std::vector<Discount> &discounts,
    util::stream::Chains &primary,
    util::stream::Chains &second_in,
    util::stream::Chains &gamma_out,
    const std::vector<uint64_t> &prune_thresholds,
    bool prune_vocab,
    const SpecialVocab &specials) {
  for (size_t i = 0; i < primary.size(); ++i) {
    util::stream::ChainConfig gamma_config = config.adder_out;
    if(prune_vocab || prune_thresholds[i] > 0)
      gamma_config.entry_size = sizeof(HashBufferEntry);
    else
      gamma_config.entry_size = sizeof(BufferEntry);

    util::stream::ChainPosition second(second_in[i].Add());
    second_in[i] >> util::stream::kRecycle;
    gamma_out.push_back(gamma_config);
    gamma_out[i] >> AddRight(discounts[i], second, prune_vocab || prune_thresholds[i] > 0);

    primary[i] >> MergeRight(config.interpolate_unigrams, gamma_out[i].Add(), discounts[i], specials);

    // Don't bother with the OnlyGamma thread for something to discard.
    if (i) gamma_out[i] >> OnlyGamma(prune_vocab || prune_thresholds[i] > 0);
  }
}

}} // namespaces
