#include "lm/builder/pipeline.hh"

#include "lm/builder/adjust_counts.hh"
#include "lm/builder/combine_counts.hh"
#include "lm/builder/corpus_count.hh"
#include "lm/builder/hash_gamma.hh"
#include "lm/builder/initial_probabilities.hh"
#include "lm/builder/interpolate.hh"
#include "lm/builder/output.hh"
#include "lm/common/compare.hh"
#include "lm/common/renumber.hh"

#include "lm/sizes.hh"
#include "lm/vocab.hh"

#include "util/exception.hh"
#include "util/file.hh"
#include "util/stream/io.hh"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

namespace lm { namespace builder {

using util::stream::Sorts;

namespace {

void PrintStatistics(const std::vector<uint64_t> &counts, const std::vector<uint64_t> &counts_pruned, const std::vector<Discount> &discounts) {
  std::cerr << "Statistics:\n";
  for (size_t i = 0; i < counts.size(); ++i) {
    std::cerr << (i + 1) << ' ' << counts_pruned[i];
    if(counts[i] != counts_pruned[i])
       std::cerr << "/" << counts[i];

    for (size_t d = 1; d <= 3; ++d)
      std::cerr << " D" << d << (d == 3 ? "+=" : "=") << discounts[i].amount[d];
    std::cerr << '\n';
  }
}

class Master {
  public:
    explicit Master(PipelineConfig &config, unsigned output_steps)
      : config_(config), chains_(config.order), unigrams_(util::MakeTemp(config_.TempPrefix())), steps_(output_steps + 4) {
      config_.minimum_block = std::max(NGram<BuildingPayload>::TotalSize(config_.order), config_.minimum_block);
    }

    const PipelineConfig &Config() const { return config_; }

    util::stream::Chains &MutableChains() { return chains_; }

    template <class T> Master &operator>>(const T &worker) {
      chains_ >> worker;
      return *this;
    }

    // This takes the (partially) sorted ngrams and sets up for adjusted counts.
    void InitForAdjust(util::stream::Sort<SuffixOrder, CombineCounts> &ngrams, WordIndex types, std::size_t subtract_for_numbering) {
      const std::size_t each_order_min = config_.minimum_block * config_.block_count;
      // We know how many unigrams there are.  Don't allocate more than needed to them.
      const std::size_t min_chains = (config_.order - 1) * each_order_min +
        std::min(types * NGram<BuildingPayload>::TotalSize(1), each_order_min);
      // Prevent overflow in subtracting.
      const std::size_t total = std::max<std::size_t>(config_.TotalMemory(), min_chains + subtract_for_numbering + config_.minimum_block);
      // Do merge sort with calculated laziness.
      const std::size_t merge_using = ngrams.Merge(std::min(total - min_chains - subtract_for_numbering, ngrams.DefaultLazy()));

      std::vector<uint64_t> count_bounds(1, types);
      CreateChains(total - merge_using - subtract_for_numbering, count_bounds);
      ngrams.Output(chains_.back(), merge_using);
    }

    // For initial probabilities, but this is generic.
    void SortAndReadTwice(const std::vector<uint64_t> &counts, Sorts<ContextOrder> &sorts, util::stream::Chains &second, util::stream::ChainConfig second_config) {
      bool unigrams_are_sorted = !config_.renumber_vocabulary;
      // Do merge first before allocating chain memory.
      for (std::size_t i = 0; i < config_.order - unigrams_are_sorted; ++i) {
        sorts[i].Merge(0);
      }
      // There's no lazy merge, so just divide memory amongst the chains.
      CreateChains(config_.TotalMemory(), counts);
      chains_.back().ActivateProgress();
      if (unigrams_are_sorted) {
        chains_[0] >> unigrams_.Source();
        second_config.entry_size = NGram<BuildingPayload>::TotalSize(1);
        second.push_back(second_config);
        second.back() >> unigrams_.Source();
      }
      for (std::size_t i = unigrams_are_sorted; i < config_.order; ++i) {
        util::scoped_fd fd(sorts[i - unigrams_are_sorted].StealCompleted());
        chains_[i].SetProgressTarget(util::SizeOrThrow(fd.get()));
        chains_[i] >> util::stream::PRead(util::DupOrThrow(fd.get()), true);
        second_config.entry_size = NGram<BuildingPayload>::TotalSize(i + 1);
        second.push_back(second_config);
        second.back() >> util::stream::PRead(fd.release(), true);
      }
    }

    // There is no sort after this, so go for broke on lazy merging.
    template <class Compare> void MaximumLazyInput(const std::vector<uint64_t> &counts, Sorts<Compare> &sorts) {
      // Determine the minimum we can use for all the chains.
      std::size_t min_chains = 0;
      for (std::size_t i = 0; i < config_.order; ++i) {
        min_chains += std::min(counts[i] * NGram<BuildingPayload>::TotalSize(i + 1), static_cast<uint64_t>(config_.minimum_block));
      }
      std::size_t for_merge = min_chains > config_.TotalMemory() ? 0 : (config_.TotalMemory() - min_chains);
      std::vector<std::size_t> laziness;
      // Prioritize longer n-grams.
      for (util::stream::Sort<SuffixOrder> *i = sorts.end() - 1; i >= sorts.begin(); --i) {
        laziness.push_back(i->Merge(for_merge));
        assert(for_merge >= laziness.back());
        for_merge -= laziness.back();
      }
      std::reverse(laziness.begin(), laziness.end());

      CreateChains(for_merge + min_chains, counts);
      chains_.back().ActivateProgress();
      chains_[0] >> unigrams_.Source();
      for (std::size_t i = 1; i < config_.order; ++i) {
        sorts[i - 1].Output(chains_[i], laziness[i - 1]);
      }
    }

    template <class Compare> void SetupSorts(Sorts<Compare> &sorts, bool exclude_unigrams) {
      sorts.Init(config_.order - exclude_unigrams);
      // Unigrams don't get sorted because their order is always the same.
      if (exclude_unigrams) chains_[0] >> unigrams_.Sink() >> util::stream::kRecycle;
      for (std::size_t i = exclude_unigrams; i < config_.order; ++i) {
        sorts.push_back(chains_[i], config_.sort, Compare(i + 1));
      }
      chains_.Wait(true);
    }

    unsigned int Steps() const { return steps_; }

  private:
    // Create chains, allocating memory to them.  Totally heuristic.  Count
    // bounds are upper bounds on the counts or not present.
    void CreateChains(std::size_t remaining_mem, const std::vector<uint64_t> &count_bounds) {
      std::vector<std::size_t> assignments;
      assignments.reserve(config_.order);
      // Start by assigning maximum memory usage (to be refined later).
      for (std::size_t i = 0; i < count_bounds.size(); ++i) {
        assignments.push_back(static_cast<std::size_t>(std::min(
            static_cast<uint64_t>(remaining_mem),
            count_bounds[i] * static_cast<uint64_t>(NGram<BuildingPayload>::TotalSize(i + 1)))));
      }
      assignments.resize(config_.order, remaining_mem);

      // Now we know how much memory everybody wants.  How much will they get?
      // Proportional to this.
      std::vector<float> portions;
      // Indices of orders that have yet to be assigned.
      std::vector<std::size_t> unassigned;
      for (std::size_t i = 0; i < config_.order; ++i) {
        portions.push_back(static_cast<float>((i+1) * NGram<BuildingPayload>::TotalSize(i+1)));
        unassigned.push_back(i);
      }
      /*If somebody doesn't eat their full dinner, give it to the rest of the
       * family.  Then somebody else might not eat their full dinner etc.  Ends
       * when everybody unassigned is hungry.
       */
      float sum;
      bool found_more;
      std::vector<std::size_t> block_count(config_.order);
      do {
        sum = 0.0;
        for (std::size_t i = 0; i < unassigned.size(); ++i) {
          sum += portions[unassigned[i]];
        }
        found_more = false;
        // If the proportional assignment is more than needed, give it just what it needs.
        for (std::vector<std::size_t>::iterator i = unassigned.begin(); i != unassigned.end();) {
          if (assignments[*i] <= remaining_mem * (portions[*i] / sum)) {
            remaining_mem -= assignments[*i];
            block_count[*i] = 1;
            i = unassigned.erase(i);
            found_more = true;
          } else {
            ++i;
          }
        }
      } while (found_more);
      for (std::vector<std::size_t>::iterator i = unassigned.begin(); i != unassigned.end(); ++i) {
        assignments[*i] = remaining_mem * (portions[*i] / sum);
        block_count[*i] = config_.block_count;
      }
      chains_.clear();
      std::cerr << "Chain sizes:";
      for (std::size_t i = 0; i < config_.order; ++i) {
        // Always have enough for at least one record.
        // This was crashing if e.g. there was no 5-gram.
        assignments[i] = std::max(assignments[i], block_count[i] * NGram<BuildingPayload>::TotalSize(i + 1));
        std::cerr << ' ' << (i+1) << ":" << assignments[i];
        chains_.push_back(util::stream::ChainConfig(NGram<BuildingPayload>::TotalSize(i + 1), block_count[i], assignments[i]));
      }
      std::cerr << std::endl;
    }

    PipelineConfig &config_;

    util::stream::Chains chains_;

    util::stream::FileBuffer unigrams_;

    const unsigned int steps_;
};

util::stream::Sort<SuffixOrder, CombineCounts> *CountText(int text_file /* input */, int vocab_file /* output */, Master &master, uint64_t &token_count, WordIndex &type_count, std::string &text_file_name, std::vector<bool> &prune_words) {
  const PipelineConfig &config = master.Config();
  std::cerr << "=== 1/" << master.Steps() << " Counting and sorting n-grams ===" << std::endl;

  const std::size_t vocab_usage = CorpusCount::VocabUsage(config.vocab_estimate);
  UTIL_THROW_IF(config.TotalMemory() < vocab_usage, util::Exception, "Vocab hash size estimate " << vocab_usage << " exceeds total memory " << config.TotalMemory());
  std::size_t memory_for_chain =
    // This much memory to work with after vocab hash table.
    static_cast<float>(config.TotalMemory() - vocab_usage) /
    // Solve for block size including the dedupe multiplier for one block.
    (static_cast<float>(config.block_count) + CorpusCount::DedupeMultiplier(config.order)) *
    // Chain likes memory expressed in terms of total memory.
    static_cast<float>(config.block_count);
  util::stream::Chain chain(util::stream::ChainConfig(NGram<BuildingPayload>::TotalSize(config.order), config.block_count, memory_for_chain));

  type_count = config.vocab_estimate;
  util::FilePiece text(text_file, NULL, &std::cerr);
  text_file_name = text.FileName();
  CorpusCount counter(text, vocab_file, true, token_count, type_count, prune_words, config.prune_vocab_file, chain.BlockSize() / chain.EntrySize(), config.disallowed_symbol_action);
  chain >> boost::ref(counter);

  util::scoped_ptr<util::stream::Sort<SuffixOrder, CombineCounts> > sorter(new util::stream::Sort<SuffixOrder, CombineCounts>(chain, config.sort, SuffixOrder(config.order), CombineCounts()));
  chain.Wait(true);
  return sorter.release();
}

void InitialProbabilities(const std::vector<uint64_t> &counts, const std::vector<uint64_t> &counts_pruned, const std::vector<Discount> &discounts, Master &master, Sorts<SuffixOrder> &primary, util::FixedArray<util::stream::FileBuffer> &gammas, const std::vector<uint64_t> &prune_thresholds, bool prune_vocab, const SpecialVocab &specials) {
  const PipelineConfig &config = master.Config();
  util::stream::Chains second(config.order);

  {
    Sorts<ContextOrder> sorts;
    master.SetupSorts(sorts, !config.renumber_vocabulary);
    PrintStatistics(counts, counts_pruned, discounts);
    lm::ngram::ShowSizes(counts_pruned);
    std::cerr << "=== 3/" << master.Steps() << " Calculating and sorting initial probabilities ===" << std::endl;
    master.SortAndReadTwice(counts_pruned, sorts, second, config.initial_probs.adder_in);
  }

  util::stream::Chains gamma_chains(config.order);
  InitialProbabilities(config.initial_probs, discounts, master.MutableChains(), second, gamma_chains, prune_thresholds, prune_vocab, specials);
  // Don't care about gamma for 0.
  gamma_chains[0] >> util::stream::kRecycle;
  gammas.Init(config.order - 1);
  for (std::size_t i = 1; i < config.order; ++i) {
    gammas.push_back(util::MakeTemp(config.TempPrefix()));
    gamma_chains[i] >> gammas[i - 1].Sink() >> util::stream::kRecycle;
  }
  // Has to be done here due to gamma_chains scope.
  master.SetupSorts(primary, true);
}

void InterpolateProbabilities(const std::vector<uint64_t> &counts, Master &master, Sorts<SuffixOrder> &primary, util::FixedArray<util::stream::FileBuffer> &gammas, Output &output, const SpecialVocab &specials) {
  std::cerr << "=== 4/" << master.Steps() << " Calculating and writing order-interpolated probabilities ===" << std::endl;
  const PipelineConfig &config = master.Config();
  master.MaximumLazyInput(counts, primary);

  util::stream::Chains gamma_chains(config.order - 1);
  for (std::size_t i = 0; i < config.order - 1; ++i) {
    util::stream::ChainConfig read_backoffs(config.read_backoffs);

    if(config.prune_vocab || config.prune_thresholds[i + 1] > 0)
        read_backoffs.entry_size = sizeof(HashGamma);
    else
        read_backoffs.entry_size = sizeof(float);

    gamma_chains.push_back(read_backoffs);
    gamma_chains.back() >> gammas[i].Source(true);
  }
  master >> Interpolate(std::max(master.Config().vocab_size_for_unk, counts[0] - 1 /* <s> is not included */), util::stream::ChainPositions(gamma_chains), config.prune_thresholds, config.prune_vocab, config.output_q, specials);
  gamma_chains >> util::stream::kRecycle;
  output.SinkProbs(master.MutableChains());
}

class VocabNumbering {
  public:
    VocabNumbering(int final_vocab, StringPiece temp_prefix, bool renumber)
      : final_vocab_(final_vocab),
        renumber_(renumber),
        specials_(kBOS, kEOS) {
      if (renumber) {
        temporary_.reset(util::MakeTemp(temp_prefix));
      }
    }

    int WriteOnTheFly() const { return renumber_ ? temporary_.get() : final_vocab_; }

    // Compute the vocabulary mapping and return the memory used.
    std::size_t ComputeMapping(WordIndex type_count) {
      if (!renumber_) return 0;
      ngram::SortedVocabulary::ComputeRenumbering(type_count, temporary_.get(), final_vocab_, vocab_mapping_);
      temporary_.reset();
      return sizeof(WordIndex) * vocab_mapping_.size();
    }

    void ApplyRenumber(util::stream::Chains &chains) {
      if (!renumber_) return;
      for (std::size_t i = 0; i < chains.size(); ++i) {
        chains[i] >> Renumber(&*vocab_mapping_.begin(), i + 1);
      }
      specials_ = SpecialVocab(vocab_mapping_[specials_.BOS()], vocab_mapping_[specials_.EOS()]);
    }

    const SpecialVocab &Specials() const { return specials_; }

  private:
    int final_vocab_;
    // Out of order vocab file created on the fly.
    util::scoped_fd temporary_;

    bool renumber_;

    std::vector<WordIndex> vocab_mapping_;

    SpecialVocab specials_;
};

} // namespace

void Pipeline(PipelineConfig &config, int text_file, Output &output) {
  // Some fail-fast sanity checks.
  if (config.sort.buffer_size * 4 > config.TotalMemory()) {
    config.sort.buffer_size = config.TotalMemory() / 4;
    std::cerr << "Warning: changing sort block size to " << config.sort.buffer_size << " bytes due to low total memory." << std::endl;
  }
  if (config.minimum_block < NGram<BuildingPayload>::TotalSize(config.order)) {
    config.minimum_block = NGram<BuildingPayload>::TotalSize(config.order);
    std::cerr << "Warning: raising minimum block to " << config.minimum_block << " to fit an ngram in every block." << std::endl;
  }
  UTIL_THROW_IF(config.sort.buffer_size < config.minimum_block, util::Exception, "Sort block size " << config.sort.buffer_size << " is below the minimum block size " << config.minimum_block << ".");
  UTIL_THROW_IF(config.TotalMemory() < config.minimum_block * config.order * config.block_count, util::Exception,
      "Not enough memory to fit " << (config.order * config.block_count) << " blocks with minimum size " << config.minimum_block << ".  Increase memory to " << (config.minimum_block * config.order * config.block_count) << " bytes or decrease the minimum block size.");

  Master master(config, output.Steps());
  // master's destructor will wait for chains.  But they might be deadlocked if
  // this thread dies because e.g. it ran out of memory.
  try {
    VocabNumbering numbering(output.VocabFile(), config.TempPrefix(), config.renumber_vocabulary);
    uint64_t token_count;
    WordIndex type_count;
    std::string text_file_name;
    std::vector<bool> prune_words;
    util::scoped_ptr<util::stream::Sort<SuffixOrder, CombineCounts> > sorted_counts(
        CountText(text_file, numbering.WriteOnTheFly(), master, token_count, type_count, text_file_name, prune_words));
    std::cerr << "Unigram tokens " << token_count << " types " << type_count << std::endl;

    // Create vocab mapping, which uses temporary memory, while nothing else is happening.
    std::size_t subtract_for_numbering = numbering.ComputeMapping(type_count);

    std::cerr << "=== 2/" << master.Steps() << " Calculating and sorting adjusted counts ===" << std::endl;
    master.InitForAdjust(*sorted_counts, type_count, subtract_for_numbering);
    sorted_counts.reset();

    std::vector<uint64_t> counts;
    std::vector<uint64_t> counts_pruned;
    std::vector<Discount> discounts;
    master >> AdjustCounts(config.prune_thresholds, counts, counts_pruned, prune_words, config.discount, discounts);
    numbering.ApplyRenumber(master.MutableChains());

    {
      util::FixedArray<util::stream::FileBuffer> gammas;
      Sorts<SuffixOrder> primary;
      InitialProbabilities(counts, counts_pruned, discounts, master, primary, gammas, config.prune_thresholds, config.prune_vocab, numbering.Specials());
      output.SetHeader(HeaderInfo(text_file_name, token_count, counts_pruned));
      // Also does output.
      InterpolateProbabilities(counts_pruned, master, primary, gammas, output, numbering.Specials());
    }
  } catch (const util::Exception &e) {
    std::cerr << e.what() << std::endl;
    abort();
  }
}

}} // namespaces
