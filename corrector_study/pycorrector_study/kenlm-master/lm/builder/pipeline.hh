#ifndef LM_BUILDER_PIPELINE_H
#define LM_BUILDER_PIPELINE_H

#include "lm/builder/adjust_counts.hh"
#include "lm/builder/initial_probabilities.hh"
#include "lm/builder/header_info.hh"
#include "lm/lm_exception.hh"
#include "lm/word_index.hh"
#include "util/stream/config.hh"
#include "util/file_piece.hh"

#include <string>
#include <cstddef>

namespace lm { namespace builder {

class Output;

struct PipelineConfig {
  std::size_t order;
  util::stream::SortConfig sort;
  InitialProbabilitiesConfig initial_probs;
  util::stream::ChainConfig read_backoffs;

  // Estimated vocabulary size.  Used for sizing CorpusCount memory and
  // initial probing hash table sizing, also in CorpusCount.
  lm::WordIndex vocab_estimate;

  // Minimum block size to tolerate.
  std::size_t minimum_block;

  // Number of blocks to use.  This will be overridden to 1 if everything fits.
  std::size_t block_count;

  // n-gram count thresholds for pruning. 0 values means no pruning for
  // corresponding n-gram order
  std::vector<uint64_t> prune_thresholds; //mjd
  bool prune_vocab;
  std::string prune_vocab_file;

  /* Renumber the vocabulary the way the trie likes it? */
  bool renumber_vocabulary;

  // What to do with discount failures.
  DiscountConfig discount;

  // Compute collapsed q values instead of probability and backoff
  bool output_q;

  /* Computing the perplexity of LMs with different vocabularies is hard.  For
   * example, the lowest perplexity is attained by a unigram model that
   * predicts p(<unk>) = 1 and has no other vocabulary.  Also, linearly
   * interpolated models will sum to more than 1 because <unk> is duplicated
   * (SRI just pretends p(<unk>) = 0 for these purposes, which makes it sum to
   * 1 but comes with its own problems).  This option will make the vocabulary
   * a particular size by replicating <unk> multiple times for purposes of
   * computing vocabulary size.  It has no effect if the actual vocabulary is
   * larger.  This parameter serves the same purpose as IRSTLM's "dub".
   */
  uint64_t vocab_size_for_unk;

  /* What to do the first time <s>, </s>, or <unk> appears in the input.  If
   * this is anything but THROW_UP, then the symbol will always be treated as
   * whitespace.
   */
  WarningAction disallowed_symbol_action;

  const std::string &TempPrefix() const { return sort.temp_prefix; }
  std::size_t TotalMemory() const { return sort.total_memory; }
};

// Takes ownership of text_file and out_arpa.
void Pipeline(PipelineConfig &config, int text_file, Output &output);

}} // namespaces
#endif // LM_BUILDER_PIPELINE_H
