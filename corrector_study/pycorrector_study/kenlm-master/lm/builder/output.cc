#include "lm/builder/output.hh"

#include "lm/common/model_buffer.hh"
#include "lm/common/print.hh"
#include "util/file_stream.hh"
#include "util/stream/multi_stream.hh"

#include <iostream>

namespace lm { namespace builder {

OutputHook::~OutputHook() {}

Output::Output(StringPiece file_base, bool keep_buffer, bool output_q)
  : buffer_(file_base, keep_buffer, output_q) {}

void Output::SinkProbs(util::stream::Chains &chains) {
  Apply(PROB_PARALLEL_HOOK, chains);
  if (!buffer_.Keep() && !Have(PROB_SEQUENTIAL_HOOK)) {
    chains >> util::stream::kRecycle;
    chains.Wait(true);
    return;
  }
  buffer_.Sink(chains, header_.counts_pruned);
  chains >> util::stream::kRecycle;
  chains.Wait(false);
  if (Have(PROB_SEQUENTIAL_HOOK)) {
    std::cerr << "=== 5/5 Writing ARPA model ===" << std::endl;
    buffer_.Source(chains);
    Apply(PROB_SEQUENTIAL_HOOK, chains);
    chains >> util::stream::kRecycle;
    chains.Wait(true);
  }
}

void Output::Apply(HookType hook_type, util::stream::Chains &chains) {
  for (boost::ptr_vector<OutputHook>::iterator entry = outputs_[hook_type].begin(); entry != outputs_[hook_type].end(); ++entry) {
    entry->Sink(header_, VocabFile(), chains);
  }
}

void PrintHook::Sink(const HeaderInfo &info, int vocab_file, util::stream::Chains &chains) {
  if (verbose_header_) {
    util::FileStream out(file_.get(), 50);
    out << "# Input file: " << info.input_file << '\n';
    out << "# Token count: " << info.token_count << '\n';
    out << "# Smoothing: Modified Kneser-Ney" << '\n';
  }
  chains >> PrintARPA(vocab_file, file_.get(), info.counts_pruned);
}

}} // namespaces
