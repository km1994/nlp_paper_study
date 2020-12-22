#ifndef LM_BUILDER_OUTPUT_H
#define LM_BUILDER_OUTPUT_H

#include "lm/builder/header_info.hh"
#include "lm/common/model_buffer.hh"
#include "util/file.hh"

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/utility.hpp>

namespace util { namespace stream { class Chains; class ChainPositions; } }

/* Outputs from lmplz: ARPA, sharded files, etc */
namespace lm { namespace builder {

// These are different types of hooks.  Values should be consecutive to enable a vector lookup.
enum HookType {
  // TODO: counts.
  PROB_PARALLEL_HOOK, // Probability and backoff (or just q).  Output must process the orders in parallel or there will be a deadlock.
  PROB_SEQUENTIAL_HOOK, // Probability and backoff (or just q).  Output can process orders any way it likes.  This requires writing the data to disk then reading.  Useful for ARPA files, which put unigrams first etc.
  NUMBER_OF_HOOKS // Keep this last so we know how many values there are.
};

class OutputHook {
  public:
    explicit OutputHook(HookType hook_type) : type_(hook_type) {}

    virtual ~OutputHook();

    virtual void Sink(const HeaderInfo &info, int vocab_file, util::stream::Chains &chains) = 0;

    HookType Type() const { return type_; }

  private:
    HookType type_;
};

class Output : boost::noncopyable {
  public:
    Output(StringPiece file_base, bool keep_buffer, bool output_q);

    // Takes ownership.
    void Add(OutputHook *hook) {
      outputs_[hook->Type()].push_back(hook);
    }

    bool Have(HookType hook_type) const {
      return !outputs_[hook_type].empty();
    }

    int VocabFile() const { return buffer_.VocabFile(); }

    void SetHeader(const HeaderInfo &header) { header_ = header; }
    const HeaderInfo &GetHeader() const { return header_; }

    // This is called by the pipeline.
    void SinkProbs(util::stream::Chains &chains);

    unsigned int Steps() const { return Have(PROB_SEQUENTIAL_HOOK); }

  private:
    void Apply(HookType hook_type, util::stream::Chains &chains);

    ModelBuffer buffer_;

    boost::ptr_vector<OutputHook> outputs_[NUMBER_OF_HOOKS];
    HeaderInfo header_;
};

class PrintHook : public OutputHook {
  public:
    // Takes ownership
    PrintHook(int write_fd, bool verbose_header)
      : OutputHook(PROB_SEQUENTIAL_HOOK), file_(write_fd), verbose_header_(verbose_header) {}

    void Sink(const HeaderInfo &info, int vocab_file, util::stream::Chains &chains);

  private:
    util::scoped_fd file_;
    bool verbose_header_;
};

}} // namespaces

#endif // LM_BUILDER_OUTPUT_H
