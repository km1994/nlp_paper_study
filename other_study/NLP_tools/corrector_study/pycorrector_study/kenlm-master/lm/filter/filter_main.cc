#include "lm/filter/arpa_io.hh"
#include "lm/filter/format.hh"
#include "lm/filter/phrase.hh"
#ifndef NTHREAD
#include "lm/filter/thread.hh"
#endif
#include "lm/filter/vocab.hh"
#include "lm/filter/wrapper.hh"
#include "util/exception.hh"
#include "util/file_piece.hh"

#include <boost/ptr_container/ptr_vector.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

namespace lm {
namespace {

void DisplayHelp(const char *name) {
  std::cerr
    << "Usage: " << name << " mode [context] [phrase] [raw|arpa] [threads:m] [batch_size:m] (vocab|model):input_file output_file\n\n"
    "copy mode just copies, but makes the format nicer for e.g. irstlm's broken\n"
    "    parser.\n"
    "single mode treats the entire input as a single sentence.\n"
    "multiple mode filters to multiple sentences in parallel.  Each sentence is on\n"
    "    a separate line.  A separate file is created for each sentence by appending\n"
    "    the 0-indexed line number to the output file name.\n"
    "union mode produces one filtered model that is the union of models created by\n"
    "    multiple mode.\n\n"
    "context means only the context (all but last word) has to pass the filter, but\n"
    "    the entire n-gram is output.\n\n"
    "phrase means that the vocabulary is actually tab-delimited phrases and that the\n"
    "    phrases can generate the n-gram when assembled in arbitrary order and\n"
    "    clipped.  Currently works with multiple or union mode.\n\n"
    "The file format is set by [raw|arpa] with default arpa:\n"
    "raw means space-separated tokens, optionally followed by a tab and arbitrary\n"
    "    text.  This is useful for ngram count files.\n"
    "arpa means the ARPA file format for n-gram language models.\n\n"
#ifndef NTHREAD
    "threads:m sets m threads (default: conccurrency detected by boost)\n"
    "batch_size:m sets the batch size for threading.  Expect memory usage from this\n"
    "    of 2*threads*batch_size n-grams.\n\n"
#else
    "This binary was compiled with -DNTHREAD, disabling threading.  If you wanted\n"
    "    threading, compile without this flag against Boost >=1.42.0.\n\n"
#endif
    "There are two inputs: vocabulary and model.  Either may be given as a file\n"
    "    while the other is on stdin.  Specify the type given as a file using\n"
    "    vocab: or model: before the file name.  \n\n"
    "For ARPA format, the output must be seekable.  For raw format, it can be a\n"
    "    stream i.e. /dev/stdout\n";
}

typedef enum {MODE_COPY, MODE_SINGLE, MODE_MULTIPLE, MODE_UNION, MODE_UNSET} FilterMode;
typedef enum {FORMAT_ARPA, FORMAT_COUNT} Format;

struct Config {
  Config() :
#ifndef NTHREAD
  batch_size(25000),
  threads(boost::thread::hardware_concurrency()),
#endif
  phrase(false),
  context(false),
  format(FORMAT_ARPA)
  {
#ifndef NTHREAD
    if (!threads) threads = 1;
#endif
  }

#ifndef NTHREAD
  size_t batch_size;
  size_t threads;
#endif
  bool phrase;
  bool context;
  FilterMode mode;
  Format format;
};

template <class Format, class Filter, class OutputBuffer, class Output> void RunThreadedFilter(const Config &config, util::FilePiece &in_lm, Filter &filter, Output &output) {
#ifndef NTHREAD
  if (config.threads == 1) {
#endif
    Format::RunFilter(in_lm, filter, output);
#ifndef NTHREAD
  } else {
    typedef Controller<Filter, OutputBuffer, Output> Threaded;
    Threaded threading(config.batch_size, config.threads * 2, config.threads, filter, output);
    Format::RunFilter(in_lm, threading, output);
  }
#endif
}

template <class Format, class Filter, class OutputBuffer, class Output> void RunContextFilter(const Config &config, util::FilePiece &in_lm, Filter filter, Output &output) {
  if (config.context) {
    ContextFilter<Filter> context_filter(filter);
    RunThreadedFilter<Format, ContextFilter<Filter>, OutputBuffer, Output>(config, in_lm, context_filter, output);
  } else {
    RunThreadedFilter<Format, Filter, OutputBuffer, Output>(config, in_lm, filter, output);
  }
}

template <class Format, class Binary> void DispatchBinaryFilter(const Config &config, util::FilePiece &in_lm, const Binary &binary, typename Format::Output &out) {
  typedef BinaryFilter<Binary> Filter;
  RunContextFilter<Format, Filter, BinaryOutputBuffer, typename Format::Output>(config, in_lm, Filter(binary), out);
}

template <class Format> void DispatchFilterModes(const Config &config, std::istream &in_vocab, util::FilePiece &in_lm, const char *out_name) {
  if (config.mode == MODE_MULTIPLE) {
    if (config.phrase) {
      typedef phrase::Multiple Filter;
      phrase::Substrings substrings;
      typename Format::Multiple out(out_name, phrase::ReadMultiple(in_vocab, substrings));
      RunContextFilter<Format, Filter, MultipleOutputBuffer, typename Format::Multiple>(config, in_lm, Filter(substrings), out);
    } else {
      typedef vocab::Multiple Filter;
      boost::unordered_map<std::string, std::vector<unsigned int> > words;
      typename Format::Multiple out(out_name, vocab::ReadMultiple(in_vocab, words));
      RunContextFilter<Format, Filter, MultipleOutputBuffer, typename Format::Multiple>(config, in_lm, Filter(words), out);
    }
    return;
  }

  typename Format::Output out(out_name);

  if (config.mode == MODE_COPY) {
    Format::Copy(in_lm, out);
    return;
  }

  if (config.mode == MODE_SINGLE) {
    vocab::Single::Words words;
    vocab::ReadSingle(in_vocab, words);
    DispatchBinaryFilter<Format, vocab::Single>(config, in_lm, vocab::Single(words), out);
    return;
  }

  if (config.mode == MODE_UNION) {
    if (config.phrase) {
      phrase::Substrings substrings;
      phrase::ReadMultiple(in_vocab, substrings);
      DispatchBinaryFilter<Format, phrase::Union>(config, in_lm, phrase::Union(substrings), out);
    } else {
      vocab::Union::Words words;
      vocab::ReadMultiple(in_vocab, words);
      DispatchBinaryFilter<Format, vocab::Union>(config, in_lm, vocab::Union(words), out);
    }
    return;
  }
}

} // namespace
} // namespace lm

int main(int argc, char *argv[]) {
  try {
    if (argc < 4) {
      lm::DisplayHelp(argv[0]);
      return 1;
    }

    // I used to have boost::program_options, but some users didn't want to compile boost.
    lm::Config config;
    config.mode = lm::MODE_UNSET;
    for (int i = 1; i < argc - 2; ++i) {
      const char *str = argv[i];
      if (!std::strcmp(str, "copy")) {
        config.mode = lm::MODE_COPY;
      } else if (!std::strcmp(str, "single")) {
        config.mode = lm::MODE_SINGLE;
      } else if (!std::strcmp(str, "multiple")) {
        config.mode = lm::MODE_MULTIPLE;
      } else if (!std::strcmp(str, "union")) {
        config.mode = lm::MODE_UNION;
      } else if (!std::strcmp(str, "phrase")) {
        config.phrase = true;
      } else if (!std::strcmp(str, "context")) {
        config.context = true;
      } else if (!std::strcmp(str, "arpa")) {
        config.format = lm::FORMAT_ARPA;
      } else if (!std::strcmp(str, "raw")) {
        config.format = lm::FORMAT_COUNT;
#ifndef NTHREAD
      } else if (!std::strncmp(str, "threads:", 8)) {
        config.threads = boost::lexical_cast<size_t>(str + 8);
        if (!config.threads) {
          std::cerr << "Specify at least one thread." << std::endl;
          return 1;
        }
      } else if (!std::strncmp(str, "batch_size:", 11)) {
        config.batch_size = boost::lexical_cast<size_t>(str + 11);
        if (config.batch_size < 5000) {
          std::cerr << "Batch size must be at least one and should probably be >= 5000" << std::endl;
          if (!config.batch_size) return 1;
        }
#endif
      } else {
        lm::DisplayHelp(argv[0]);
        return 1;
      }
    }

    if (config.mode == lm::MODE_UNSET) {
      lm::DisplayHelp(argv[0]);
      return 1;
    }

    if (config.phrase && config.mode != lm::MODE_UNION && config.mode != lm::MODE_MULTIPLE) {
      std::cerr << "Phrase constraint currently only works in multiple or union mode.  If you really need it for single, put everything on one line and use union." << std::endl;
      return 1;
    }

    bool cmd_is_model = true;
    const char *cmd_input = argv[argc - 2];
    if (!strncmp(cmd_input, "vocab:", 6)) {
      cmd_is_model = false;
      cmd_input += 6;
    } else if (!strncmp(cmd_input, "model:", 6)) {
      cmd_input += 6;
    } else if (strchr(cmd_input, ':')) {
      std::cerr << "Specify vocab: or model: before the input file name, not " << cmd_input << std::endl;
      return 1;
    } else {
      std::cerr << "Assuming that " << cmd_input << " is a model file" << std::endl;
    }
    std::ifstream cmd_file;
    std::istream *vocab;
    if (cmd_is_model) {
      vocab = &std::cin;
    } else {
      cmd_file.open(cmd_input, std::ios::in);
      UTIL_THROW_IF(!cmd_file, util::ErrnoException, "Failed to open " << cmd_input);
      vocab = &cmd_file;
    }

    util::FilePiece model(cmd_is_model ? util::OpenReadOrThrow(cmd_input) : 0, cmd_is_model ? cmd_input : NULL, &std::cerr);

    if (config.format == lm::FORMAT_ARPA) {
      lm::DispatchFilterModes<lm::ARPAFormat>(config, *vocab, model, argv[argc - 1]);
    } else if (config.format == lm::FORMAT_COUNT) {
      lm::DispatchFilterModes<lm::CountFormat>(config, *vocab, model, argv[argc - 1]);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
