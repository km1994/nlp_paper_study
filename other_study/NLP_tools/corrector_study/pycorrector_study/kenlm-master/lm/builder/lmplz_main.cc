#include "lm/builder/output.hh"
#include "lm/builder/pipeline.hh"
#include "lm/common/size_option.hh"
#include "lm/lm_exception.hh"
#include "util/file.hh"
#include "util/file_piece.hh"
#include "util/usage.hh"

#include <iostream>

#include <boost/program_options.hpp>
#include <boost/version.hpp>
#include <vector>

namespace {

// Parse and validate pruning thresholds then return vector of threshold counts
// for each n-grams order.
std::vector<uint64_t> ParsePruning(const std::vector<std::string> &param, std::size_t order) {
  // convert to vector of integers
  std::vector<uint64_t> prune_thresholds;
  prune_thresholds.reserve(order);
  for (std::vector<std::string>::const_iterator it(param.begin()); it != param.end(); ++it) {
    try {
      prune_thresholds.push_back(boost::lexical_cast<uint64_t>(*it));
    } catch(const boost::bad_lexical_cast &) {
      UTIL_THROW(util::Exception, "Bad pruning threshold " << *it);
    }
  }

  // Fill with zeros by default.
  if (prune_thresholds.empty()) {
    prune_thresholds.resize(order, 0);
    return prune_thresholds;
  }

  // validate pruning threshold if specified
  // throw if each n-gram order has not  threshold specified
  UTIL_THROW_IF(prune_thresholds.size() > order, util::Exception, "You specified pruning thresholds for orders 1 through " << prune_thresholds.size() << " but the model only has order " << order);
  // threshold for unigram can only be 0 (no pruning)

  // check if threshold are not in decreasing order
  uint64_t lower_threshold = 0;
  for (std::vector<uint64_t>::iterator it = prune_thresholds.begin(); it != prune_thresholds.end(); ++it) {
    UTIL_THROW_IF(lower_threshold > *it, util::Exception, "Pruning thresholds should be in non-decreasing order.  Otherwise substrings would be removed, which is bad for query-time data structures.");
    lower_threshold = *it;
  }

  // Pad to all orders using the last value.
  prune_thresholds.resize(order, prune_thresholds.back());
  return prune_thresholds;
}

lm::builder::Discount ParseDiscountFallback(const std::vector<std::string> &param) {
  lm::builder::Discount ret;
  UTIL_THROW_IF(param.size() > 3, util::Exception, "Specify at most three fallback discounts: 1, 2, and 3+");
  UTIL_THROW_IF(param.empty(), util::Exception, "Fallback discounting enabled, but no discount specified");
  ret.amount[0] = 0.0;
  for (unsigned i = 0; i < 3; ++i) {
    float discount = boost::lexical_cast<float>(param[i < param.size() ? i : (param.size() - 1)]);
    UTIL_THROW_IF(discount < 0.0 || discount > static_cast<float>(i+1), util::Exception, "The discount for count " << (i+1) << " was parsed as " << discount << " which is not in the range [0, " << (i+1) << "].");
    ret.amount[i + 1] = discount;
  }
  return ret;
}

} // namespace

int main(int argc, char *argv[]) {
  try {
    namespace po = boost::program_options;
    po::options_description options("Language model building options");
    lm::builder::PipelineConfig pipeline;

    std::string text, intermediate, arpa;
    std::vector<std::string> pruning;
    std::vector<std::string> discount_fallback;
    std::vector<std::string> discount_fallback_default;
    discount_fallback_default.push_back("0.5");
    discount_fallback_default.push_back("1");
    discount_fallback_default.push_back("1.5");
    bool verbose_header;

    options.add_options()
      ("help,h", po::bool_switch(), "Show this help message")
      ("order,o", po::value<std::size_t>(&pipeline.order)
#if BOOST_VERSION >= 104200
         ->required()
#endif
         , "Order of the model")
      ("interpolate_unigrams", po::value<bool>(&pipeline.initial_probs.interpolate_unigrams)->default_value(true)->implicit_value(true), "Interpolate the unigrams (default) as opposed to giving lots of mass to <unk> like SRI.  If you want SRI's behavior with a large <unk> and the old lmplz default, use --interpolate_unigrams 0.")
      ("skip_symbols", po::bool_switch(), "Treat <s>, </s>, and <unk> as whitespace instead of throwing an exception")
      ("temp_prefix,T", po::value<std::string>(&pipeline.sort.temp_prefix)->default_value(util::DefaultTempDirectory()), "Temporary file prefix")
      ("memory,S", lm:: SizeOption(pipeline.sort.total_memory, util::GuessPhysicalMemory() ? "80%" : "1G"), "Sorting memory")
      ("minimum_block", lm::SizeOption(pipeline.minimum_block, "8K"), "Minimum block size to allow")
      ("sort_block", lm::SizeOption(pipeline.sort.buffer_size, "64M"), "Size of IO operations for sort (determines arity)")
      ("block_count", po::value<std::size_t>(&pipeline.block_count)->default_value(2), "Block count (per order)")
      ("vocab_estimate", po::value<lm::WordIndex>(&pipeline.vocab_estimate)->default_value(1000000), "Assume this vocabulary size for purposes of calculating memory in step 1 (corpus count) and pre-sizing the hash table")
      ("vocab_pad", po::value<uint64_t>(&pipeline.vocab_size_for_unk)->default_value(0), "If the vocabulary is smaller than this value, pad with <unk> to reach this size. Requires --interpolate_unigrams")
      ("verbose_header", po::bool_switch(&verbose_header), "Add a verbose header to the ARPA file that includes information such as token count, smoothing type, etc.")
      ("text", po::value<std::string>(&text), "Read text from a file instead of stdin")
      ("arpa", po::value<std::string>(&arpa), "Write ARPA to a file instead of stdout")
      ("intermediate", po::value<std::string>(&intermediate), "Write ngrams to intermediate files.  Turns off ARPA output (which can be reactivated by --arpa file).  Forces --renumber on.")
      ("renumber", po::bool_switch(&pipeline.renumber_vocabulary), "Renumber the vocabulary identifiers so that they are monotone with the hash of each string.  This is consistent with the ordering used by the trie data structure.")
      ("collapse_values", po::bool_switch(&pipeline.output_q), "Collapse probability and backoff into a single value, q that yields the same sentence-level probabilities.  See http://kheafield.com/professional/edinburgh/rest_paper.pdf for more details, including a proof.")
      ("prune", po::value<std::vector<std::string> >(&pruning)->multitoken(), "Prune n-grams with count less than or equal to the given threshold.  Specify one value for each order i.e. 0 0 1 to prune singleton trigrams and above.  The sequence of values must be non-decreasing and the last value applies to any remaining orders. Default is to not prune, which is equivalent to --prune 0.")
      ("limit_vocab_file", po::value<std::string>(&pipeline.prune_vocab_file)->default_value(""), "Read allowed vocabulary separated by whitespace. N-grams that contain vocabulary items not in this list will be pruned. Can be combined with --prune arg")
      ("discount_fallback", po::value<std::vector<std::string> >(&discount_fallback)->multitoken()->implicit_value(discount_fallback_default, "0.5 1 1.5"), "The closed-form estimate for Kneser-Ney discounts does not work without singletons or doubletons.  It can also fail if these values are out of range.  This option falls back to user-specified discounts when the closed-form estimate fails.  Note that this option is generally a bad idea: you should deduplicate your corpus instead.  However, class-based models need custom discounts because they lack singleton unigrams.  Provide up to three discounts (for adjusted counts 1, 2, and 3+), which will be applied to all orders where the closed-form estimates fail.");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);

    if (argc == 1 || vm["help"].as<bool>()) {
      std::cerr <<
        "Builds unpruned language models with modified Kneser-Ney smoothing.\n\n"
        "Please cite:\n"
        "@inproceedings{Heafield-estimate,\n"
        "  author = {Kenneth Heafield and Ivan Pouzyrevsky and Jonathan H. Clark and Philipp Koehn},\n"
        "  title = {Scalable Modified {Kneser-Ney} Language Model Estimation},\n"
        "  year = {2013},\n"
        "  month = {8},\n"
        "  booktitle = {Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics},\n"
        "  address = {Sofia, Bulgaria},\n"
        "  url = {http://kheafield.com/professional/edinburgh/estimate\\_paper.pdf},\n"
        "}\n\n"
        "Provide the corpus on stdin.  The ARPA file will be written to stdout.  Order of\n"
        "the model (-o) is the only mandatory option.  As this is an on-disk program,\n"
        "setting the temporary file location (-T) and sorting memory (-S) is recommended.\n\n"
        "Memory sizes are specified like GNU sort: a number followed by a unit character.\n"
        "Valid units are \% for percentage of memory (supported platforms only) and (in\n"
        "increasing powers of 1024): b, K, M, G, T, P, E, Z, Y.  Default is K (*1024).\n";
      uint64_t mem = util::GuessPhysicalMemory();
      if (mem) {
        std::cerr << "This machine has " << mem << " bytes of memory.\n\n";
      } else {
        std::cerr << "Unable to determine the amount of memory on this machine.\n\n";
      }
      std::cerr << options << std::endl;
      return 1;
    }

    po::notify(vm);

    // required() appeared in Boost 1.42.0.
#if BOOST_VERSION < 104200
    if (!vm.count("order")) {
      std::cerr << "the option '--order' is required but missing" << std::endl;
      return 1;
    }
#endif

    if (pipeline.vocab_size_for_unk && !pipeline.initial_probs.interpolate_unigrams) {
      std::cerr << "--vocab_pad requires --interpolate_unigrams be on" << std::endl;
      return 1;
    }

    if (vm["skip_symbols"].as<bool>()) {
      pipeline.disallowed_symbol_action = lm::COMPLAIN;
    } else {
      pipeline.disallowed_symbol_action = lm::THROW_UP;
    }

    if (vm.count("discount_fallback")) {
      pipeline.discount.fallback = ParseDiscountFallback(discount_fallback);
      pipeline.discount.bad_action = lm::COMPLAIN;
    } else {
      // Unused, just here to prevent the compiler from complaining about uninitialized.
      pipeline.discount.fallback = lm::builder::Discount();
      pipeline.discount.bad_action = lm::THROW_UP;
    }

    // parse pruning thresholds.  These depend on order, so it is not done as a notifier.
    pipeline.prune_thresholds = ParsePruning(pruning, pipeline.order);

    if (!vm["limit_vocab_file"].as<std::string>().empty()) {
      pipeline.prune_vocab = true;
    }
    else {
      pipeline.prune_vocab = false;
    }

    util::NormalizeTempPrefix(pipeline.sort.temp_prefix);

    lm::builder::InitialProbabilitiesConfig &initial = pipeline.initial_probs;
    // TODO: evaluate options for these.
    initial.adder_in.total_memory = 32768;
    initial.adder_in.block_count = 2;
    initial.adder_out.total_memory = 32768;
    initial.adder_out.block_count = 2;
    pipeline.read_backoffs = initial.adder_out;

    // Read from stdin, write to stdout by default
    util::scoped_fd in(0), out(1);
    if (vm.count("text")) {
      in.reset(util::OpenReadOrThrow(text.c_str()));
    }
    if (vm.count("arpa")) {
      out.reset(util::CreateOrThrow(arpa.c_str()));
    }

    try {
      bool writing_intermediate = vm.count("intermediate");
      if (writing_intermediate) {
        pipeline.renumber_vocabulary = true;
      }
      lm::builder::Output output(writing_intermediate ? intermediate : pipeline.sort.temp_prefix, writing_intermediate, pipeline.output_q);
      if (!writing_intermediate || vm.count("arpa")) {
        output.Add(new lm::builder::PrintHook(out.release(), verbose_header));
      }
      lm::builder::Pipeline(pipeline, in.release(), output);
    } catch (const util::MallocException &e) {
      std::cerr << e.what() << std::endl;
      std::cerr << "Try rerunning with a more conservative -S setting than " << vm["memory"].as<std::string>() << std::endl;
      return 1;
    }
    util::PrintUsage(std::cerr);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
