#include "lm/builder/combine_counts.hh"
#include "lm/builder/corpus_count.hh"
#include "lm/common/compare.hh"
#include "util/stream/chain.hh"
#include "util/stream/io.hh"
#include "util/stream/sort.hh"
#include "util/file.hh"
#include "util/file_piece.hh"
#include "util/usage.hh"

#include <boost/program_options.hpp>

#include <string>

namespace {
class SizeNotify {
  public:
    SizeNotify(std::size_t &out) : behind_(out) {}

    void operator()(const std::string &from) {
      behind_ = util::ParseSize(from);
    }

  private:
    std::size_t &behind_;
};

boost::program_options::typed_value<std::string> *SizeOption(std::size_t &to, const char *default_value) {
  return boost::program_options::value<std::string>()->notifier(SizeNotify(to))->default_value(default_value);
}

} // namespace

int main(int argc, char *argv[]) {
  namespace po = boost::program_options;
  unsigned order;
  std::size_t ram;
  std::string temp_prefix, vocab_table, vocab_list;
  po::options_description options("corpus count");
  options.add_options()
    ("help,h", po::bool_switch(), "Show this help message")
    ("order,o", po::value<unsigned>(&order)->required(), "Order")
    ("temp_prefix,T", po::value<std::string>(&temp_prefix)->default_value(util::DefaultTempDirectory()), "Temporary file prefix")
    ("memory,S", SizeOption(ram, "80%"), "RAM")
    ("read_vocab_table", po::value<std::string>(&vocab_table), "Vocabulary hash table to read.  This should be a probing hash table with size at the beginning.")
    ("write_vocab_list", po::value<std::string>(&vocab_list), "Vocabulary list to write as null-delimited strings.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  if (argc == 1 || vm["help"].as<bool>()) {
    std::cerr << "Counts n-grams from standard input.\n" << options << std::endl;
    return 1;
  }
  po::notify(vm);

  if (!(vocab_table.empty() ^ vocab_list.empty())) {
    std::cerr << "Specify one of --read_vocab_table or --write_vocab_list for vocabulary handling." << std::endl;
    return 1;
  }

  util::NormalizeTempPrefix(temp_prefix);

  util::scoped_fd vocab_file(vocab_table.empty() ? util::CreateOrThrow(vocab_list.c_str()) : util::OpenReadOrThrow(vocab_table.c_str()));

  std::size_t blocks = 2;
  std::size_t remaining_size = ram - util::SizeOrThrow(vocab_file.get());

  std::size_t memory_for_chain =
    // This much memory to work with after vocab hash table.
    static_cast<float>(remaining_size) /
    // Solve for block size including the dedupe multiplier for one block.
    (static_cast<float>(blocks) + lm::builder::CorpusCount::DedupeMultiplier(order)) *
    // Chain likes memory expressed in terms of total memory.
    static_cast<float>(blocks);
  std::cerr << "Using " << memory_for_chain << " for chains." << std::endl;
  
  util::stream::Chain chain(util::stream::ChainConfig(lm::NGram<uint64_t>::TotalSize(order), blocks, memory_for_chain));
  util::FilePiece f(0, NULL, &std::cerr);
  uint64_t token_count = 0;
  lm::WordIndex type_count = 0;
  std::vector<bool> empty_prune;
  std::string empty_string;
  lm::builder::CorpusCount counter(f, vocab_file.get(), vocab_table.empty(), token_count, type_count, empty_prune, empty_string, chain.BlockSize() / chain.EntrySize(), lm::THROW_UP);
  chain >> boost::ref(counter);

  util::stream::SortConfig sort_config;
  sort_config.temp_prefix = temp_prefix;
  sort_config.buffer_size = 64 * 1024 * 1024;
  // Intended to run in parallel.
  sort_config.total_memory = remaining_size;
  util::stream::Sort<lm::SuffixOrder, lm::builder::CombineCounts> sorted(chain, sort_config, lm::SuffixOrder(order), lm::builder::CombineCounts());
  chain.Wait(true);
  util::stream::Chain chain2(util::stream::ChainConfig(lm::NGram<uint64_t>::TotalSize(order), blocks, sort_config.buffer_size));
  sorted.Output(chain2);
  // Inefficiently copies if there's only one block.
  chain2 >> util::stream::WriteAndRecycle(1);
  chain2.Wait(true);
  return 0;
}
