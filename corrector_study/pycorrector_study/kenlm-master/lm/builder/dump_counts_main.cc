#include "lm/common/print.hh"
#include "lm/word_index.hh"
#include "util/file.hh"
#include "util/read_compressed.hh"

#include <boost/lexical_cast.hpp>

#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " counts vocabulary order\n"
    "The counts file contains records with 4-byte vocabulary ids followed by 8-byte\n"
    "counts.  Each record has order many vocabulary ids.\n"
    "The vocabulary file contains the words delimited by NULL in order of id.\n"
    "The vocabulary file may not be compressed because it is mmapped but the counts\n"
    "file can be compressed.\n";
    return 1;
  }
  util::ReadCompressed counts(util::OpenReadOrThrow(argv[1]));
  util::scoped_fd vocab_file(util::OpenReadOrThrow(argv[2]));
  lm::VocabReconstitute vocab(vocab_file.get());
  unsigned int order = boost::lexical_cast<unsigned int>(argv[3]);
  std::vector<char> record(sizeof(uint32_t) * order + sizeof(uint64_t));
  while (std::size_t got = counts.ReadOrEOF(&*record.begin(), record.size())) {
    UTIL_THROW_IF(got != record.size(), util::Exception, "Read " << got << " bytes at the end of file, which is not a complete record of length " << record.size());
    const lm::WordIndex *words = reinterpret_cast<const lm::WordIndex*>(&*record.begin());
    for (const lm::WordIndex *i = words; i != words + order; ++i) {
      UTIL_THROW_IF(*i >= vocab.Size(), util::Exception, "Vocab ID " << *i << " is larger than the vocab file's maximum of " << vocab.Size() << ".  Are you sure you have the right order and vocab file for these counts?");
      std::cout << vocab.Lookup(*i) << ' ';
    }
    // TODO don't use std::cout because it is slow.  Add fast uint64_t printing support to FileStream.
    std::cout << *reinterpret_cast<const uint64_t*>(words + order) << '\n';
  }
}
