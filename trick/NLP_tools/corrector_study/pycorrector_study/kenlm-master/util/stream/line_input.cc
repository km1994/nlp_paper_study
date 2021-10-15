#include "util/stream/line_input.hh"

#include "util/exception.hh"
#include "util/file.hh"
#include "util/read_compressed.hh"
#include "util/stream/chain.hh"

#include <algorithm>
#include <vector>

namespace util { namespace stream {

void LineInput::Run(const ChainPosition &position) {
  ReadCompressed reader(fd_);
  // Holding area for beginning of line to be placed in next block.
  std::vector<char> carry;

  for (Link block(position); ; ++block) {
    char *to = static_cast<char*>(block->Get());
    char *begin = to;
    char *end = to + position.GetChain().BlockSize();
    std::copy(carry.begin(), carry.end(), to);
    to += carry.size();
    while (to != end) {
      std::size_t got = reader.Read(to, end - to);
      if (!got) {
        // EOF
        block->SetValidSize(to - begin);
        ++block;
        block.Poison();
        return;
      }
      to += got;
    }

    // Find the last newline.
    char *newline;
    for (newline = to - 1; ; --newline) {
      UTIL_THROW_IF(newline < begin, Exception, "Did not find a newline in " << position.GetChain().BlockSize() << " bytes of input of " << NameFromFD(fd_) << ".  Is this a text file?");
      if (*newline == '\n') break;
    }

    // Copy everything after the last newline to the carry.
    carry.clear();
    carry.resize(to - (newline + 1));
    std::copy(newline + 1, to, &*carry.begin());

    block->SetValidSize(newline + 1 - begin);
  }
}

}} // namespaces
