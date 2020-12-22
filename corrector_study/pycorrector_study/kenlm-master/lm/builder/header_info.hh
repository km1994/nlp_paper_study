#ifndef LM_BUILDER_HEADER_INFO_H
#define LM_BUILDER_HEADER_INFO_H

#include <string>
#include <vector>
#include <stdint.h>

namespace lm { namespace builder {

// Some configuration info that is used to add
// comments to the beginning of an ARPA file
struct HeaderInfo {
  std::string input_file;
  uint64_t token_count;
  std::vector<uint64_t> counts_pruned;

  HeaderInfo() {}

  HeaderInfo(const std::string& input_file_in, uint64_t token_count_in, const std::vector<uint64_t> &counts_pruned_in)
    : input_file(input_file_in), token_count(token_count_in), counts_pruned(counts_pruned_in) {}

  // TODO: Add smoothing type
  // TODO: More info if multiple models were interpolated
};

}} // namespaces

#endif
