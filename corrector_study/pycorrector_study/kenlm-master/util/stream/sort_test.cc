#include "util/stream/sort.hh"

#define BOOST_TEST_MODULE SortTest
#include <boost/test/unit_test.hpp>

#include <algorithm>

#include <unistd.h>

namespace util { namespace stream { namespace {

struct CompareUInt64 : public std::binary_function<const void *, const void *, bool> {
  bool operator()(const void *first, const void *second) const {
    return *static_cast<const uint64_t*>(first) < *reinterpret_cast<const uint64_t*>(second);
  }
};

const uint64_t kSize = 100000;

struct Putter {
  Putter(std::vector<uint64_t> &shuffled) : shuffled_(shuffled) {}

  void Run(const ChainPosition &position) {
    Stream put_shuffled(position);
    for (uint64_t i = 0; i < shuffled_.size(); ++i, ++put_shuffled) {
      *static_cast<uint64_t*>(put_shuffled.Get()) = shuffled_[i];
    }
    put_shuffled.Poison();
  }
  std::vector<uint64_t> &shuffled_;
};

BOOST_AUTO_TEST_CASE(FromShuffled) {
  std::vector<uint64_t> shuffled;
  shuffled.reserve(kSize);
  for (uint64_t i = 0; i < kSize; ++i) {
    shuffled.push_back(i);
  }
  std::random_shuffle(shuffled.begin(), shuffled.end());

  ChainConfig config;
  config.entry_size = 8;
  config.total_memory = 800;
  config.block_count = 3;

  SortConfig merge_config;
  merge_config.temp_prefix = "sort_test_temp";
  merge_config.buffer_size = 800;
  merge_config.total_memory = 3300;

  Chain chain(config);
  chain >> Putter(shuffled);
  BlockingSort(chain, merge_config, CompareUInt64(), NeverCombine());
  Stream sorted;
  chain >> sorted >> kRecycle;
  for (uint64_t i = 0; i < kSize; ++i, ++sorted) {
    BOOST_CHECK_EQUAL(i, *static_cast<const uint64_t*>(sorted.Get()));
  }
  BOOST_CHECK(!sorted);
}

}}} // namespaces
