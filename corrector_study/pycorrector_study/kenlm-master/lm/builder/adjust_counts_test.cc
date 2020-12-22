#include "lm/builder/adjust_counts.hh"

#include "lm/common/ngram_stream.hh"
#include "lm/builder/payload.hh"
#include "util/scoped.hh"

#include <boost/thread/thread.hpp>
#define BOOST_TEST_MODULE AdjustCounts
#include <boost/test/unit_test.hpp>

namespace lm { namespace builder { namespace {

class KeepCopy {
  public:
    KeepCopy() : size_(0) {}

    void Run(const util::stream::ChainPosition &position) {
      for (util::stream::Link link(position); link; ++link) {
        mem_.call_realloc(size_ + link->ValidSize());
        memcpy(static_cast<uint8_t*>(mem_.get()) + size_, link->Get(), link->ValidSize());
        size_ += link->ValidSize();
      }
    }

    uint8_t *Get() { return static_cast<uint8_t*>(mem_.get()); }
    std::size_t Size() const { return size_; }

  private:
    util::scoped_malloc mem_;
    std::size_t size_;
};

struct Gram4 {
  WordIndex ids[4];
  uint64_t count;
};

class WriteInput {
  public:
    void Run(const util::stream::ChainPosition &position) {
      NGramStream<BuildingPayload> input(position);
      Gram4 grams[] = {
        {{0,0,0,0},10},
        {{0,0,3,0},3},
        // bos
        {{1,1,1,2},5},
        {{0,0,3,2},5},
      };
      for (size_t i = 0; i < sizeof(grams) / sizeof(Gram4); ++i, ++input) {
        memcpy(input->begin(), grams[i].ids, sizeof(WordIndex) * 4);
        input->Value().count = grams[i].count;
      }
      input.Poison();
    }
};

BOOST_AUTO_TEST_CASE(Simple) {
  KeepCopy outputs[4];
  std::vector<uint64_t> counts;
  std::vector<Discount> discount;
  {
    util::stream::ChainConfig config;
    config.total_memory = 100;
    config.block_count = 1;
    util::stream::Chains chains(4);
    for (unsigned i = 0; i < 4; ++i) {
      config.entry_size = NGram<BuildingPayload>::TotalSize(i + 1);
      chains.push_back(config);
    }

    chains[3] >> WriteInput();
    util::stream::ChainPositions for_adjust(chains);
    for (unsigned i = 0; i < 4; ++i) {
      chains[i] >> boost::ref(outputs[i]);
    }
    chains >> util::stream::kRecycle;
    std::vector<uint64_t> counts_pruned(4);
    std::vector<uint64_t> prune_thresholds(4);
    DiscountConfig discount_config;
    discount_config.fallback = Discount();
    discount_config.bad_action = THROW_UP;
    BOOST_CHECK_THROW(AdjustCounts(prune_thresholds, counts, counts_pruned, std::vector<bool>(), discount_config, discount).Run(for_adjust), BadDiscountException);
  }
  BOOST_REQUIRE_EQUAL(4UL, counts.size());
  BOOST_CHECK_EQUAL(4UL, counts[0]);
  // These are no longer set because the discounts are bad.
/*  BOOST_CHECK_EQUAL(4UL, counts[1]);
  BOOST_CHECK_EQUAL(3UL, counts[2]);
  BOOST_CHECK_EQUAL(3UL, counts[3]);*/
  BOOST_REQUIRE_EQUAL(NGram<BuildingPayload>::TotalSize(1) * 4, outputs[0].Size());
  NGram<BuildingPayload> uni(outputs[0].Get(), 1);
  BOOST_CHECK_EQUAL(kUNK, *uni.begin());
  BOOST_CHECK_EQUAL(0ULL, uni.Value().count);
  uni.NextInMemory();
  BOOST_CHECK_EQUAL(kBOS, *uni.begin());
  BOOST_CHECK_EQUAL(0ULL, uni.Value().count);
  uni.NextInMemory();
  BOOST_CHECK_EQUAL(0UL, *uni.begin());
  BOOST_CHECK_EQUAL(2ULL, uni.Value().count);
  uni.NextInMemory();
  BOOST_CHECK_EQUAL(2ULL, uni.Value().count);
  BOOST_CHECK_EQUAL(2UL, *uni.begin());

  BOOST_REQUIRE_EQUAL(NGram<BuildingPayload>::TotalSize(2) * 4, outputs[1].Size());
  NGram<BuildingPayload> bi(outputs[1].Get(), 2);
  BOOST_CHECK_EQUAL(0UL, *bi.begin());
  BOOST_CHECK_EQUAL(0UL, *(bi.begin() + 1));
  BOOST_CHECK_EQUAL(1ULL, bi.Value().count);
  bi.NextInMemory();
}

}}} // namespaces
