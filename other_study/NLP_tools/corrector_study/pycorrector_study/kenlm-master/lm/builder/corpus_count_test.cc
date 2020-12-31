#include "lm/builder/corpus_count.hh"

#include "lm/builder/payload.hh"
#include "lm/common/ngram_stream.hh"
#include "lm/common/ngram.hh"

#include "util/file.hh"
#include "util/file_piece.hh"
#include "util/tokenize_piece.hh"
#include "util/stream/chain.hh"
#include "util/stream/stream.hh"

#define BOOST_TEST_MODULE CorpusCountTest
#include <boost/test/unit_test.hpp>

namespace lm { namespace builder { namespace {

#define Check(str, cnt) { \
  BOOST_REQUIRE(stream); \
  w = stream->begin(); \
  for (util::TokenIter<util::AnyCharacter, true> t(str, " "); t; ++t, ++w) { \
    BOOST_CHECK_EQUAL(*t, v[*w]); \
  } \
  BOOST_CHECK_EQUAL((uint64_t)cnt, stream->Value().count); \
  ++stream; \
}

class CheckAnswers {
  public:
    void Run(const util::stream::ChainPosition &position) {
      NGramStream<BuildingPayload> stream(position);
      const char *v[] = {"<unk>", "<s>", "</s>", "looking", "on", "a", "little", "more", "loin", "foo", "bar"};
      WordIndex *w;

      Check("<s> <s> looking", 1);
      Check("<s> looking on", 1);
      Check("looking on a", 1);
      Check("on a little", 2);
      Check("a little more", 2);
      Check("little more loin", 2);
      Check("more loin </s>", 2);
      Check("<s> <s> on", 2);
      Check("<s> on a", 1);
      Check("<s> on foo", 1);
      Check("on foo little", 1);
      Check("foo little more", 1);
      Check("little more loin", 1);
      Check("more loin </s>", 1);
      Check("<s> <s> bar", 1);
      Check("<s> bar </s>", 1);
      Check("<s> <s> </s>", 1);
      BOOST_CHECK(!stream);
    }
};

BOOST_AUTO_TEST_CASE(Short) {
  util::scoped_fd input_file(util::MakeTemp("corpus_count_test_temp"));
  const char input[] = "looking on a little more loin\non a little more loin\non foo little more loin\nbar\n\n";
  // Blocks of 10 are
  // looking on a little more loin </s> on a little[duplicate] more[duplicate] loin[duplicate] </s>[duplicate] on[duplicate] foo
  // little more loin </s> bar </s> </s>

  util::WriteOrThrow(input_file.get(), input, sizeof(input) - 1);
  util::SeekOrThrow(input_file.get(), 0);
  util::FilePiece input_piece(input_file.release(), "temp file");

  util::stream::ChainConfig config;
  config.entry_size = NGram<BuildingPayload>::TotalSize(3);
  config.total_memory = config.entry_size * 20;
  config.block_count = 2;

  util::scoped_fd vocab(util::MakeTemp("corpus_count_test_vocab"));

  uint64_t token_count;
  WordIndex type_count = 10;
  std::vector<bool> prune_words;
  util::stream::Chain chain(config);
  CorpusCount counter(input_piece, vocab.get(), true, token_count, type_count, prune_words, "", chain.BlockSize() / chain.EntrySize(), SILENT);
  chain >> boost::ref(counter) >> CheckAnswers() >> util::stream::kRecycle;

  chain.Wait();
  BOOST_CHECK_EQUAL(11, type_count);
}

}}} // namespaces
