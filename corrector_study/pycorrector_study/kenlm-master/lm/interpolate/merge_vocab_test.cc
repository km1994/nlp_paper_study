#define BOOST_TEST_MODULE InterpolateMergeVocabTest
#include <boost/test/unit_test.hpp>

#include "lm/enumerate_vocab.hh"
#include "lm/interpolate/merge_vocab.hh"
#include "lm/interpolate/universal_vocab.hh"
#include "lm/lm_exception.hh"
#include "lm/vocab.hh"
#include "lm/word_index.hh"
#include "util/file.hh"
#include "util/file_piece.hh"
#include "util/file_stream.hh"
#include "util/tokenize_piece.hh"

#include <algorithm>
#include <cstring>
#include <vector>

namespace lm {
namespace interpolate {
namespace {

struct VocabEntry {
  explicit VocabEntry(StringPiece value) :
    str(value), hash(util::MurmurHash64A(value.data(), value.size())) {}
  StringPiece str;
  uint64_t hash;
  bool operator<(const VocabEntry &other) const {
    return hash < other.hash;
  }
};

int WriteVocabFile(const std::vector<VocabEntry> &vocab, util::scoped_fd &file) {
  file.reset(util::MakeTemp(util::DefaultTempDirectory()));
  {
    util::FileStream out(file.get(), 128);
    for (std::vector<VocabEntry>::const_iterator i = vocab.begin(); i != vocab.end(); ++i) {
      out << i->str << '\0';
    }
  }
  util::SeekOrThrow(file.get(), 0);
  return file.get();
}

std::vector<VocabEntry> ParseVocab(StringPiece words) {
  std::vector<VocabEntry> entries;
  entries.push_back(VocabEntry("<unk>"));
  for (util::TokenIter<util::SingleCharacter> i(words, '\t'); i; ++i) {
    entries.push_back(VocabEntry(*i));
  }
  std::sort(entries.begin() + 1, entries.end());
  return entries;
}

int WriteVocabFile(StringPiece words, util::scoped_fd &file) {
  return WriteVocabFile(ParseVocab(words), file);
}

class TestFiles {
  public:
    TestFiles() {}
    int Test0() {
      return WriteVocabFile("this\tis\ta\tfirst\tcut", test[0]);
    }
    int Test1() {
      return WriteVocabFile("is this\tthis a\tfirst cut\ta first", test[1]);
    }
    int Test2() {
      return WriteVocabFile("is\tsecd\ti", test[2]);
    }
    int NoUNK() {
      std::vector<VocabEntry> no_unk_vec;
      no_unk_vec.push_back(VocabEntry("toto"));
      return WriteVocabFile(no_unk_vec, no_unk);
    }
    int BadOrder() {
      std::vector<VocabEntry> bad_order_vec;
      bad_order_vec.push_back(VocabEntry("<unk>"));
      bad_order_vec.push_back(VocabEntry("0"));
      bad_order_vec.push_back(VocabEntry("1"));
      bad_order_vec.push_back(VocabEntry("2"));
      bad_order_vec.push_back(VocabEntry("a"));
      return WriteVocabFile(bad_order_vec, bad_order);
    }
  private:
    util::scoped_fd test[3], no_unk, bad_order;
};

class DoNothingEnumerate : public EnumerateVocab {
  public:
    void Add(WordIndex, const StringPiece &) {}
};

BOOST_AUTO_TEST_CASE(MergeVocabTest) {
  TestFiles files;

  util::FixedArray<int> used_files(3);
  used_files.push_back(files.Test0());
  used_files.push_back(files.Test1());
  used_files.push_back(files.Test2());

  std::vector<lm::WordIndex> model_max_idx;
  model_max_idx.push_back(10);
  model_max_idx.push_back(10);
  model_max_idx.push_back(10);

  util::scoped_fd combined(util::MakeTemp(util::DefaultTempDirectory()));

  UniversalVocab universal_vocab(model_max_idx);
  {
    ngram::ImmediateWriteWordsWrapper writer(NULL, combined.get(), 0);
    MergeVocab(used_files, universal_vocab, writer);
  }

  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(0, 0), 0);
  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(1, 0), 0);
  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(2, 0), 0);
  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(0, 1), 1);
  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(1, 1), 2);
  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(2, 1), 8);
  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(0, 5), 11);
#if BYTE_ORDER == LITTLE_ENDIAN
  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(1, 3), 4);
#elif BYTE_ORDER == BIG_ENDIAN
  // MurmurHash has a different ordering of the vocabulary.
  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(1, 3), 5);
#endif
  BOOST_CHECK_EQUAL(universal_vocab.GetUniversalIdx(2, 3), 10);

  util::SeekOrThrow(combined.get(), 0);
  util::FilePiece f(combined.release());
  std::vector<VocabEntry> expected = ParseVocab("a\tis this\tthis a\tfirst cut\tthis\ta first\tcut\tis\ti\tsecd\tfirst");
  for (std::vector<VocabEntry>::const_iterator i = expected.begin(); i != expected.end(); ++i) {
    BOOST_CHECK_EQUAL(i->str, f.ReadLine('\0'));
  }
  BOOST_CHECK_THROW(f.ReadLine('\0'), util::EndOfFileException);
}

BOOST_AUTO_TEST_CASE(MergeVocabNoUnkTest) {
  TestFiles files;
  util::FixedArray<int> used_files(1);
  used_files.push_back(files.NoUNK());

  std::vector<lm::WordIndex> model_max_idx;
  model_max_idx.push_back(10);

  UniversalVocab universal_vocab(model_max_idx);
  DoNothingEnumerate nothing;
  BOOST_CHECK_THROW(MergeVocab(used_files, universal_vocab, nothing), FormatLoadException);
}

BOOST_AUTO_TEST_CASE(MergeVocabWrongOrderTest) {
  TestFiles files;

  util::FixedArray<int> used_files(2);
  used_files.push_back(files.Test0());
  used_files.push_back(files.BadOrder());

  std::vector<lm::WordIndex> model_max_idx;
  model_max_idx.push_back(10);
  model_max_idx.push_back(10);

  lm::interpolate::UniversalVocab universal_vocab(model_max_idx);
  DoNothingEnumerate nothing;
  BOOST_CHECK_THROW(MergeVocab(used_files, universal_vocab, nothing), FormatLoadException);
}

}}} // namespaces
