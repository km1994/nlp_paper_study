#include "lm/builder/corpus_count.hh"

#include "lm/builder/payload.hh"
#include "lm/common/ngram.hh"
#include "lm/lm_exception.hh"
#include "lm/vocab.hh"
#include "lm/word_index.hh"
#include "util/file_stream.hh"
#include "util/file.hh"
#include "util/file_piece.hh"
#include "util/murmur_hash.hh"
#include "util/probing_hash_table.hh"
#include "util/scoped.hh"
#include "util/stream/chain.hh"
#include "util/tokenize_piece.hh"

#include <functional>

#include <stdint.h>

namespace lm {
namespace builder {
namespace {

class DedupeHash : public std::unary_function<const WordIndex *, bool> {
  public:
    explicit DedupeHash(std::size_t order) : size_(order * sizeof(WordIndex)) {}

    std::size_t operator()(const WordIndex *start) const {
      return util::MurmurHashNative(start, size_);
    }

  private:
    const std::size_t size_;
};

class DedupeEquals : public std::binary_function<const WordIndex *, const WordIndex *, bool> {
  public:
    explicit DedupeEquals(std::size_t order) : size_(order * sizeof(WordIndex)) {}

    bool operator()(const WordIndex *first, const WordIndex *second) const {
      return !memcmp(first, second, size_);
    }

  private:
    const std::size_t size_;
};

struct DedupeEntry {
  typedef WordIndex *Key;
  Key GetKey() const { return key; }
  void SetKey(WordIndex *to) { key = to; }
  Key key;
  static DedupeEntry Construct(WordIndex *at) {
    DedupeEntry ret;
    ret.key = at;
    return ret;
  }
};


// TODO: don't have this here, should be with probing hash table defaults?
const float kProbingMultiplier = 1.5;

typedef util::ProbingHashTable<DedupeEntry, DedupeHash, DedupeEquals> Dedupe;

class Writer {
  public:
    Writer(std::size_t order, const util::stream::ChainPosition &position, void *dedupe_mem, std::size_t dedupe_mem_size)
      : block_(position), gram_(block_->Get(), order),
        dedupe_invalid_(order, std::numeric_limits<WordIndex>::max()),
        dedupe_(dedupe_mem, dedupe_mem_size, &dedupe_invalid_[0], DedupeHash(order), DedupeEquals(order)),
        buffer_(new WordIndex[order - 1]),
        block_size_(position.GetChain().BlockSize()) {
      dedupe_.Clear();
      assert(Dedupe::Size(position.GetChain().BlockSize() / position.GetChain().EntrySize(), kProbingMultiplier) == dedupe_mem_size);
      if (order == 1) {
        // Add special words.  AdjustCounts is responsible if order != 1.
        AddUnigramWord(kUNK);
        AddUnigramWord(kBOS);
      }
    }

    ~Writer() {
      block_->SetValidSize(reinterpret_cast<const uint8_t*>(gram_.begin()) - static_cast<const uint8_t*>(block_->Get()));
      (++block_).Poison();
    }

    // Write context with a bunch of <s>
    void StartSentence() {
      for (WordIndex *i = gram_.begin(); i != gram_.end() - 1; ++i) {
        *i = kBOS;
      }
    }

    void Append(WordIndex word) {
      *(gram_.end() - 1) = word;
      Dedupe::MutableIterator at;
      bool found = dedupe_.FindOrInsert(DedupeEntry::Construct(gram_.begin()), at);
      if (found) {
        // Already present.
        NGram<BuildingPayload> already(at->key, gram_.Order());
        ++(already.Value().count);
        // Shift left by one.
        memmove(gram_.begin(), gram_.begin() + 1, sizeof(WordIndex) * (gram_.Order() - 1));
        return;
      }
      // Complete the write.
      gram_.Value().count = 1;
      // Prepare the next n-gram.
      if (reinterpret_cast<uint8_t*>(gram_.begin()) + gram_.TotalSize() != static_cast<uint8_t*>(block_->Get()) + block_size_) {
        NGram<BuildingPayload> last(gram_);
        gram_.NextInMemory();
        std::copy(last.begin() + 1, last.end(), gram_.begin());
        return;
      }
      // Block end.  Need to store the context in a temporary buffer.
      std::copy(gram_.begin() + 1, gram_.end(), buffer_.get());
      dedupe_.Clear();
      block_->SetValidSize(block_size_);
      gram_.ReBase((++block_)->Get());
      std::copy(buffer_.get(), buffer_.get() + gram_.Order() - 1, gram_.begin());
    }

  private:
    void AddUnigramWord(WordIndex index) {
      *gram_.begin() = index;
      gram_.Value().count = 0;
      gram_.NextInMemory();
      if (gram_.Base() == static_cast<uint8_t*>(block_->Get()) + block_size_) {
        block_->SetValidSize(block_size_);
        gram_.ReBase((++block_)->Get());
      }
    }

    util::stream::Link block_;

    NGram<BuildingPayload> gram_;

    // This is the memory behind the invalid value in dedupe_.
    std::vector<WordIndex> dedupe_invalid_;
    // Hash table combiner implementation.
    Dedupe dedupe_;

    // Small buffer to hold existing ngrams when shifting across a block boundary.
    boost::scoped_array<WordIndex> buffer_;

    const std::size_t block_size_;
};

} // namespace

float CorpusCount::DedupeMultiplier(std::size_t order) {
  return kProbingMultiplier * static_cast<float>(sizeof(DedupeEntry)) / static_cast<float>(NGram<BuildingPayload>::TotalSize(order));
}

std::size_t CorpusCount::VocabUsage(std::size_t vocab_estimate) {
  return ngram::GrowableVocab<ngram::WriteUniqueWords>::MemUsage(vocab_estimate);
}

CorpusCount::CorpusCount(util::FilePiece &from, int vocab_write, bool dynamic_vocab, uint64_t &token_count, WordIndex &type_count, std::vector<bool> &prune_words, const std::string& prune_vocab_filename, std::size_t entries_per_block, WarningAction disallowed_symbol)
  : from_(from), vocab_write_(vocab_write), dynamic_vocab_(dynamic_vocab), token_count_(token_count), type_count_(type_count),
    prune_words_(prune_words), prune_vocab_filename_(prune_vocab_filename),
    dedupe_mem_size_(Dedupe::Size(entries_per_block, kProbingMultiplier)),
    dedupe_mem_(util::MallocOrThrow(dedupe_mem_size_)),
    disallowed_symbol_action_(disallowed_symbol) {
}

namespace {
void ComplainDisallowed(StringPiece word, WarningAction &action) {
  switch (action) {
    case SILENT:
      return;
    case COMPLAIN:
      std::cerr << "Warning: " << word << " appears in the input.  All instances of <s>, </s>, and <unk> will be interpreted as whitespace." << std::endl;
      action = SILENT;
      return;
    case THROW_UP:
      UTIL_THROW(FormatLoadException, "Special word " << word << " is not allowed in the corpus.  I plan to support models containing <unk> in the future.  Pass --skip_symbols to convert these symbols to whitespace.");
  }
}

// Vocab ids are given in a precompiled hash table.
class VocabGiven {
  public:
    explicit VocabGiven(int fd) {
      util::MapRead(util::POPULATE_OR_READ, fd, 0, util::CheckOverflow(util::SizeOrThrow(fd)), table_backing_);
      // Leave space for header with size.
      table_ = Table(static_cast<char*>(table_backing_.get()) + sizeof(uint64_t), table_backing_.size() - sizeof(uint64_t));
      bos_ = FindOrInsert("<s>");
      eos_ = FindOrInsert("</s>");
    }

    WordIndex FindOrInsert(const StringPiece &word) const {
      Table::ConstIterator it;
      if (table_.Find(util::MurmurHash64A(word.data(), word.size()), it)) {
        return it->value;
      } else {
        return 0; // <unk>.
      }
    }

    WordIndex Index(const StringPiece &word) const {
      return FindOrInsert(word);
    }

    WordIndex Size() const {
      return *static_cast<const uint64_t*>(table_backing_.get());
    }

    bool IsSpecial(WordIndex word) const {
      return word == 0 || word == bos_ || word == eos_;
    }

  private:
    util::scoped_memory table_backing_;

    typedef util::ProbingHashTable<ngram::ProbingVocabularyEntry, util::IdentityHash> Table;
    Table table_;

    WordIndex bos_, eos_;
};
} // namespace

void CorpusCount::Run(const util::stream::ChainPosition &position) {
  if (dynamic_vocab_) {
    ngram::GrowableVocab<ngram::WriteUniqueWords> vocab(type_count_, vocab_write_);
    RunWithVocab(position, vocab);
  } else {
    VocabGiven vocab(vocab_write_);
    RunWithVocab(position, vocab);
  }
}

template <class Vocab> void CorpusCount::RunWithVocab(const util::stream::ChainPosition &position, Vocab &vocab) {
  token_count_ = 0;
  type_count_ = 0;
  const WordIndex end_sentence = vocab.FindOrInsert("</s>");
  Writer writer(NGram<BuildingPayload>::OrderFromSize(position.GetChain().EntrySize()), position, dedupe_mem_.get(), dedupe_mem_size_);
  uint64_t count = 0;
  bool delimiters[256];
  util::BoolCharacter::Build("\0\t\n\r ", delimiters);
  StringPiece w;
  while(true) {
    writer.StartSentence();
    while (from_.ReadWordSameLine(w, delimiters)) {
      WordIndex word = vocab.FindOrInsert(w);
      if (UTIL_UNLIKELY(vocab.IsSpecial(word))) {
        ComplainDisallowed(w, disallowed_symbol_action_);
        continue;
      }
      writer.Append(word);
      ++count;
    }
    if (!from_.ReadLineOrEOF(w)) break;
    writer.Append(end_sentence);
  }
  token_count_ = count;
  type_count_ = vocab.Size();

  // Create list of unigrams that are supposed to be pruned
  if (!prune_vocab_filename_.empty()) {
    try {
      util::FilePiece prune_vocab_file(prune_vocab_filename_.c_str());

      prune_words_.resize(vocab.Size(), true);
      try {
        while (true) {
          StringPiece word(prune_vocab_file.ReadDelimited(delimiters));
          prune_words_[vocab.Index(word)] = false;
        }
      } catch (const util::EndOfFileException &e) {}

      // Never prune <unk>, <s>, </s>
      prune_words_[kUNK] = false;
      prune_words_[kBOS] = false;
      prune_words_[kEOS] = false;

    } catch (const util::Exception &e) {
      std::cerr << e.what() << std::endl;
      abort();
    }
  }
}

} // namespace builder
} // namespace lm
