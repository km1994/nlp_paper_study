#include "util/file_stream.hh"
#include "util/file_piece.hh"
#include "util/murmur_hash.hh"
#include "util/pool.hh"
#include "util/string_piece.hh"
#include "util/string_piece_hash.hh"
#include "util/tokenize_piece.hh"

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <cstddef>
#include <vector>

namespace {

struct MutablePiece {
  mutable StringPiece behind;
  bool operator==(const MutablePiece &other) const {
    return behind == other.behind;
  }
};

std::size_t hash_value(const MutablePiece &m) {
  return hash_value(m.behind);
}

class InternString {
  public:
    const char *Add(StringPiece str) {
      MutablePiece mut;
      mut.behind = str;
      std::pair<boost::unordered_set<MutablePiece>::iterator, bool> res(strs_.insert(mut));
      if (res.second) {
        void *mem = backing_.Allocate(str.size() + 1);
        memcpy(mem, str.data(), str.size());
        static_cast<char*>(mem)[str.size()] = 0;
        res.first->behind = StringPiece(static_cast<char*>(mem), str.size());
      }
      return res.first->behind.data();
    }

  private:
    util::Pool backing_;
    boost::unordered_set<MutablePiece> strs_;
};

class TargetWords {
  public:
    void Introduce(StringPiece source) {
      vocab_.resize(vocab_.size() + 1);
      std::vector<unsigned int> temp(1, vocab_.size() - 1);
      Add(temp, source);
    }

    void Add(const std::vector<unsigned int> &sentences, StringPiece target) {
      if (sentences.empty()) return;
      interns_.clear();
      for (util::TokenIter<util::SingleCharacter, true> i(target, ' '); i; ++i) {
        interns_.push_back(intern_.Add(*i));
      }
      for (std::vector<unsigned int>::const_iterator i(sentences.begin()); i != sentences.end(); ++i) {
        boost::unordered_set<const char *> &vocab = vocab_[*i];
        for (std::vector<const char *>::const_iterator j = interns_.begin(); j != interns_.end(); ++j) {
          vocab.insert(*j);
        }
      }
    }

    void Print() const {
      util::FileStream out(1);
      for (std::vector<boost::unordered_set<const char *> >::const_iterator i = vocab_.begin(); i != vocab_.end(); ++i) {
        for (boost::unordered_set<const char *>::const_iterator j = i->begin(); j != i->end(); ++j) {
          out << *j << ' ';
        }
        out << '\n';
      }
    }

  private:
    InternString intern_;

    std::vector<boost::unordered_set<const char *> > vocab_;

    // Temporary in Add.
    std::vector<const char *> interns_;
};

class Input {
  public:
    explicit Input(std::size_t max_length)
      : max_length_(max_length), sentence_id_(0), empty_() {}

    void AddSentence(StringPiece sentence, TargetWords &targets) {
      canonical_.clear();
      starts_.clear();
      starts_.push_back(0);
      for (util::TokenIter<util::AnyCharacter, true> i(sentence, StringPiece("\0 \t", 3)); i; ++i) {
        canonical_.append(i->data(), i->size());
        canonical_ += ' ';
        starts_.push_back(canonical_.size());
      }
      targets.Introduce(canonical_);
      for (std::size_t i = 0; i < starts_.size() - 1; ++i) {
        std::size_t subtract = starts_[i];
        const char *start = &canonical_[subtract];
        for (std::size_t j = i + 1; j < std::min(starts_.size(), i + max_length_ + 1); ++j) {
          map_[util::MurmurHash64A(start, &canonical_[starts_[j]] - start - 1)].push_back(sentence_id_);
        }
      }
      ++sentence_id_;
    }

    // Assumes single space-delimited phrase with no space at the beginning or end.
    const std::vector<unsigned int> &Matches(StringPiece phrase) const {
      Map::const_iterator i = map_.find(util::MurmurHash64A(phrase.data(), phrase.size()));
      return i == map_.end() ? empty_ : i->second;
    }

  private:
    const std::size_t max_length_;

    // hash of phrase is the key, array of sentences is the value.
    typedef boost::unordered_map<uint64_t, std::vector<unsigned int> > Map;
    Map map_;

    std::size_t sentence_id_;

    // Temporaries in AddSentence.
    std::string canonical_;
    std::vector<std::size_t> starts_;

    const std::vector<unsigned int> empty_;
};

} // namespace

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Expected source text on the command line" << std::endl;
    return 1;
  }
  Input input(7);
  TargetWords targets;
  try {
    util::FilePiece inputs(argv[1], &std::cerr);
    while (true)
      input.AddSentence(inputs.ReadLine(), targets);
  } catch (const util::EndOfFileException &e) {}

  util::FilePiece table(0, NULL, &std::cerr);
  StringPiece line;
  const StringPiece pipes("|||");
  while (true) {
    try {
      line = table.ReadLine();
    } catch (const util::EndOfFileException &e) { break; }
    util::TokenIter<util::MultiCharacter> it(line, pipes);
    StringPiece source(*it);
    if (!source.empty() && source[source.size() - 1] == ' ')
      source.remove_suffix(1);
    targets.Add(input.Matches(source), *++it);
  }
  targets.Print();
}
