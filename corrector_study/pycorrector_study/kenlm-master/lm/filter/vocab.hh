#ifndef LM_FILTER_VOCAB_H
#define LM_FILTER_VOCAB_H

// Vocabulary-based filters for language models.

#include "util/multi_intersection.hh"
#include "util/string_piece.hh"
#include "util/string_piece_hash.hh"
#include "util/tokenize_piece.hh"

#include <boost/noncopyable.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <boost/unordered/unordered_set.hpp>

#include <string>
#include <vector>

namespace lm {
namespace vocab {

void ReadSingle(std::istream &in, boost::unordered_set<std::string> &out);

// Read one sentence vocabulary per line.  Return the number of sentences.
unsigned int ReadMultiple(std::istream &in, boost::unordered_map<std::string, std::vector<unsigned int> > &out);

/* Is this a special tag like <s> or <UNK>?  This actually includes anything
 * surrounded with < and >, which most tokenizers separate for real words, so
 * this should not catch real words as it looks at a single token.
 */
inline bool IsTag(const StringPiece &value) {
  // The parser should never give an empty string.
  assert(!value.empty());
  return (value.data()[0] == '<' && value.data()[value.size() - 1] == '>');
}

class Single {
  public:
    typedef boost::unordered_set<std::string> Words;

    explicit Single(const Words &vocab) : vocab_(vocab) {}

    template <class Iterator> bool PassNGram(const Iterator &begin, const Iterator &end) {
      for (Iterator i = begin; i != end; ++i) {
        if (IsTag(*i)) continue;
        if (FindStringPiece(vocab_, *i) == vocab_.end()) return false;
      }
      return true;
    }

  private:
    const Words &vocab_;
};

class Union {
  public:
    typedef boost::unordered_map<std::string, std::vector<unsigned int> > Words;

    explicit Union(const Words &vocabs) : vocabs_(vocabs) {}

    template <class Iterator> bool PassNGram(const Iterator &begin, const Iterator &end) {
      sets_.clear();

      for (Iterator i(begin); i != end; ++i) {
        if (IsTag(*i)) continue;
        Words::const_iterator found(FindStringPiece(vocabs_, *i));
        if (vocabs_.end() == found) return false;
        sets_.push_back(boost::iterator_range<const unsigned int*>(&*found->second.begin(), &*found->second.end()));
      }
      return (sets_.empty() || util::FirstIntersection(sets_));
    }

  private:
    const Words &vocabs_;

    std::vector<boost::iterator_range<const unsigned int*> > sets_;
};

class Multiple {
  public:
    typedef boost::unordered_map<std::string, std::vector<unsigned int> > Words;

    Multiple(const Words &vocabs) : vocabs_(vocabs) {}

  private:
    // Callback from AllIntersection that does AddNGram.
    template <class Output> class Callback {
      public:
        Callback(Output &out, const StringPiece &line) : out_(out), line_(line) {}

        void operator()(unsigned int index) {
          out_.SingleAddNGram(index, line_);
        }

      private:
        Output &out_;
        const StringPiece &line_;
    };

  public:
    template <class Iterator, class Output> void AddNGram(const Iterator &begin, const Iterator &end, const StringPiece &line, Output &output) {
      sets_.clear();
      for (Iterator i(begin); i != end; ++i) {
        if (IsTag(*i)) continue;
        Words::const_iterator found(FindStringPiece(vocabs_, *i));
        if (vocabs_.end() == found) return;
        sets_.push_back(boost::iterator_range<const unsigned int*>(&*found->second.begin(), &*found->second.end()));
      }
      if (sets_.empty()) {
        output.AddNGram(line);
        return;
      }

      Callback<Output> cb(output, line);
      util::AllIntersection(sets_, cb);
    }

    template <class Output> void AddNGram(const StringPiece &ngram, const StringPiece &line, Output &output) {
      AddNGram(util::TokenIter<util::SingleCharacter, true>(ngram, ' '), util::TokenIter<util::SingleCharacter, true>::end(), line, output);
    }

    void Flush() const {}

  private:
    const Words &vocabs_;

    std::vector<boost::iterator_range<const unsigned int*> > sets_;
};

} // namespace vocab
} // namespace lm

#endif // LM_FILTER_VOCAB_H
