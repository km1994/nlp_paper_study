#ifndef LM_FILTER_WRAPPER_H
#define LM_FILTER_WRAPPER_H

#include "util/string_piece.hh"

#include <algorithm>
#include <string>
#include <vector>

namespace lm {

// Provide a single-output filter with the same interface as a
// multiple-output filter so clients code against one interface.
template <class Binary> class BinaryFilter {
  public:
    // Binary modes are just references (and a set) and it makes the API cleaner to copy them.
    explicit BinaryFilter(Binary binary) : binary_(binary) {}

    template <class Iterator, class Output> void AddNGram(const Iterator &begin, const Iterator &end, const StringPiece &line, Output &output) {
      if (binary_.PassNGram(begin, end))
        output.AddNGram(line);
    }

    template <class Output> void AddNGram(const StringPiece &ngram, const StringPiece &line, Output &output) {
      AddNGram(util::TokenIter<util::SingleCharacter, true>(ngram, ' '), util::TokenIter<util::SingleCharacter, true>::end(), line, output);
    }

    void Flush() const {}

  private:
    Binary binary_;
};

// Wrap another filter to pay attention only to context words
template <class FilterT> class ContextFilter {
  public:
    typedef FilterT Filter;

    explicit ContextFilter(Filter &backend) : backend_(backend) {}

    template <class Output> void AddNGram(const StringPiece &ngram, const StringPiece &line, Output &output) {
      // Find beginning of string or last space.
      const char *last_space;
      for (last_space = ngram.data() + ngram.size() - 1; last_space > ngram.data() && *last_space != ' '; --last_space) {}
      backend_.AddNGram(StringPiece(ngram.data(), last_space - ngram.data()), line, output);
    }

    void Flush() const {}

  private:
    Filter backend_;
};

} // namespace lm

#endif // LM_FILTER_WRAPPER_H
