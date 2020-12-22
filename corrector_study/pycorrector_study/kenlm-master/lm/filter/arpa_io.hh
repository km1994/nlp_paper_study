#ifndef LM_FILTER_ARPA_IO_H
#define LM_FILTER_ARPA_IO_H
/* Input and output for ARPA format language model files.
 */
#include "lm/read_arpa.hh"
#include "util/exception.hh"
#include "util/file_stream.hh"
#include "util/string_piece.hh"
#include "util/tokenize_piece.hh"

#include <boost/noncopyable.hpp>
#include <boost/scoped_array.hpp>

#include <fstream>
#include <string>
#include <vector>

#include <cstring>
#include <stdint.h>

namespace util { class FilePiece; }

namespace lm {

class ARPAInputException : public util::Exception {
  public:
    explicit ARPAInputException(const StringPiece &message) throw();
    explicit ARPAInputException(const StringPiece &message, const StringPiece &line) throw();
    virtual ~ARPAInputException() throw();
};

// Handling for the counts of n-grams at the beginning of ARPA files.
size_t SizeNeededForCounts(const std::vector<uint64_t> &number);

/* Writes an ARPA file.  This has to be seekable so the counts can be written
 * at the end.  Hence, I just have it own a std::fstream instead of accepting
 * a separately held std::ostream.  TODO: use the fast one from estimation.
 */
class ARPAOutput : boost::noncopyable {
  public:
    explicit ARPAOutput(const char *name, size_t buffer_size = 65536);

    void ReserveForCounts(std::streampos reserve);

    void BeginLength(unsigned int length);

    void AddNGram(const StringPiece &line) {
      file_ << line << '\n';
      ++fast_counter_;
    }

    void AddNGram(const StringPiece &ngram, const StringPiece &line) {
      AddNGram(line);
    }

    template <class Iterator> void AddNGram(const Iterator &begin, const Iterator &end, const StringPiece &line) {
      AddNGram(line);
    }

    void EndLength(unsigned int length);

    void Finish();

  private:
    util::scoped_fd file_backing_;
    util::FileStream file_;
    uint64_t fast_counter_;
    std::vector<uint64_t> counts_;
};


template <class Output> void ReadNGrams(util::FilePiece &in, unsigned int length, uint64_t number, Output &out) {
  ReadNGramHeader(in, length);
  out.BeginLength(length);
  for (uint64_t i = 0; i < number; ++i) {
    StringPiece line = in.ReadLine();
    util::TokenIter<util::SingleCharacter> tabber(line, '\t');
    if (!tabber) throw ARPAInputException("blank line", line);
    if (!++tabber) throw ARPAInputException("no tab", line);

    out.AddNGram(*tabber, line);
  }
  out.EndLength(length);
}

template <class Output> void ReadARPA(util::FilePiece &in_lm, Output &out) {
  std::vector<uint64_t> number;
  ReadARPACounts(in_lm, number);
  out.ReserveForCounts(SizeNeededForCounts(number));
  for (unsigned int i = 0; i < number.size(); ++i) {
    ReadNGrams(in_lm, i + 1, number[i], out);
  }
  ReadEnd(in_lm);
  out.Finish();
}

} // namespace lm

#endif // LM_FILTER_ARPA_IO_H
