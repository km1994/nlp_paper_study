#include "lm/filter/arpa_io.hh"
#include "util/file_piece.hh"
#include "util/string_stream.hh"

#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <cctype>
#include <cerrno>
#include <cstring>

namespace lm {

ARPAInputException::ARPAInputException(const StringPiece &message) throw() {
  *this << message;
}

ARPAInputException::ARPAInputException(const StringPiece &message, const StringPiece &line) throw() {
  *this << message << " in line " << line;
}

ARPAInputException::~ARPAInputException() throw() {}

// Seeking is the responsibility of the caller.
template <class Stream> void WriteCounts(Stream &out, const std::vector<uint64_t> &number) {
  out << "\n\\data\\\n";
  for (unsigned int i = 0; i < number.size(); ++i) {
    out << "ngram " << i+1 << "=" << number[i] << '\n';
  }
  out << '\n';
}

size_t SizeNeededForCounts(const std::vector<uint64_t> &number) {
  util::StringStream stream;
  WriteCounts(stream, number);
  return stream.str().size();
}

bool IsEntirelyWhiteSpace(const StringPiece &line) {
  for (size_t i = 0; i < static_cast<size_t>(line.size()); ++i) {
    if (!isspace(line.data()[i])) return false;
  }
  return true;
}

ARPAOutput::ARPAOutput(const char *name, size_t buffer_size)
  : file_backing_(util::CreateOrThrow(name)), file_(file_backing_.get(), buffer_size) {}

void ARPAOutput::ReserveForCounts(std::streampos reserve) {
  for (std::streampos i = 0; i < reserve; i += std::streampos(1)) {
    file_ << '\n';
  }
}

void ARPAOutput::BeginLength(unsigned int length) {
  file_ << '\\' << length << "-grams:" << '\n';
  fast_counter_ = 0;
}

void ARPAOutput::EndLength(unsigned int length) {
  file_ << '\n';
  if (length > counts_.size()) {
    counts_.resize(length);
  }
  counts_[length - 1] = fast_counter_;
}

void ARPAOutput::Finish() {
  file_ << "\\end\\\n";
  file_.seekp(0);
  WriteCounts(file_, counts_);
  file_.flush();
}

} // namespace lm
