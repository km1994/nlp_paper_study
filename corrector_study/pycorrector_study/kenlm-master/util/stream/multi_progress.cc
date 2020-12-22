#include "util/stream/multi_progress.hh"

// TODO: merge some functionality with the simple progress bar?
#include "util/ersatz_progress.hh"

#include <iostream>
#include <limits>

#include <cstring>

#if !defined(_WIN32) && !defined(_WIN64)
#include <unistd.h>
#endif

namespace util { namespace stream {

namespace {
const char kDisplayCharacters[] = "-+*#0123456789";

uint64_t Next(unsigned char stone, uint64_t complete) {
  return (static_cast<uint64_t>(stone + 1) * complete + MultiProgress::kWidth - 1) / MultiProgress::kWidth;
}

} // namespace

MultiProgress::MultiProgress() : active_(false), complete_(std::numeric_limits<uint64_t>::max()), character_handout_(0) {}

MultiProgress::~MultiProgress() {
  if (active_ && complete_ != std::numeric_limits<uint64_t>::max())
    std::cerr << '\n';
}

void MultiProgress::Activate() {
  active_ =
#if !defined(_WIN32) && !defined(_WIN64)
    // Is stderr a terminal?
    (isatty(2) == 1)
#else
    true
#endif
    ;
}

void MultiProgress::SetTarget(uint64_t complete) {
  if (!active_) return;
  complete_ = complete;
  if (!complete) complete_ = 1;
  memset(display_, 0, sizeof(display_));
  character_handout_ = 0;
  std::cerr << kProgressBanner;
}

WorkerProgress MultiProgress::Add() {
  if (!active_)
    return WorkerProgress(std::numeric_limits<uint64_t>::max(), *this, '\0');
  std::size_t character_index;
  {
    boost::unique_lock<boost::mutex> lock(mutex_);
    character_index = character_handout_++;
    if (character_handout_ == sizeof(kDisplayCharacters) - 1)
      character_handout_ = 0;
  }
  return WorkerProgress(Next(0, complete_), *this, kDisplayCharacters[character_index]);
}

void MultiProgress::Finished() {
  if (!active_ || complete_ == std::numeric_limits<uint64_t>::max()) return;
  std::cerr << '\n';
  complete_ = std::numeric_limits<uint64_t>::max();
}

void MultiProgress::Milestone(WorkerProgress &worker) {
  if (!active_ || complete_ == std::numeric_limits<uint64_t>::max()) return;
  unsigned char stone = std::min(static_cast<uint64_t>(kWidth), worker.current_ * kWidth / complete_);
  for (char *i = &display_[worker.stone_]; i < &display_[stone]; ++i) {
    *i = worker.character_;
  }
  worker.next_ = Next(stone, complete_);
  worker.stone_ = stone;
  {
    boost::unique_lock<boost::mutex> lock(mutex_);
    std::cerr << '\r' << display_ << std::flush;
  }
}

}} // namespaces
