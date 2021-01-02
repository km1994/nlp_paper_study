/* Progress bar suitable for chains of workers */
#ifndef UTIL_STREAM_MULTI_PROGRESS_H
#define UTIL_STREAM_MULTI_PROGRESS_H

#include <boost/thread/mutex.hpp>

#include <cstddef>
#include <stdint.h>

namespace util { namespace stream {

class WorkerProgress;

class MultiProgress {
  public:
    static const unsigned char kWidth = 100;

    MultiProgress();

    ~MultiProgress();

    // Turns on showing (requires SetTarget too).
    void Activate();

    void SetTarget(uint64_t complete);

    WorkerProgress Add();

    void Finished();

  private:
    friend class WorkerProgress;
    void Milestone(WorkerProgress &worker);

    bool active_;

    uint64_t complete_;

    boost::mutex mutex_;

    // \0 at the end.
    char display_[kWidth + 1];

    std::size_t character_handout_;

    MultiProgress(const MultiProgress &);
    MultiProgress &operator=(const MultiProgress &);
};

class WorkerProgress {
  public:
    // Default contrutor must be initialized with operator= later.
    WorkerProgress() : parent_(NULL) {}

    // Not threadsafe for the same worker by default.
    WorkerProgress &operator++() {
      if (++current_ >= next_) {
        parent_->Milestone(*this);
      }
      return *this;
    }

    WorkerProgress &operator+=(uint64_t amount) {
      current_ += amount;
      if (current_ >= next_) {
        parent_->Milestone(*this);
      }
      return *this;
    }

  private:
    friend class MultiProgress;
    WorkerProgress(uint64_t next, MultiProgress &parent, char character)
      : current_(0), next_(next), parent_(&parent), stone_(0), character_(character) {}

    uint64_t current_, next_;

    MultiProgress *parent_;

    // Previous milestone reached.
    unsigned char stone_;

    // Character to display in bar.
    char character_;
};

}} // namespaces

#endif // UTIL_STREAM_MULTI_PROGRESS_H
