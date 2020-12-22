#ifndef UTIL_STREAM_STREAM_H
#define UTIL_STREAM_STREAM_H

#include "util/stream/chain.hh"

#include <boost/noncopyable.hpp>

#include <cassert>
#include <stdint.h>

namespace util {
namespace stream {

class Stream : boost::noncopyable {
  public:
    Stream() : current_(NULL), end_(NULL) {}

    void Init(const ChainPosition &position) {
      entry_size_ = position.GetChain().EntrySize();
      block_size_ = position.GetChain().BlockSize();
      block_it_.Init(position);
      StartBlock();
    }

    explicit Stream(const ChainPosition &position) {
      Init(position);
    }

    operator bool() const { return current_ != NULL; }
    bool operator!() const { return current_ == NULL; }

    const void *Get() const { return current_; }
    void *Get() { return current_; }

    void Poison() {
      block_it_->SetValidSize(current_ - static_cast<uint8_t*>(block_it_->Get()));
      ++block_it_;
      block_it_.Poison();
    }

    Stream &operator++() {
      assert(*this);
      assert(current_ < end_);
      current_ += entry_size_;
      if (current_ == end_) {
        ++block_it_;
        StartBlock();
      }
      return *this;
    }

  private:
    void StartBlock() {
      for (; block_it_ && !block_it_->ValidSize(); ++block_it_) {}
      current_ = static_cast<uint8_t*>(block_it_->Get());
      end_ = current_ + block_it_->ValidSize();
    }

    // The following are pointers to raw memory
    // current_ is the current record
    // end_ is the end of the block (so we know when to move to the next block)
    uint8_t *current_, *end_;

    std::size_t entry_size_;
    std::size_t block_size_;

    Link block_it_;
};

inline Chain &operator>>(Chain &chain, Stream &stream) {
  stream.Init(chain.Add());
  return chain;
}

} // namespace stream
} // namespace util
#endif // UTIL_STREAM_STREAM_H
