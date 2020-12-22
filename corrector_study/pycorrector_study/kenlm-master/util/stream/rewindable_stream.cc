#include "util/stream/rewindable_stream.hh"
#include "util/pcqueue.hh"

#include <iostream>

namespace util {
namespace stream {

RewindableStream::RewindableStream()
    : current_(NULL), in_(NULL), out_(NULL), poisoned_(true) {
  // nothing
}

void RewindableStream::Init(const ChainPosition &position) {
  UTIL_THROW_IF2(in_, "RewindableStream::Init twice");
  in_ = position.in_;
  out_ = position.out_;
  hit_poison_ = false;
  poisoned_ = false;
  progress_ = position.progress_;
  entry_size_ = position.GetChain().EntrySize();
  block_size_ = position.GetChain().BlockSize();
  block_count_ = position.GetChain().BlockCount();
  blocks_it_ = 0;
  marked_ = NULL;
  UTIL_THROW_IF2(block_count_ < 2, "RewindableStream needs block_count at least two");
  AppendBlock();
}

RewindableStream &RewindableStream::operator++() {
  assert(*this);
  assert(current_ < block_end_);
  assert(current_);
  assert(blocks_it_ < blocks_.size());
  current_ += entry_size_;
  if (UTIL_UNLIKELY(current_ == block_end_)) {
    // Fetch another block if necessary.
    if (++blocks_it_ == blocks_.size()) {
      if (!marked_) {
        Flush(blocks_.begin() + blocks_it_);
        blocks_it_ = 0;
      }
      AppendBlock();
      assert(poisoned_ || (blocks_it_ == blocks_.size() - 1));
      if (poisoned_) return *this;
    }
    Block &cur_block = blocks_[blocks_it_];
    current_ = static_cast<uint8_t*>(cur_block.Get());
    block_end_ = current_ + cur_block.ValidSize();
  }
  assert(current_);
  assert(current_ >= static_cast<uint8_t*>(blocks_[blocks_it_].Get()));
  assert(current_ < block_end_);
  assert(block_end_ == blocks_[blocks_it_].ValidEnd());
  return *this;
}

void RewindableStream::Mark() {
  marked_ = current_;
  Flush(blocks_.begin() + blocks_it_);
  blocks_it_ = 0;
}

void RewindableStream::Rewind() {
  if (current_ != marked_) {
    poisoned_ = false;
  }
  blocks_it_ = 0;
  current_ = marked_;
  block_end_ = static_cast<const uint8_t*>(blocks_[blocks_it_].ValidEnd());

  assert(current_);
  assert(current_ >= static_cast<uint8_t*>(blocks_[blocks_it_].Get()));
  assert(current_ < block_end_);
  assert(block_end_ == blocks_[blocks_it_].ValidEnd());
}

void RewindableStream::Poison() {
  if (blocks_.empty()) return;
  assert(*this);
  assert(blocks_it_ == blocks_.size() - 1);

  // Produce all buffered blocks.
  blocks_.back().SetValidSize(current_ - static_cast<uint8_t*>(blocks_.back().Get()));
  Flush(blocks_.end());
  blocks_it_ = 0;

  Block poison;
  if (!hit_poison_) {
    in_->Consume(poison);
  }
  poison.SetToPoison();
  out_->Produce(poison);
  hit_poison_ = true;
  poisoned_ = true;
}

void RewindableStream::AppendBlock() {
  if (UTIL_UNLIKELY(blocks_.size() >= block_count_)) {
    std::cerr << "RewindableStream trying to use more blocks than available" << std::endl;
    abort();
  }
  if (UTIL_UNLIKELY(hit_poison_)) {
    poisoned_ = true;
    return;
  }
  Block get;
  // The loop is needed since it is *feasible* that we're given 0 sized but
  // valid blocks
  do {
    in_->Consume(get);
    if (UTIL_LIKELY(get)) {
      blocks_.push_back(get);
    } else {
      hit_poison_ = true;
      poisoned_ = true;
      return;
    }
  } while (UTIL_UNLIKELY(get.ValidSize() == 0));
  current_ = static_cast<uint8_t*>(blocks_.back().Get());
  block_end_ = static_cast<const uint8_t*>(blocks_.back().ValidEnd());
  blocks_it_ = blocks_.size() - 1;
}

void RewindableStream::Flush(std::deque<Block>::iterator to) {
  for (std::deque<Block>::iterator i = blocks_.begin(); i != to; ++i) {
    out_->Produce(*i);
    progress_ += i->ValidSize();
  }
  blocks_.erase(blocks_.begin(), to);
}

}
}
