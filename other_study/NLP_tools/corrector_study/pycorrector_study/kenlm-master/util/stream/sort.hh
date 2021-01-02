/* Usage:
 * Sort<Compare> sorter(temp, compare);
 * Chain(config) >> Read(file) >> sorter.Unsorted();
 * Stream stream;
 * Chain chain(config) >> sorter.Sorted(internal_config, lazy_config) >> stream;
 *
 * Note that sorter must outlive any threads that use Unsorted or Sorted.
 *
 * Combiners take the form:
 * bool operator()(void *into, const void *option, const Compare &compare) const
 * which returns true iff a combination happened.  The sorting algorithm
 * guarantees compare(into, option).  But it does not guarantee
 * compare(option, into).
 * Currently, combining is only done in merge steps, not during on-the-fly
 * sort.  Use a hash table for that.
 */

#ifndef UTIL_STREAM_SORT_H
#define UTIL_STREAM_SORT_H

#include "util/stream/chain.hh"
#include "util/stream/config.hh"
#include "util/stream/io.hh"
#include "util/stream/stream.hh"

#include "util/file.hh"
#include "util/fixed_array.hh"
#include "util/scoped.hh"
#include "util/sized_iterator.hh"

#include <algorithm>
#include <iostream>
#include <queue>
#include <string>

namespace util {
namespace stream {

struct NeverCombine {
  template <class Compare> bool operator()(const void *, const void *, const Compare &) const {
    return false;
  }
};

// Manage the offsets of sorted blocks in a file.
class Offsets {
  public:
    explicit Offsets(int fd) : log_(fd) {
      Reset();
    }

    int File() const { return log_; }

    void Append(uint64_t length) {
      if (!length) return;
      ++block_count_;
      if (length == cur_.length) {
        ++cur_.run;
        return;
      }
      WriteOrThrow(log_, &cur_, sizeof(Entry));
      cur_.length = length;
      cur_.run = 1;
    }

    void FinishedAppending() {
      WriteOrThrow(log_, &cur_, sizeof(Entry));
      SeekOrThrow(log_, sizeof(Entry)); // Skip 0,0 at beginning.
      cur_.run = 0;
      if (block_count_) {
        ReadOrThrow(log_, &cur_, sizeof(Entry));
        assert(cur_.length);
        assert(cur_.run);
      }
    }

    uint64_t RemainingBlocks() const { return block_count_; }

    uint64_t TotalOffset() const { return output_sum_; }

    uint64_t PeekSize() const {
      return cur_.length;
    }

    uint64_t NextSize() {
      assert(block_count_);
      uint64_t ret = cur_.length;
      output_sum_ += ret;

      --cur_.run;
      --block_count_;
      if (!cur_.run && block_count_) {
        ReadOrThrow(log_, &cur_, sizeof(Entry));
        assert(cur_.length);
        assert(cur_.run);
      }
      return ret;
    }

    void Reset() {
      SeekOrThrow(log_, 0);
      ResizeOrThrow(log_, 0);
      cur_.length = 0;
      cur_.run = 0;
      block_count_ = 0;
      output_sum_ = 0;
    }

  private:
    int log_;

    struct Entry {
      uint64_t length;
      uint64_t run;
    };
    Entry cur_;

    uint64_t block_count_;

    uint64_t output_sum_;
};

// A priority queue of entries backed by file buffers
template <class Compare> class MergeQueue {
  public:
    MergeQueue(int fd, std::size_t buffer_size, std::size_t entry_size, const Compare &compare)
      : queue_(Greater(compare)), in_(fd), buffer_size_(buffer_size), entry_size_(entry_size) {}

    void Push(void *base, uint64_t offset, uint64_t amount) {
      queue_.push(Entry(base, in_, offset, amount, buffer_size_));
    }

    const void *Top() const {
      return queue_.top().Current();
    }

    void Pop() {
      Entry top(queue_.top());
      queue_.pop();
      if (top.Increment(in_, buffer_size_, entry_size_))
        queue_.push(top);
    }

    std::size_t Size() const {
      return queue_.size();
    }

    bool Empty() const {
      return queue_.empty();
    }

  private:
    // Priority queue contains these entries.
    class Entry {
      public:
        Entry() {}

        Entry(void *base, int fd, uint64_t offset, uint64_t amount, std::size_t buf_size) {
          offset_ = offset;
          remaining_ = amount;
          buffer_end_ = static_cast<uint8_t*>(base) + buf_size;
          Read(fd, buf_size);
        }

        bool Increment(int fd, std::size_t buf_size, std::size_t entry_size) {
          current_ += entry_size;
          if (current_ != buffer_end_) return true;
          return Read(fd, buf_size);
        }

        const void *Current() const { return current_; }

      private:
        bool Read(int fd, std::size_t buf_size) {
          current_ = buffer_end_ - buf_size;
          std::size_t amount;
          if (static_cast<uint64_t>(buf_size) < remaining_) {
            amount = buf_size;
          } else if (!remaining_) {
            return false;
          } else {
            amount = remaining_;
            buffer_end_ = current_ + remaining_;
          }
          ErsatzPRead(fd, current_, amount, offset_);
          // Try to free the space, but don't be disappointed if we can't.
          try {
            HolePunch(fd, offset_, amount);
          } catch (const util::Exception &) {}
          offset_ += amount;
          assert(current_ <= buffer_end_);
          remaining_ -= amount;
          return true;
        }

        // Buffer
        uint8_t *current_, *buffer_end_;
        // File
        uint64_t remaining_, offset_;
    };

    // Wrapper comparison function for queue entries.
    class Greater : public std::binary_function<const Entry &, const Entry &, bool> {
      public:
        explicit Greater(const Compare &compare) : compare_(compare) {}

        bool operator()(const Entry &first, const Entry &second) const {
          return compare_(second.Current(), first.Current());
        }

      private:
        const Compare compare_;
    };

    typedef std::priority_queue<Entry, std::vector<Entry>, Greater> Queue;
    Queue queue_;

    const int in_;
    const std::size_t buffer_size_;
    const std::size_t entry_size_;
};

/* A worker object that merges.  If the number of pieces to merge exceeds the
 * arity, it outputs multiple sorted blocks, recording to out_offsets.
 * However, users will only every see a single sorted block out output because
 * Sort::Sorted insures the arity is higher than the number of pieces before
 * returning this.
 */
template <class Compare, class Combine> class MergingReader {
  public:
    MergingReader(int in, Offsets *in_offsets, Offsets *out_offsets, std::size_t buffer_size, std::size_t total_memory, const Compare &compare, const Combine &combine) :
        compare_(compare), combine_(combine),
        in_(in),
        in_offsets_(in_offsets), out_offsets_(out_offsets),
        buffer_size_(buffer_size), total_memory_(total_memory) {}

    void Run(const ChainPosition &position) {
      Run(position, false);
    }

    void Run(const ChainPosition &position, bool assert_one) {
      // Special case: nothing to read.
      if (!in_offsets_->RemainingBlocks()) {
        Link l(position);
        l.Poison();
        return;
      }
      // If there's just one entry, just read.
      if (in_offsets_->RemainingBlocks() == 1) {
        // Sequencing is important.
        uint64_t offset = in_offsets_->TotalOffset();
        uint64_t amount = in_offsets_->NextSize();
        ReadSingle(offset, amount, position);
        if (out_offsets_) out_offsets_->Append(amount);
        return;
      }

      Stream str(position);
      scoped_malloc buffer(MallocOrThrow(total_memory_));
      uint8_t *const buffer_end = static_cast<uint8_t*>(buffer.get()) + total_memory_;

      const std::size_t entry_size = position.GetChain().EntrySize();

      while (in_offsets_->RemainingBlocks()) {
        // Use bigger buffers if there's less remaining.
        uint64_t per_buffer = static_cast<uint64_t>(std::max<std::size_t>(
            buffer_size_,
            static_cast<std::size_t>((static_cast<uint64_t>(total_memory_) / in_offsets_->RemainingBlocks()))));
        per_buffer -= per_buffer % entry_size;
        assert(per_buffer);

        // Populate queue.
        MergeQueue<Compare> queue(in_, per_buffer, entry_size, compare_);
        for (uint8_t *buf = static_cast<uint8_t*>(buffer.get());
            in_offsets_->RemainingBlocks() && (buf + std::min(per_buffer, in_offsets_->PeekSize()) <= buffer_end);) {
          uint64_t offset = in_offsets_->TotalOffset();
          uint64_t size = in_offsets_->NextSize();
          queue.Push(buf, offset, size);
          buf += static_cast<std::size_t>(std::min<uint64_t>(size, per_buffer));
        }
        // This shouldn't happen but it's probably better to die than loop indefinitely.
        if (queue.Size() < 2 && in_offsets_->RemainingBlocks()) {
          std::cerr << "Bug in sort implementation: not merging at least two stripes." << std::endl;
          abort();
        }
        if (assert_one && in_offsets_->RemainingBlocks()) {
          std::cerr << "Bug in sort implementation: should only be one merge group for lazy sort" << std::endl;
          abort();
        }

        uint64_t written = 0;
        // Merge including combiner support.
        memcpy(str.Get(), queue.Top(), entry_size);
        for (queue.Pop(); !queue.Empty(); queue.Pop()) {
          if (!combine_(str.Get(), queue.Top(), compare_)) {
            ++written; ++str;
            memcpy(str.Get(), queue.Top(), entry_size);
          }
        }
        ++written; ++str;
        if (out_offsets_)
          out_offsets_->Append(written * entry_size);
      }
      str.Poison();
    }

  private:
    void ReadSingle(uint64_t offset, const uint64_t size, const ChainPosition &position) {
      // Special case: only one to read.
      const uint64_t end = offset + size;
      const uint64_t block_size = position.GetChain().BlockSize();
      Link l(position);
      for (; offset + block_size < end; ++l, offset += block_size) {
        ErsatzPRead(in_, l->Get(), block_size, offset);
        l->SetValidSize(block_size);
      }
      ErsatzPRead(in_, l->Get(), end - offset, offset);
      l->SetValidSize(end - offset);
      (++l).Poison();
      return;
    }

    Compare compare_;
    Combine combine_;

    int in_;

  protected:
    Offsets *in_offsets_;

  private:
    Offsets *out_offsets_;

    std::size_t buffer_size_;
    std::size_t total_memory_;
};

// The lazy step owns the remaining files.  This keeps track of them.
template <class Compare, class Combine> class OwningMergingReader : public MergingReader<Compare, Combine> {
  private:
    typedef MergingReader<Compare, Combine> P;
  public:
    OwningMergingReader(int data, const Offsets &offsets, std::size_t buffer, std::size_t lazy, const Compare &compare, const Combine &combine)
      : P(data, NULL, NULL, buffer, lazy, compare, combine),
        data_(data),
        offsets_(offsets) {}

    void Run(const ChainPosition &position) {
      P::in_offsets_ = &offsets_;
      scoped_fd data(data_);
      scoped_fd offsets_file(offsets_.File());
      P::Run(position, true);
    }

  private:
    int data_;
    Offsets offsets_;
};

// Don't use this directly.  Worker that sorts blocks.
template <class Compare> class BlockSorter {
  public:
    BlockSorter(Offsets &offsets, const Compare &compare) :
      offsets_(&offsets), compare_(compare) {}

    void Run(const ChainPosition &position) {
      const std::size_t entry_size = position.GetChain().EntrySize();
      for (Link link(position); link; ++link) {
        // Record the size of each block in a separate file.
        offsets_->Append(link->ValidSize());
        void *end = static_cast<uint8_t*>(link->Get()) + link->ValidSize();
        SizedSort(link->Get(), end, entry_size, compare_);
      }
      offsets_->FinishedAppending();
    }

  private:
    Offsets *offsets_;
    Compare compare_;
};

class BadSortConfig : public Exception {
  public:
    BadSortConfig() throw() {}
    ~BadSortConfig() throw() {}
};

/** Sort */
template <class Compare, class Combine = NeverCombine> class Sort {
  public:
    /** Constructs an object capable of sorting */
    Sort(Chain &in, const SortConfig &config, const Compare &compare = Compare(), const Combine &combine = Combine())
      : config_(config),
        data_(MakeTemp(config.temp_prefix)),
        offsets_file_(MakeTemp(config.temp_prefix)), offsets_(offsets_file_.get()),
        compare_(compare), combine_(combine),
        entry_size_(in.EntrySize()) {
      UTIL_THROW_IF(!entry_size_, BadSortConfig, "Sorting entries of size 0");
      // Make buffer_size a multiple of the entry_size.
      config_.buffer_size -= config_.buffer_size % entry_size_;
      UTIL_THROW_IF(!config_.buffer_size, BadSortConfig, "Sort buffer too small");
      UTIL_THROW_IF(config_.total_memory < config_.buffer_size * 4, BadSortConfig, "Sorting memory " << config_.total_memory << " is too small for four buffers (two read and two write).");
      in >> BlockSorter<Compare>(offsets_, compare_) >> WriteAndRecycle(data_.get());
    }

    uint64_t Size() const {
      return SizeOrThrow(data_.get());
    }

    // Do merge sort, terminating when lazy merge could be done with the
    // specified memory.  Return the minimum memory necessary to do lazy merge.
    std::size_t Merge(std::size_t lazy_memory) {
      if (offsets_.RemainingBlocks() <= 1) return 0;
      const uint64_t lazy_arity = std::max<uint64_t>(1, lazy_memory / config_.buffer_size);
      uint64_t size = Size();
      /* No overflow because
       * offsets_.RemainingBlocks() * config_.buffer_size <= lazy_memory ||
       * size < lazy_memory
       */
      if (offsets_.RemainingBlocks() <= lazy_arity || size <= static_cast<uint64_t>(lazy_memory))
        return std::min<std::size_t>(size, offsets_.RemainingBlocks() * config_.buffer_size);

      scoped_fd data2(MakeTemp(config_.temp_prefix));
      int fd_in = data_.get(), fd_out = data2.get();
      scoped_fd offsets2_file(MakeTemp(config_.temp_prefix));
      Offsets offsets2(offsets2_file.get());
      Offsets *offsets_in = &offsets_, *offsets_out = &offsets2;

      // Double buffered writing.
      ChainConfig chain_config;
      chain_config.entry_size = entry_size_;
      chain_config.block_count = 2;
      chain_config.total_memory = config_.buffer_size * 2;
      Chain chain(chain_config);

      while (offsets_in->RemainingBlocks() > lazy_arity) {
        if (size <= static_cast<uint64_t>(lazy_memory)) break;
        std::size_t reading_memory = config_.total_memory - 2 * config_.buffer_size;
        if (size < static_cast<uint64_t>(reading_memory)) {
          reading_memory = static_cast<std::size_t>(size);
        }
        SeekOrThrow(fd_in, 0);
        chain >>
          MergingReader<Compare, Combine>(
              fd_in,
              offsets_in, offsets_out,
              config_.buffer_size,
              reading_memory,
              compare_, combine_) >>
          WriteAndRecycle(fd_out);
        chain.Wait();
        offsets_out->FinishedAppending();
        ResizeOrThrow(fd_in, 0);
        offsets_in->Reset();
        std::swap(fd_in, fd_out);
        std::swap(offsets_in, offsets_out);
        size = SizeOrThrow(fd_in);
      }

      SeekOrThrow(fd_in, 0);
      if (fd_in == data2.get()) {
        data_.reset(data2.release());
        offsets_file_.reset(offsets2_file.release());
        offsets_ = offsets2;
      }
      if (offsets_.RemainingBlocks() <= 1) return 0;
      // No overflow because the while loop exited.
      return std::min(size, offsets_.RemainingBlocks() * static_cast<uint64_t>(config_.buffer_size));
    }

    // Output to chain, using this amount of memory, maximum, for lazy merge
    // sort.
    void Output(Chain &out, std::size_t lazy_memory) {
      Merge(lazy_memory);
      out.SetProgressTarget(Size());
      out >> OwningMergingReader<Compare, Combine>(data_.get(), offsets_, config_.buffer_size, lazy_memory, compare_, combine_);
      data_.release();
      offsets_file_.release();
    }

    /* If a pipeline step is reading sorted input and writing to a different
     * sort order, then there's a trade-off between using RAM to read lazily
     * (avoiding copying the file) and using RAM to increase block size and,
     * therefore, decrease the number of merge sort passes in the next
     * iteration.
     *
     * Merge sort takes log_{arity}(pieces) passes.  Thus, each time the chain
     * block size is multiplied by arity, the number of output passes decreases
     * by one.  Up to a constant, then, log_{arity}(chain) is the number of
     * passes saved.  Chain simply divides the memory evenly over all blocks.
     *
     * Lazy sort saves this many passes (up to a constant)
     *   log_{arity}((memory-lazy)/block_count) + 1
     * Non-lazy sort saves this many passes (up to the same constant):
     *   log_{arity}(memory/block_count)
     * Add log_{arity}(block_count) to both:
     *   log_{arity}(memory-lazy) + 1 versus log_{arity}(memory)
     * Take arity to the power of both sizes (arity > 1)
     *   (memory - lazy)*arity versus memory
     * Solve for lazy
     *   lazy = memory * (arity - 1) / arity
     */
    std::size_t DefaultLazy() {
      float arity = static_cast<float>(config_.total_memory / config_.buffer_size);
      return static_cast<std::size_t>(static_cast<float>(config_.total_memory) * (arity - 1.0) / arity);
    }

    // Same as Output with default lazy memory setting.
    void Output(Chain &out) {
      Output(out, DefaultLazy());
    }

    // Completely merge sort and transfer ownership to the caller.
    int StealCompleted() {
      // Merge all the way.
      Merge(0);
      SeekOrThrow(data_.get(), 0);
      offsets_file_.reset();
      return data_.release();
    }

  private:
    SortConfig config_;

    scoped_fd data_;

    scoped_fd offsets_file_;
    Offsets offsets_;

    const Compare compare_;
    const Combine combine_;
    const std::size_t entry_size_;
};

// returns bytes to be read on demand.
template <class Compare, class Combine> uint64_t BlockingSort(Chain &chain, const SortConfig &config, const Compare &compare = Compare(), const Combine &combine = NeverCombine()) {
  Sort<Compare, Combine> sorter(chain, config, compare, combine);
  chain.Wait(true);
  uint64_t size = sorter.Size();
  sorter.Output(chain);
  return size;
}

/**
 * Represents an @ref util::FixedArray "array" capable of storing @ref util::stream::Sort "Sort" objects.
 *
 * In the anticipated use case, an instance of this class will maintain one @ref util::stream::Sort "Sort" object
 * for each n-gram order (ranging from 1 up to the maximum n-gram order being processed).
 * Use in this manner would enable the n-grams each n-gram order to be sorted, in parallel.
 *
 * @tparam Compare An @ref Comparator "ngram comparator" to use during sorting.
 */
template <class Compare, class Combine = NeverCombine> class Sorts : public FixedArray<Sort<Compare, Combine> > {
  private:
    typedef Sort<Compare, Combine> S;
    typedef FixedArray<S> P;

  public:
    /**
     * Constructs, but does not initialize.
     *
     * @ref util::FixedArray::Init() "Init" must be called before use.
     *
     * @see util::FixedArray::Init()
     */
    Sorts() {}

    /**
     * Constructs an @ref util::FixedArray "array" capable of storing a fixed number of @ref util::stream::Sort "Sort" objects.
     *
     * @param number The maximum number of @ref util::stream::Sort "sorters" that can be held by this @ref util::FixedArray "array"
     * @see util::FixedArray::FixedArray()
     */
    explicit Sorts(std::size_t number) : FixedArray<Sort<Compare, Combine> >(number) {}

    /**
     * Constructs a new @ref util::stream::Sort "Sort" object which is stored in this @ref util::FixedArray "array".
     *
     * The new @ref util::stream::Sort "Sort" object is constructed using the provided @ref util::stream::SortConfig "SortConfig" and @ref Comparator "ngram   comparator";
     * once constructed, a new worker @ref util::stream::Thread "thread" (owned by the @ref util::stream::Chain "chain") will sort the n-gram data stored
     * in the @ref util::stream::Block "blocks" of the provided @ref util::stream::Chain "chain".
     *
     * @see util::stream::Sort::Sort()
     * @see util::stream::Chain::operator>>()
     */
    void push_back(util::stream::Chain &chain, const util::stream::SortConfig &config, const Compare &compare = Compare(), const Combine &combine = Combine()) {
      new (P::end()) S(chain, config, compare, combine); // use "placement new" syntax to initalize S in an already-allocated memory location
      P::Constructed();
    }
};

} // namespace stream
} // namespace util

#endif // UTIL_STREAM_SORT_H
