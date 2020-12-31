#ifndef UTIL_STREAM_REWINDABLE_STREAM_H
#define UTIL_STREAM_REWINDABLE_STREAM_H

#include "util/stream/chain.hh"

#include <boost/noncopyable.hpp>

#include <deque>

namespace util {
namespace stream {

/**
 * A RewindableStream is like a Stream (but one that is only used for
 * creating input at the start of a chain) except that it can be rewound to
 * be able to re-write a part of the stream before it is sent. Rewinding
 * has a limit of 2 * block_size_ - 1 in distance (it does *not* buffer an
 * entire stream into memory, only a maximum of 2 * block_size_).
 */
class RewindableStream : boost::noncopyable {
  public:
    /**
     * Creates an uninitialized RewindableStream. You **must** call Init()
     * on it later!
     */
    RewindableStream();

    ~RewindableStream() {
      Poison();
    }

    /**
     * Initializes an existing RewindableStream at a specific position in
     * a Chain.
     *
     * @param position The position in the chain to get input from and
     *  produce output on
     */
    void Init(const ChainPosition &position);

    /**
     * Constructs a RewindableStream at a specific position in a Chain all
     * in one step.
     *
     * Equivalent to RewindableStream a(); a.Init(....);
     */
    explicit RewindableStream(const ChainPosition &position)
      : in_(NULL) {
      Init(position);
    }

    /**
     * Gets the record at the current stream position. Const version.
     */
    const void *Get() const {
      assert(!poisoned_);
      assert(current_);
      return current_;
    }

    /**
     * Gets the record at the current stream position.
     */
    void *Get() {
      assert(!poisoned_);
      assert(current_);
      return current_;
    }

    operator bool() const { return !poisoned_; }

    bool operator!() const { return poisoned_; }

    /**
     * Marks the current position in the stream to be rewound to later.
     * Note that you can only rewind back as far as 2 * block_size_ - 1!
     */
    void Mark();

    /**
     * Rewinds the stream back to the marked position. This will throw an
     * exception if the marked position is too far away.
     */
    void Rewind();

    /**
     * Moves the stream forward to the next record. This internally may
     * buffer a block for the purposes of rewinding.
     */
    RewindableStream& operator++();

    /**
     * Poisons the stream. This sends any buffered blocks down the chain
     * and sends a poison block as well (sending at most 2 non-poison and 1
     * poison block).
     */
    void Poison();

  private:
    void AppendBlock();

    void Flush(std::deque<Block>::iterator to);

    std::deque<Block> blocks_;
    // current_ is in blocks_[blocks_it_] unless poisoned_.
    std::size_t blocks_it_;

    std::size_t entry_size_;
    std::size_t block_size_;
    std::size_t block_count_;

    uint8_t *marked_, *current_;
    const uint8_t *block_end_;

    PCQueue<Block> *in_, *out_;

    // Have we hit poison at the end of the stream, even if rewinding?
    bool hit_poison_;
    // Is the curren position poison?
    bool poisoned_;

    WorkerProgress progress_;
};

inline Chain &operator>>(Chain &chain, RewindableStream &stream) {
  stream.Init(chain.Add());
  return chain;
}

}
}
#endif
