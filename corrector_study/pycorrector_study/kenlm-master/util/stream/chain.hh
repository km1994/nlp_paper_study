#ifndef UTIL_STREAM_CHAIN_H
#define UTIL_STREAM_CHAIN_H

#include "util/stream/block.hh"
#include "util/stream/config.hh"
#include "util/stream/multi_progress.hh"
#include "util/scoped.hh"

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/thread/thread.hpp>

#include <cstddef>
#include <cassert>

namespace util {
template <class T> class PCQueue;
namespace stream {

class ChainConfigException : public Exception {
  public:
    ChainConfigException() throw();
    ~ChainConfigException() throw();
};

class Chain;
class RewindableStream;

/**
 * Encapsulates a @ref PCQueue "producer queue" and a @ref PCQueue "consumer queue" within a @ref Chain "chain".
 *
 * Specifies position in chain for Link constructor.
 */
class ChainPosition {
  public:
    const Chain &GetChain() const { return *chain_; }
  private:
    friend class Chain;
    friend class Link;
    friend class RewindableStream;
    ChainPosition(PCQueue<Block> &in, PCQueue<Block> &out, Chain *chain, MultiProgress &progress)
      : in_(&in), out_(&out), chain_(chain), progress_(progress.Add()) {}

    PCQueue<Block> *in_, *out_;

    Chain *chain_;

    WorkerProgress progress_;
};


/**
 * Encapsulates a worker thread processing data at a given position in the chain.
 *
 * Each instance of this class owns one boost thread in which the worker is Run().
 */
class Thread {
  public:

    /**
     * Constructs a new Thread in which the provided Worker is Run().
     *
     * Position is usually ChainPosition but if there are multiple streams involved, this can be ChainPositions.
     *
     * After a call to this constructor, the provided worker will be running within a boost thread owned by the newly constructed Thread object.
     */
    template <class Position, class Worker> Thread(const Position &position, const Worker &worker)
      : thread_(boost::ref(*this), position, worker) {}

    ~Thread();

    /**
     * Launches the provided worker in this object's boost thread.
     *
     * This method is called automatically by this class's @ref Thread() "constructor".
     */
    template <class Position, class Worker> void operator()(const Position &position, Worker &worker) {
      try {
        worker.Run(position);
      } catch (const std::exception &e) {
        UnhandledException(e);
      }
    }

  private:
    void UnhandledException(const std::exception &e);

    boost::thread thread_;
};

/**
 * This resets blocks to full valid size.  Used to close the loop in Chain by recycling blocks.
 */
class Recycler {
  public:
    /**
     * Resets the blocks in the chain such that the blocks' respective valid sizes match the chain's block size.
     *
     * @see Block::SetValidSize()
     * @see Chain::BlockSize()
     */
    void Run(const ChainPosition &position);
};

extern const Recycler kRecycle;
class WriteAndRecycle;

/**
 * Represents a sequence of workers, through which @ref Block "blocks" can pass.
 */
class Chain {
  private:
    template <class T, void (T::*ptr)(const ChainPosition &) = &T::Run> struct CheckForRun {
      typedef Chain type;
    };

  public:

    /**
     * Constructs a configured Chain.
     *
     * @param config Specifies how to configure the Chain.
     */
    explicit Chain(const ChainConfig &config);

    /**
     * Destructs a Chain.
     *
     * This method waits for the chain's threads to complete,
     * and frees the memory held by this chain.
     */
    ~Chain();

    void ActivateProgress() {
      assert(!Running());
      progress_.Activate();
    }

    void SetProgressTarget(uint64_t target) {
      progress_.SetTarget(target);
    }

    /**
     * Gets the number of bytes in each record of a Block.
     *
     * @see ChainConfig::entry_size
     */
    std::size_t EntrySize() const {
      return config_.entry_size;
    }

    /**
     * Gets the inital @ref Block::ValidSize "valid size" for @ref Block "blocks" in this chain.
     *
     * @see Block::ValidSize
     */
    std::size_t BlockSize() const {
      return block_size_;
    }

    /**
     * Number of blocks going through the Chain.
     */
    std::size_t BlockCount() const {
      return config_.block_count;
    }

    /** Two ways to add to the chain: Add() or operator>>. */
    ChainPosition Add();

    /**
     * Adds a new worker to this chain,
     * and runs that worker in a new Thread owned by this chain.
     *
     * The worker must have a Run method that accepts a position argument.
     *
     * @see Thread::operator()()
     */
    template <class Worker> typename CheckForRun<Worker>::type &operator>>(const Worker &worker) {
      assert(!complete_called_);
      threads_.push_back(new Thread(Add(), worker));
      return *this;
    }

  /**
   * Adds a new worker to this chain (but avoids copying that worker),
   * and runs that worker in a new Thread owned by this chain.
   *
   * The worker must have a Run method that accepts a position argument.
   *
   * @see Thread::operator()()
   */
    template <class Worker> typename CheckForRun<Worker>::type &operator>>(const boost::reference_wrapper<Worker> &worker) {
      assert(!complete_called_);
      threads_.push_back(new Thread(Add(), worker));
      return *this;
    }

    // Note that Link and Stream also define operator>> outside this class.

    // To complete the loop, call CompleteLoop(), >> kRecycle, or the destructor.
    void CompleteLoop() {
      threads_.push_back(new Thread(Complete(), kRecycle));
    }

    /**
     * Adds a Recycler worker to this chain,
     * and runs that worker in a new Thread owned by this chain.
     */
    Chain &operator>>(const Recycler &) {
      CompleteLoop();
      return *this;
    }

    /**
     * Adds a WriteAndRecycle worker to this chain,
     * and runs that worker in a new Thread owned by this chain.
     */
    Chain &operator>>(const WriteAndRecycle &writer);

    // Chains are reusable.  Call Wait to wait for everything to finish and free memory.
    void Wait(bool release_memory = true);

    // Waits for the current chain to complete (if any) then starts again.
    void Start();

    bool Running() const { return !queues_.empty(); }

  private:
    ChainPosition Complete();

    ChainConfig config_;

    std::size_t block_size_;

    scoped_malloc memory_;

    boost::ptr_vector<PCQueue<Block> > queues_;

    bool complete_called_;

    boost::ptr_vector<Thread> threads_;

    MultiProgress progress_;
};

// Create the link in the worker thread using the position token.
/**
 * Represents a C++ style iterator over @ref Block "blocks".
 */
class Link {
  public:

    // Either default construct and Init or just construct all at once.

    /**
     * Constructs an @ref Init "initialized" link.
     *
     * @see Init
     */
    explicit Link(const ChainPosition &position);

    /**
     * Constructs a link that must subsequently be @ref Init "initialized".
     *
     * @see Init
     */
    Link();

    /**
     * Initializes the link with the input @ref PCQueue "consumer queue" and output @ref PCQueue "producer queue" at a given @ref ChainPosition "position" in the @ref Chain "chain".
     *
     * @see Link()
     */
    void Init(const ChainPosition &position);

    /**
     * Destructs the link object.
     *
     * If necessary, this method will pass a poison block
     * to this link's output @ref PCQueue "producer queue".
     *
     * @see Block::SetToPoison()
     */
    ~Link();

    /**
     * Gets a reference to the @ref Block "block" at this link.
     */
    Block &operator*() { return current_; }

    /**
     * Gets a const reference to the @ref Block "block" at this link.
     */
    const Block &operator*() const { return current_; }

    /**
     * Gets a pointer to the @ref Block "block" at this link.
     */
    Block *operator->() { return &current_; }

    /**
     * Gets a const pointer to the @ref Block "block" at this link.
     */
    const Block *operator->() const { return &current_; }

    /**
     * Gets the link at the next @ref ChainPosition "position" in the @ref Chain "chain".
     */
    Link &operator++();

    /**
     * Returns true if the @ref Block "block" at this link encapsulates a valid (non-NULL) block of memory.
     *
     * This method is a user-defined implicit conversion function to boolean;
     * among other things, this method enables bare instances of this class
     * to be used as the condition of an if statement.
     */
    operator bool() const { return current_; }

    /**
     * @ref Block::SetToPoison() "Poisons" the @ref Block "block" at this link,
     * and passes this now-poisoned block to this link's output @ref PCQueue "producer queue".
     *
     * @see Block::SetToPoison()
     */
    void Poison();

  private:
    Block current_;
    PCQueue<Block> *in_, *out_;

    bool poisoned_;

    WorkerProgress progress_;
};

inline Chain &operator>>(Chain &chain, Link &link) {
  link.Init(chain.Add());
  return chain;
}

} // namespace stream
} // namespace util

#endif // UTIL_STREAM_CHAIN_H
