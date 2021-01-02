#ifndef UTIL_STREAM_BLOCK_H
#define UTIL_STREAM_BLOCK_H

#include <cstddef>
#include <stdint.h>

namespace util {
namespace stream {

/**
 * Encapsulates a block of memory.
 */
class Block {
  public:

    /**
     * Constructs an empty block.
     */
    Block() : mem_(NULL), valid_size_(0) {}

    /**
     * Constructs a block that encapsulates a segment of memory.
     *
     * @param[in] mem  The segment of memory to encapsulate
     * @param[in] size The size of the memory segment in bytes
     */
    Block(void *mem, std::size_t size) : mem_(mem), valid_size_(size) {}

    /**
     * Set the number of bytes in this block that should be interpreted as valid.
     *
     * @param[in] to Number of bytes
     */
    void SetValidSize(std::size_t to) { valid_size_ = to; }

    /**
     * Gets the number of bytes in this block that should be interpreted as valid.
     * This is important because read might fill in less than Allocated at EOF.
     */
    std::size_t ValidSize() const { return valid_size_; }

    /** Gets a void pointer to the memory underlying this block. */
    void *Get() { return mem_; }

    /** Gets a const void pointer to the memory underlying this block. */
    const void *Get() const { return mem_; }


    /**
     * Gets a const void pointer to the end of the valid section of memory
     * encapsulated by this block.
     */
    const void *ValidEnd() const {
      return reinterpret_cast<const uint8_t*>(mem_) + valid_size_;
    }

    /**
     * Returns true if this block encapsulates a valid (non-NULL) block of memory.
     *
     * This method is a user-defined implicit conversion function to boolean;
     * among other things, this method enables bare instances of this class
     * to be used as the condition of an if statement.
     */
    operator bool() const { return mem_ != NULL; }

    /**
     * Returns true if this block is empty.
     *
     * In other words, if Get()==NULL, this method will return true.
     */
    bool operator!() const { return mem_ == NULL; }

  private:
    friend class Link;
    friend class RewindableStream;

    /**
     * Points this block's memory at NULL.
     *
     * This class defines poison as a block whose memory pointer is NULL.
     */
    void SetToPoison() {
      mem_ = NULL;
    }

    void *mem_;
    std::size_t valid_size_;
};

} // namespace stream
} // namespace util

#endif // UTIL_STREAM_BLOCK_H
