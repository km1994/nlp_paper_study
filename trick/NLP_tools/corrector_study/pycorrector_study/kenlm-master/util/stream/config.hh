#ifndef UTIL_STREAM_CONFIG_H
#define UTIL_STREAM_CONFIG_H

#include <cstddef>
#include <string>

namespace util { namespace stream {

/**
 * Represents how a chain should be configured.
 */
struct ChainConfig {

  /** Constructs an configuration with underspecified (or default) parameters. */
  ChainConfig() {}

  /**
   * Constructs a chain configuration object.
   *
   * @param [in] in_entry_size   Number of bytes in each record.
   * @param [in] in_block_count  Number of blocks in the chain.
   * @param [in] in_total_memory Total number of bytes available to the chain.
   *             This value will be divided amongst the blocks in the chain.
   */
  ChainConfig(std::size_t in_entry_size, std::size_t in_block_count, std::size_t in_total_memory)
    : entry_size(in_entry_size), block_count(in_block_count), total_memory(in_total_memory) {}

  /**
   * Number of bytes in each record.
   */
  std::size_t entry_size;

  /**
   * Number of blocks in the chain.
   */
  std::size_t block_count;

  /**
   * Total number of bytes available to the chain.
   * This value will be divided amongst the blocks in the chain.
   * Chain's constructor will make this a multiple of entry_size.
   */
  std::size_t total_memory;
};


/**
 * Represents how a sorter should be configured.
 */
struct SortConfig {

  /** Filename prefix where temporary files should be placed. */
  std::string temp_prefix;

  /** Size of each input/output buffer. */
  std::size_t buffer_size;

  /** Total memory to use when running alone. */
  std::size_t total_memory;
};

}} // namespaces
#endif // UTIL_STREAM_CONFIG_H
