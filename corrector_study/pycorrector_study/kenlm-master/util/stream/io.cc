#include "util/stream/io.hh"

#include "util/file.hh"
#include "util/stream/chain.hh"

#include <cstddef>

namespace util {
namespace stream {

ReadSizeException::ReadSizeException() throw() {}
ReadSizeException::~ReadSizeException() throw() {}

void Read::Run(const ChainPosition &position) {
  const std::size_t block_size = position.GetChain().BlockSize();
  const std::size_t entry_size = position.GetChain().EntrySize();
  for (Link link(position); link; ++link) {
    std::size_t got = util::ReadOrEOF(file_, link->Get(), block_size);
    UTIL_THROW_IF(got % entry_size, ReadSizeException, "File ended with " << got << " bytes, not a multiple of " << entry_size << ".");
    if (got == 0) {
      link.Poison();
      return;
    } else {
      link->SetValidSize(got);
    }
  }
}

void PRead::Run(const ChainPosition &position) {
  scoped_fd owner;
  if (own_) owner.reset(file_);
  const uint64_t size = SizeOrThrow(file_);
  UTIL_THROW_IF(size % static_cast<uint64_t>(position.GetChain().EntrySize()), ReadSizeException, "File size " << file_ << " size is " << size << " not a multiple of " << position.GetChain().EntrySize());
  const std::size_t block_size = position.GetChain().BlockSize();
  const uint64_t block_size64 = static_cast<uint64_t>(block_size);
  Link link(position);
  uint64_t offset = 0;
  for (; offset + block_size64 < size; offset += block_size64, ++link) {
    ErsatzPRead(file_, link->Get(), block_size, offset);
    link->SetValidSize(block_size);
  }
  // size - offset is <= block_size, so it casts to 32-bit fine.
  if (size - offset) {
    ErsatzPRead(file_, link->Get(), size - offset, offset);
    link->SetValidSize(size - offset);
    ++link;
  }
  link.Poison();
}

void Write::Run(const ChainPosition &position) {
  for (Link link(position); link; ++link) {
    WriteOrThrow(file_, link->Get(), link->ValidSize());
  }
}

void WriteAndRecycle::Run(const ChainPosition &position) {
  const std::size_t block_size = position.GetChain().BlockSize();
  for (Link link(position); link; ++link) {
    WriteOrThrow(file_, link->Get(), link->ValidSize());
    link->SetValidSize(block_size);
  }
}

void PWrite::Run(const ChainPosition &position) {
  uint64_t offset = 0;
  for (Link link(position); link; ++link) {
    ErsatzPWrite(file_, link->Get(), link->ValidSize(), offset);
    offset += link->ValidSize();
  }
  // Trim file to size.
  util::ResizeOrThrow(file_, offset);
}

} // namespace stream
} // namespace util
