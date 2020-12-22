#ifndef UTIL_STREAM_IO_H
#define UTIL_STREAM_IO_H

#include "util/exception.hh"
#include "util/file.hh"

namespace util {
namespace stream {

class ChainPosition;

class ReadSizeException : public util::Exception {
  public:
    ReadSizeException() throw();
    ~ReadSizeException() throw();
};

class Read {
  public:
    explicit Read(int fd) : file_(fd) {}
    void Run(const ChainPosition &position);
  private:
    int file_;
};

// Like read but uses pread so that the file can be accessed from multiple threads.
class PRead {
  public:
    explicit PRead(int fd, bool take_own = false) : file_(fd), own_(take_own) {}
    void Run(const ChainPosition &position);
  private:
    int file_;
    bool own_;
};

class Write {
  public:
    explicit Write(int fd) : file_(fd) {}
    void Run(const ChainPosition &position);
  private:
    int file_;
};

// It's a common case that stuff is written and then recycled.  So rather than
// spawn another thread to Recycle, this combines the two roles.
class WriteAndRecycle {
  public:
    explicit WriteAndRecycle(int fd) : file_(fd) {}
    void Run(const ChainPosition &position);
  private:
    int file_;
};

class PWrite {
  public:
    explicit PWrite(int fd) : file_(fd) {}
    void Run(const ChainPosition &position);
  private:
    int file_;
};


// Reuse the same file over and over again to buffer output.
class FileBuffer {
  public:
    explicit FileBuffer(int fd) : file_(fd) {}

    PWrite Sink() const {
      util::SeekOrThrow(file_.get(), 0);
      return PWrite(file_.get());
    }

    PRead Source(bool discard = false) {
      return PRead(discard ? file_.release() : file_.get(), discard);
    }

    uint64_t Size() const {
      return SizeOrThrow(file_.get());
    }

  private:
    scoped_fd file_;
};

} // namespace stream
} // namespace util
#endif // UTIL_STREAM_IO_H
