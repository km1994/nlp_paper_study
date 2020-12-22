#ifndef UTIL_STREAM_LINE_INPUT_H
#define UTIL_STREAM_LINE_INPUT_H
namespace util {namespace stream {

class ChainPosition;

/* Worker that reads input into blocks, ensuring that blocks contain whole
 * lines.  Assumes that the maximum size of a line is less than the block size
 */
class LineInput {
  public:
    // Takes ownership upon thread execution.
    explicit LineInput(int fd);

    void Run(const ChainPosition &position);

  private:
    int fd_;
};

}} // namespaces
#endif // UTIL_STREAM_LINE_INPUT_H
