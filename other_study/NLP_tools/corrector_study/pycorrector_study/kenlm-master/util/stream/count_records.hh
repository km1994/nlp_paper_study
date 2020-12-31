#include <stdint.h>

namespace util { namespace stream {

class ChainPosition;

class CountRecords {
  public:
    explicit CountRecords(uint64_t *out)
      : count_(out) {
      *count_ = 0;
    }

    void Run(const ChainPosition &position);

  private:
    uint64_t *count_;
};

}} // namespaces
