#ifndef LM_BUILDER_PAYLOAD_H
#define LM_BUILDER_PAYLOAD_H

#include "lm/weights.hh"
#include "lm/word_index.hh"
#include <stdint.h>

namespace lm { namespace builder {

struct Uninterpolated {
  float prob;  // Uninterpolated probability.
  float gamma; // Interpolation weight for lower order.
};

union BuildingPayload {
  uint64_t count;
  Uninterpolated uninterp;
  ProbBackoff complete;

  /*mjd**********************************************************************/
  bool IsMarked() const {
    return count >> (sizeof(count) * 8 - 1);
  }

  void Mark() {
    count |= (1ULL << (sizeof(count) * 8 - 1));
  }

  void Unmark() {
    count &= ~(1ULL << (sizeof(count) * 8 - 1));
  }

  uint64_t UnmarkedCount() const {
    return count & ~(1ULL << (sizeof(count) * 8 - 1));
  }

  uint64_t CutoffCount() const {
    return IsMarked() ? 0 : UnmarkedCount();
  }
  /*mjd**********************************************************************/
};

const WordIndex kBOS = 1;
const WordIndex kEOS = 2;

}} // namespaces

#endif // LM_BUILDER_PAYLOAD_H
