#ifndef LM_BUILDER_HASH_GAMMA__
#define LM_BUILDER_HASH_GAMMA__

#include <stdint.h>

namespace lm { namespace builder {

#pragma pack(push)
#pragma pack(4)

struct HashGamma {
    uint64_t hash_value;
    float gamma;
};

#pragma pack(pop)

}} // namespaces
#endif // LM_BUILDER_HASH_GAMMA__
