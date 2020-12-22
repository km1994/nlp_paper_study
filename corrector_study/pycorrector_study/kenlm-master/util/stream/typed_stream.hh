#ifndef UTIL_STREAM_TYPED_STREAM_H
#define UTIL_STREAM_TYPED_STREAM_H
// A typed wrapper to Stream for POD types.

#include "util/stream/stream.hh"

namespace util { namespace stream {

template <class T> class TypedStream : public Stream {
  public:
    // After using the default constructor, call Init (in the parent class)
    TypedStream() {}

    explicit TypedStream(const ChainPosition &position) : Stream(position) {}

    const T *operator->() const { return static_cast<const T*>(Get()); }
    T *operator->() { return static_cast<T*>(Get()); }

    const T &operator*() const { return *static_cast<const T*>(Get()); }
    T &operator*() { return *static_cast<T*>(Get()); }
};

}} // namespaces

#endif // UTIL_STREAM_TYPED_STREAM_H
