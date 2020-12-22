#ifndef UTIL_STREAM_MULTI_STREAM_H
#define UTIL_STREAM_MULTI_STREAM_H

#include "util/fixed_array.hh"
#include "util/scoped.hh"
#include "util/stream/chain.hh"
#include "util/stream/stream.hh"

#include <cstddef>
#include <new>

#include <cassert>
#include <cstdlib>

namespace util { namespace stream {

class Chains;

class ChainPositions : public util::FixedArray<util::stream::ChainPosition> {
  public:
    ChainPositions() {}

    explicit ChainPositions(std::size_t bound) :
      util::FixedArray<util::stream::ChainPosition>(bound) {}

    void Init(Chains &chains);

    explicit ChainPositions(Chains &chains) {
      Init(chains);
    }
};

class Chains : public util::FixedArray<util::stream::Chain> {
  private:
    template <class T, void (T::*ptr)(const ChainPositions &) = &T::Run> struct CheckForRun {
      typedef Chains type;
    };

  public:
    // Must call Init.
    Chains() {}

    explicit Chains(std::size_t limit) : util::FixedArray<util::stream::Chain>(limit) {}

    template <class Worker> typename CheckForRun<Worker>::type &operator>>(const Worker &worker) {
      threads_.push_back(new util::stream::Thread(ChainPositions(*this), worker));
      return *this;
    }

    template <class Worker> typename CheckForRun<Worker>::type &operator>>(const boost::reference_wrapper<Worker> &worker) {
      threads_.push_back(new util::stream::Thread(ChainPositions(*this), worker));
      return *this;
    }

    Chains &operator>>(const util::stream::Recycler &recycler) {
      for (util::stream::Chain *i = begin(); i != end(); ++i)
        *i >> recycler;
      return *this;
    }

    void Wait(bool release_memory = true) {
      threads_.clear();
      for (util::stream::Chain *i = begin(); i != end(); ++i) {
        i->Wait(release_memory);
      }
    }

  private:
    boost::ptr_vector<util::stream::Thread> threads_;

    Chains(const Chains &);
    void operator=(const Chains &);
};

inline void ChainPositions::Init(Chains &chains) {
  util::FixedArray<util::stream::ChainPosition>::Init(chains.size());
  for (util::stream::Chain *i = chains.begin(); i != chains.end(); ++i) {
    // use "placement new" syntax to initalize ChainPosition in an already-allocated memory location
    new (end()) util::stream::ChainPosition(i->Add()); Constructed();
  }
}

inline Chains &operator>>(Chains &chains, ChainPositions &positions) {
  positions.Init(chains);
  return chains;
}

template <class T> class GenericStreams : public util::FixedArray<T> {
  private:
    typedef util::FixedArray<T> P;
  public:
    GenericStreams() {}

    // Limit restricts to positions[0,limit)
    void Init(const ChainPositions &positions, std::size_t limit) {
      P::Init(limit);
      for (const util::stream::ChainPosition *i = positions.begin(); i != positions.begin() + limit; ++i) {
        P::push_back(*i);
      }
    }
    void Init(const ChainPositions &positions) {
      Init(positions, positions.size());
    }

    GenericStreams(const ChainPositions &positions) {
      Init(positions);
    }

    void Init(std::size_t amount) {
      P::Init(amount);
    }
};

template <class T> inline Chains &operator>>(Chains &chains, GenericStreams<T> &streams) {
  ChainPositions positions;
  chains >> positions;
  streams.Init(positions);
  return chains;
}

typedef GenericStreams<Stream> Streams;

}} // namespaces
#endif // UTIL_STREAM_MULTI_STREAM_H
