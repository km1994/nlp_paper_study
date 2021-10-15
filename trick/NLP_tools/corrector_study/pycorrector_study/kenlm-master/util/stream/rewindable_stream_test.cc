#include "util/stream/io.hh"

#include "util/stream/rewindable_stream.hh"
#include "util/file.hh"

#define BOOST_TEST_MODULE RewindableStreamTest
#include <boost/test/unit_test.hpp>

namespace util {
namespace stream {
namespace {

BOOST_AUTO_TEST_CASE(RewindableStreamTest) {
  scoped_fd in(MakeTemp("io_test_temp"));
  for (uint64_t i = 0; i < 100000; ++i) {
    WriteOrThrow(in.get(), &i, sizeof(uint64_t));
  }
  SeekOrThrow(in.get(), 0);

  ChainConfig config;
  config.entry_size = 8;
  config.total_memory = 100;
  config.block_count = 6;

  Chain chain(config);
  RewindableStream s;
  chain >> Read(in.get()) >> s >> kRecycle;
  uint64_t i = 0;
  for (; s; ++s, ++i) {
    BOOST_CHECK_EQUAL(i, *static_cast<const uint64_t*>(s.Get()));
    if (100000UL - i == 2)
      s.Mark();
  }
  BOOST_CHECK_EQUAL(100000ULL, i);
  s.Rewind();
  BOOST_CHECK_EQUAL(100000ULL - 2, *static_cast<const uint64_t*>(s.Get()));
}

}
}
}
