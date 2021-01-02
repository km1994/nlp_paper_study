#include "util/stream/io.hh"

#include "util/stream/chain.hh"
#include "util/file.hh"

#define BOOST_TEST_MODULE IOTest
#include <boost/test/unit_test.hpp>

#include <unistd.h>

namespace util { namespace stream { namespace {

BOOST_AUTO_TEST_CASE(CopyFile) {
  std::string temps("io_test_temp");

  scoped_fd in(MakeTemp(temps));
  for (uint64_t i = 0; i < 100000; ++i) {
    WriteOrThrow(in.get(), &i, sizeof(uint64_t));
  }
  SeekOrThrow(in.get(), 0);
  scoped_fd out(MakeTemp(temps));

  ChainConfig config;
  config.entry_size = 8;
  config.total_memory = 1024;
  config.block_count = 10;

  Chain(config) >> PRead(in.get()) >> Write(out.get());

  SeekOrThrow(out.get(), 0);
  for (uint64_t i = 0; i < 100000; ++i) {
    uint64_t got;
    ReadOrThrow(out.get(), &got, sizeof(uint64_t));
    BOOST_CHECK_EQUAL(i, got);
  }
}

}}} // namespaces
