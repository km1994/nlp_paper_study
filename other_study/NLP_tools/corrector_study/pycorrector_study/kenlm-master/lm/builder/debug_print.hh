#ifndef LM_BUILDER_DEBUG_PRINT_H
#define LM_BUILDER_DEBUG_PRINT_H

#include "lm/builder/payload.hh"
#include "lm/common/print.hh"
#include "lm/common/ngram_stream.hh"
#include "util/file_stream.hh"
#include "util/file.hh"

#include <boost/lexical_cast.hpp>

namespace lm { namespace builder {
// Not defined, only specialized.
template <class T> void PrintPayload(util::FileStream &to, const BuildingPayload &payload);
template <> inline void PrintPayload<uint64_t>(util::FileStream &to, const BuildingPayload &payload) {
  to << payload.count;
}
template <> inline void PrintPayload<Uninterpolated>(util::FileStream &to, const BuildingPayload &payload) {
  to << log10(payload.uninterp.prob) << ' ' << log10(payload.uninterp.gamma);
}
template <> inline void PrintPayload<ProbBackoff>(util::FileStream &to, const BuildingPayload &payload) {
  to << payload.complete.prob << ' ' << payload.complete.backoff;
}

// template parameter is the type stored.
template <class V> class Print {
  public:
    static void DumpSeparateFiles(const VocabReconstitute &vocab, const std::string &file_base, util::stream::Chains &chains) {
      for (unsigned int i = 0; i < chains.size(); ++i) {
        std::string file(file_base + boost::lexical_cast<std::string>(i));
        chains[i] >> Print(vocab, util::CreateOrThrow(file.c_str()));
      }
    }

    explicit Print(const VocabReconstitute &vocab, int fd) : vocab_(vocab), to_(fd) {}

    void Run(const util::stream::ChainPositions &chains) {
      util::scoped_fd fd(to_);
      util::FileStream out(to_);
      NGramStreams<BuildingPayload> streams(chains);
      for (NGramStream<BuildingPayload> *s = streams.begin(); s != streams.end(); ++s) {
        DumpStream(*s, out);
      }
    }

    void Run(const util::stream::ChainPosition &position) {
      util::scoped_fd fd(to_);
      util::FileStream out(to_);
      NGramStream<BuildingPayload> stream(position);
      DumpStream(stream, out);
    }

  private:
    void DumpStream(NGramStream<BuildingPayload> &stream, util::FileStream &to) {
      for (; stream; ++stream) {
        PrintPayload<V>(to, stream->Value());
        for (const WordIndex *w = stream->begin(); w != stream->end(); ++w) {
          to << ' ' << vocab_.Lookup(*w) << '=' << *w;
        }
        to << '\n';
      }
    }

    const VocabReconstitute &vocab_;
    int to_;
};

}} // namespaces

#endif // LM_BUILDER_DEBUG_PRINT_H
