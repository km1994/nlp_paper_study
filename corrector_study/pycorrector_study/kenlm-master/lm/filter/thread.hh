#ifndef LM_FILTER_THREAD_H
#define LM_FILTER_THREAD_H

#include "util/thread_pool.hh"

#include <boost/utility/in_place_factory.hpp>

#include <deque>
#include <stack>

namespace lm {

template <class OutputBuffer> class ThreadBatch {
  public:
    ThreadBatch() {}

    void Reserve(size_t size) {
      input_.Reserve(size);
      output_.Reserve(size);
     }

    // File reading thread.
    InputBuffer &Fill(uint64_t sequence) {
      sequence_ = sequence;
      // Why wait until now to clear instead of after output?  free in the same
      // thread as allocated.
      input_.Clear();
      return input_;
    }

    // Filter worker thread.
    template <class Filter> void CallFilter(Filter &filter) {
      input_.CallFilter(filter, output_);
    }

    uint64_t Sequence() const { return sequence_; }

    // File writing thread.
    template <class RealOutput> void Flush(RealOutput &output) {
      output_.Flush(output);
    }

  private:
    InputBuffer input_;
    OutputBuffer output_;

    uint64_t sequence_;
};

template <class Batch, class Filter> class FilterWorker {
  public:
    typedef Batch *Request;

    FilterWorker(const Filter &filter, util::PCQueue<Request> &done) : filter_(filter), done_(done) {}

    void operator()(Request request) {
      request->CallFilter(filter_);
      done_.Produce(request);
    }

  private:
    Filter filter_;

    util::PCQueue<Request> &done_;
};

// There should only be one OutputWorker.
template <class Batch, class Output> class OutputWorker {
  public:
    typedef Batch *Request;

    OutputWorker(Output &output, util::PCQueue<Request> &done) : output_(output), done_(done), base_sequence_(0) {}

    void operator()(Request request) {
      assert(request->Sequence() >= base_sequence_);
      // Assemble the output in order.
      uint64_t pos = request->Sequence() - base_sequence_;
      if (pos >= ordering_.size()) {
        ordering_.resize(pos + 1, NULL);
      }
      ordering_[pos] = request;
      while (!ordering_.empty() && ordering_.front()) {
        ordering_.front()->Flush(output_);
        done_.Produce(ordering_.front());
        ordering_.pop_front();
        ++base_sequence_;
      }
    }

  private:
    Output &output_;

    util::PCQueue<Request> &done_;

    std::deque<Request> ordering_;

    uint64_t base_sequence_;
};

template <class Filter, class OutputBuffer, class RealOutput> class Controller : boost::noncopyable {
  private:
    typedef ThreadBatch<OutputBuffer> Batch;

  public:
    Controller(size_t batch_size, size_t queue, size_t workers, const Filter &filter, RealOutput &output)
      : batch_size_(batch_size), queue_size_(queue),
        batches_(queue),
        to_read_(queue),
        output_(queue, 1, boost::in_place(boost::ref(output), boost::ref(to_read_)), NULL),
        filter_(queue, workers, boost::in_place(boost::ref(filter), boost::ref(output_.In())), NULL),
        sequence_(0) {
      for (size_t i = 0; i < queue; ++i) {
        batches_[i].Reserve(batch_size);
        local_read_.push(&batches_[i]);
      }
      NewInput();
    }

    void AddNGram(const StringPiece &ngram, const StringPiece &line, RealOutput &output) {
      input_->AddNGram(ngram, line, output);
      if (input_->Size() == batch_size_) {
        FlushInput();
        NewInput();
      }
    }

    void Flush() {
      FlushInput();
      while (local_read_.size() < queue_size_) {
        MoveRead();
      }
      NewInput();
    }

  private:
    void FlushInput() {
      if (input_->Empty()) return;
      filter_.Produce(local_read_.top());
      local_read_.pop();
      if (local_read_.empty()) MoveRead();
    }

    void NewInput() {
      input_ = &local_read_.top()->Fill(sequence_++);
    }

    void MoveRead() {
      local_read_.push(to_read_.Consume());
    }

    const size_t batch_size_;
    const size_t queue_size_;

    std::vector<Batch> batches_;

    util::PCQueue<Batch*> to_read_;
    std::stack<Batch*> local_read_;
    util::ThreadPool<OutputWorker<Batch, RealOutput> > output_;
    util::ThreadPool<FilterWorker<Batch, Filter> > filter_;

    uint64_t sequence_;
    InputBuffer *input_;
};

} // namespace lm

#endif // LM_FILTER_THREAD_H
