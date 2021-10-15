#include "lm/state.hh"
#include "lm/virtual_interface.hh"
#include "util/tokenize_piece.hh"

#include <algorithm>
#include <utility>

namespace lm {
namespace base {

float ScoreSentence(const base::Model *model, const char *sentence) {
  // TODO: reduce virtual dispatch to one per sentence?
  const base::Vocabulary &vocab = model->BaseVocabulary();
  // We know it's going to be a KenLM State.
  lm::ngram::State state_vec[2];
  lm::ngram::State *state = &state_vec[0];
  lm::ngram::State *state2 = &state_vec[1];
  model->BeginSentenceWrite(state);
  float ret = 0.0;
  for (util::TokenIter<util::BoolCharacter, true> i(sentence, util::kSpaces); i; ++i) {
    lm::WordIndex index = vocab.Index(*i);
    ret += model->BaseScore(state, index, state2);
    std::swap(state, state2);
  }
  ret += model->BaseScore(state, vocab.EndSentence(), state2);
  return ret;
}

} // namespace base
} // namespace lm
