// Score an entire sentence splitting on whitespace.  This should not be needed
// for C++ users (who should do it themselves), but it's faster for python users.
#pragma once

namespace lm {
namespace base {

class Model;

float ScoreSentence(const Model *model, const char *sentence);

} // namespace base
} // namespace lm
