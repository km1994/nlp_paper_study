from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

def get_char_vocab(input_filenames, output_filename):
  vocab = set()
  for filename in input_filenames:
    with open(filename) as f:
      for line in f.readlines():
        for sentence in json.loads(line)["sentences"]:
          for word in sentence:
            vocab.update(word)
  vocab = sorted(list(vocab))
  with open(output_filename, "w") as f:
    for char in vocab:
      f.write(u"{}\n".format(char).encode("utf8"))
  print("Wrote {} characters to {}".format(len(vocab), output_filename))


if __name__ == "__main__":
  get_char_vocab(sys.argv[2:],"char_vocab.%s.txt"%sys.argv[1])
