# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import six
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile
from tensorflow.contrib import learn  # pylint: disable=g-bad-import-order
import jieba


def tokenizer_word(iterator):
    jieba.load_userdict('./dict.txt')
    for sentence in iterator:
        sentence = sentence.decode("utf8")
        sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。：？?、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),
                          sentence)
        yield list(jieba.lcut(sentence))


class MyVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None):

        tokenizer_fn = tokenizer_word
        self.sup = super(MyVocabularyProcessor, self)
        self.sup.__init__(max_document_length, min_frequency, vocabulary, tokenizer_fn)

    def transform(self, raw_documents):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        # print('len(raw_documents)= {}'.format(len(raw_documents)))
        # print('raw_documents= {}'.format(raw_documents))

        # for index,value in enumerate(raw_documents):
        #     print(index, value)

        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids
