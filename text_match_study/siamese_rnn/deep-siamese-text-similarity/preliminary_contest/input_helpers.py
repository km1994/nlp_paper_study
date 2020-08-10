# coding=utf-8
import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
# from gensim.models.word2vec import Word2Vec
import gensim
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys
import jieba

reload(sys)
sys.setdefaultencoding("utf-8")


class InputHelper(object):
    pre_emb = dict()
    vocab_processor = None

    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from " + filepath)
        x1 = []
        x2 = []
        for line in open(filepath):
            l = line.strip().split("\t")
            x1.append(l[1])
            x2.append(l[2])
        return np.asarray(x1), np.asarray(x2)

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    # Data Preparatopn
    # ==================================================

    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp, x2_temp = self.getTsvTestData(data_path)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print ('len(vocab_processor.vocabulary_)', len(vocab_processor.vocabulary_))
        # sys.exit(0)

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2
