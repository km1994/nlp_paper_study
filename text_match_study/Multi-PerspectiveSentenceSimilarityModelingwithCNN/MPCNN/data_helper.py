import codecs
from nltk.tokenize import word_tokenize
import numpy as np
import os


def load_sts(dsfile, glove, skip_unlabeled=True):
    """ load a dataset in the sts tsv format """
    s0 = []
    s1 = []
    labels = []
    with codecs.open(dsfile, encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            label, s0x, s1x = line.split('\t')
            if label == '':
                continue
            else:
                score_int = int(round(float(label)))
                y = [0] * 6
                y[score_int] = 1
                labels.append(np.array(y))
            for i, ss in enumerate([s0x, s1x]):
                words = word_tokenize(ss)
                index = []
                for word in words:
                    word = word.lower()
                    if word in glove.w:
                        index.append(glove.w[word])
                    else:
                        index.append(glove.w['UKNOW'])
                left = 100 - len(words)
                pad = [0]*left
                index.extend(pad)
                if i == 0:
                    s0.append(np.array(index))
                else:
                    s1.append(np.array(index))
            #s0.append(word_tokenize(s0x))
            #s1.append(word_tokenize(s1x))
    print(len(s0))
    return (s0, s1, labels)


def concat_datasets(datasets):
    """ Concatenate multiple loaded datasets into a single large one.

    Example: s0, s1, lab = concat_datasets([load_sts(d) for glob.glob('data/sts/semeval-sts/all/201[0-4]*')]) """
    s0 = []
    s1 = []
    labels = []
    for s0x, s1x, labelsx in datasets:
        s0 += s0x
        s1 += s1x
        labels += labelsx
    return (np.array(s0), np.array(s1), np.array(labels))


def load_embedded(glove, s0, s1, labels, ndim=0, s0pad=25, s1pad=60):
    """ Post-process loaded (s0, s1, labels) by mapping it to embeddings,
    plus optionally balancing (if labels are binary) and optionally not
    averaging but padding and returning full-sequence matrices.

    Note that this is now deprecated, especially if you use Keras - use the
    vocab.Vocabulary class. """

    if ndim == 1:
        # for averaging:
        e0 = np.array(glove.map_set(s0, ndim=1))
        e1 = np.array(glove.map_set(s1, ndim=1))
    else:
        # for padding and sequences (e.g. keras RNNs):
        # print('(%s) s0[-1000]: %d tokens' % (globmask, np.sort([np.shape(s) for s in s0], axis=0)[-1000]))
        # print('(%s) s1[-1000]: %d tokens' % (globmask, np.sort([np.shape(s) for s in s1], axis=0)[-1000]))
        e0 = glove.pad_set(glove.map_set(s0), s0pad)
        e1 = glove.pad_set(glove.map_set(s1), s1pad)
    return (e0, e1, s0, s1, labels)


def load_set(glove, path):
    files = []
    for file in os.listdir(path):
        if os.path.isfile(path + '/' + file):
            files.append(path + '/' + file)
    s0, s1, labels = concat_datasets([load_sts(d, glove) for d in files])
    #s0, s1, labels = np.array(s0), np.array(s1), np.array(labels)
    print('(%s) Loaded dataset: %d' % (path, len(s0)))
    #e0, e1, s0, s1, labels = load_embedded(glove, s0, s1, labels)
    return ([s0, s1], labels)


def get_embedding():
    gfile_path = os.path.join("./glove.6B", "glove.6B.300d.txt")
    f = open(gfile_path, 'r')
    embeddings = {}
    for line in f:
        sp_value = line.split()
        word = sp_value[0]
        embedding = [float(value) for value in sp_value[1:]]
        embeddings[word] = embedding
    print("read word2vec finished!")
    f.close()
    return embeddings

#load_sts('./sts/semeval-sts/2016/answer-answer.test.tsv')


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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