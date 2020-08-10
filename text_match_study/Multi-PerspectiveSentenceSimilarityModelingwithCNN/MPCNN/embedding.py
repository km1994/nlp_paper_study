"""
A toolkit you may find useful for mapping sentences to embeddings.

Download and unzip the standard GloVe embeddings to use this.

Skip-thoughts use unigram/bigram information from the Children Book dataset.
"""


from __future__ import print_function

import numpy as np


class Embedder(object):
    """ Generic embedding interface.

    Required:
      * w: dict mapping tokens to indices
      * g: matrix with one row per token index
      * N: embedding dimensionality
    """

    def map_tokens(self, tokens, ndim=2):
        """ for the given list of tokens, return a list of GloVe embeddings,
        or a single plain bag-of-words average embedding if ndim=1.

        Unseen words (that's actually *very* rare) are mapped to 0-vectors. """
        gtokens = [self.g[self.w[t]] for t in tokens if t in self.w]
        if not gtokens:
            return np.zeros((1, self.N)) if ndim == 2 else np.zeros(self.N)
        gtokens = np.array(gtokens)
        if ndim == 2:
            return gtokens
        else:
            return gtokens.mean(axis=0)

    def map_set(self, ss, ndim=2):
        """ apply map_tokens on a whole set of sentences """
        return [self.map_tokens(s, ndim=ndim) for s in ss]

    def map_jset(self, sj):
        """ for a set of sentence emb indices, get per-token embeddings """
        return self.g[sj]

    def pad_set(self, ss, spad, N=None):
        """ Given a set of sentences transformed to per-word embeddings
        (using glove.map_set()), convert them to a 3D matrix with fixed
        sentence sizes - padded or trimmed to spad embeddings per sentence.

        Output is a tensor of shape (len(ss), spad, N).

        To determine spad, use something like
            np.sort([np.shape(s) for s in s0], axis=0)[-1000]
        so that typically everything fits, but you don't go to absurd lengths
        to accomodate outliers.
        """
        ss2 = []
        if N is None:
            N = self.N
        for s in ss:
            if spad > s.shape[0]:
                if s.ndim == 2:
                    s = np.vstack((s, np.zeros((spad - s.shape[0], N))))
                else:  # pad non-embeddings (e.g. toklabels) too
                    s = np.hstack((s, np.zeros(spad - s.shape[0])))
            elif spad < s.shape[0]:
                s = s[:spad]
            ss2.append(s)
        return np.array(ss2)


class GloVe(Embedder):
    """ A GloVe dictionary and the associated N-dimensional vector space """
    def __init__(self, N=50, glovepath='D:/project/python_wp/nlp/data/glove.6B/glove.6B.%dd.txt'):
        """ Load GloVe dictionary from the standard distributed text file.

        Glovepath should contain %d, which is substituted for the embedding
        dimension N. """
        self.N = N
        self.w = dict()
        self.g = []
        self.glovepath = glovepath % (N,)

        # [0] must be a zero vector
        self.g.append(np.zeros(self.N))

        with open(self.glovepath, 'r', encoding='utf-8') as f:
            for line in f:
                l = line.split()
                word = l[0]
                self.w[word] = len(self.g)
                self.g.append(np.array(l[1:]).astype(float))
        self.w['UKNOW'] = len(self.g)
        self.g.append(np.zeros(self.N))
        self.g = np.array(self.g, dtype='float32')
